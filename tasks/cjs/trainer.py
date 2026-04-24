"""RL trainer for Cluster Job Scheduling (Decima-style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from distillation import cwr_initialise, rl_policy_distillation_loss
from distillation.losses import gae_advantages, ppo_surrogate_loss
from models import (
    GNNEncoder,
    HybridMambaStudent,
    ModalityProjector,
    SchedulingPolicyHead,
    TeacherLLM,
)
from models.hybrid_mamba import StudentConfig
from models.teacher import TeacherConfig
from utils import AverageMeter, build_optimizer, count_parameters

from .env import ClusterSchedulingEnv


# ---------------------------------------------------------------------------
class CJSAgent(nn.Module):
    def __init__(self, cfg: dict[str, Any], with_teacher: bool = True) -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        stu_cfg = cfg["student"]
        head_cfg = cfg["head"]

        self.gnn = GNNEncoder(
            enc_cfg["node_feat_dim"], enc_cfg["gnn_hidden"],
            enc_cfg["gnn_layers"], enc_cfg["gnn_type"],
        )
        # Student: pool node embeddings + global scalar features as tokens
        dims = {"nodes_pooled": self.gnn.out_dim, "global": 3}
        self.projector_student = ModalityProjector(dims, stu_cfg["embed_dim"])
        self.student = HybridMambaStudent(StudentConfig(
            embed_dim=stu_cfg["embed_dim"],
            num_mamba_layers=stu_cfg["num_mamba_layers"],
            num_transformer_layers=stu_cfg["num_transformer_layers"],
            transformer_position=stu_cfg["transformer_position"],
            num_heads=stu_cfg["num_heads"],
            ffn_mult=stu_cfg["ffn_mult"],
            dropout=stu_cfg["dropout"],
            ssm_d_state=stu_cfg["ssm_d_state"],
            ssm_d_conv=stu_cfg["ssm_d_conv"],
            ssm_expand=stu_cfg["ssm_expand"],
        ))
        # Map node embeddings (gnn_hidden) to student embed_dim for the
        # per-node stage head.
        self.node_to_student = nn.Linear(self.gnn.out_dim, stu_cfg["embed_dim"])
        self.student_head = SchedulingPolicyHead(
            stu_cfg["embed_dim"], head_cfg["num_executor_choices"],
        )

        self.with_teacher = with_teacher
        if with_teacher:
            tcfg = cfg["teacher"]
            self.teacher = TeacherLLM(TeacherConfig(
                name_or_path=tcfg["name_or_path"],
                use_lora=tcfg.get("use_lora", True),
                lora_rank=tcfg.get("lora_rank", 16),
                lora_alpha=tcfg.get("lora_alpha", 32),
                lora_dropout=tcfg.get("lora_dropout", 0.05),
                lora_target_modules=tuple(tcfg.get("lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            ))
            self.projector_teacher = ModalityProjector(dims, self.teacher.hidden_size)
            self.node_to_teacher = nn.Linear(self.gnn.out_dim, self.teacher.hidden_size)
            self.teacher_head = SchedulingPolicyHead(
                self.teacher.hidden_size, head_cfg["num_executor_choices"],
            )
        else:
            self.teacher = None

    # ------------------------------------------------------------------
    def forward(
        self,
        node_feat: torch.Tensor,         # (N, F)
        edge_index: torch.Tensor,        # (2, E)
        global_feat: torch.Tensor,       # (G,)
        frontier_mask: torch.Tensor,     # (N,)
    ) -> dict[str, torch.Tensor]:
        h_nodes = self.gnn(node_feat, edge_index)
        pooled = h_nodes.mean(dim=0, keepdim=True)           # (1, H)

        feats = {
            "nodes_pooled": pooled.unsqueeze(1),             # (1,1,H)
            "global": global_feat.view(1, 1, -1),
        }
        h_s = self.student(self.projector_student(feats))
        node_s = self.node_to_student(h_nodes)
        s_out = self.student_head(node_s, h_s, frontier_mask)
        out = {
            "stage_logits": s_out["stage_logits"],
            "exec_logits": s_out["exec_logits"].squeeze(0),
            "value": s_out["value"],
        }

        if self.teacher is not None:
            e_t = self.projector_teacher(feats)
            t_out = self.teacher(inputs_embeds=e_t, output_hidden_states=True)
            node_t = self.node_to_teacher(h_nodes)
            t_head = self.teacher_head(node_t, t_out["last_hidden_state"], frontier_mask)
            out.update({
                "teacher_stage_logits": t_head["stage_logits"],
                "teacher_exec_logits": t_head["exec_logits"].squeeze(0),
            })
        return out


# ---------------------------------------------------------------------------
class CJSTrainer:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.agent = CJSAgent(cfg, with_teacher=True).to(self.device)

        if cfg.get("cwr", {}).get("enabled", False):
            info = cwr_initialise(self.agent.student, self.agent.teacher.attention_weights, rank=cfg["cwr"].get("rank", 64))
            print(f"[CWR] warm-started: {info}")

        self.env = ClusterSchedulingEnv(cfg["data"].get("traces"))
        self._setup_optim()
        print(f"[params] student trainable={count_parameters(self.agent.student):,}")

    # ------------------------------------------------------------------
    def _setup_optim(self) -> None:
        teacher, student = [], []
        for n, p in self.agent.named_parameters():
            if not p.requires_grad:
                continue
            (teacher if "teacher" in n else student).append(p)
        self.opt_student = build_optimizer(student, self.cfg["train"])
        self.opt_teacher = build_optimizer(teacher, {**self.cfg["train"], "lr": self.cfg["train"]["teacher_lr"]}) if teacher else None

    # ------------------------------------------------------------------
    def _state_to_device(self, state):
        return (
            state.node_feat.to(self.device),
            state.edge_index.to(self.device),
            state.global_feat.to(self.device),
            state.frontier_mask.to(self.device),
        )

    # ------------------------------------------------------------------
    def _rollout(self, length: int):
        buffer = []
        state = self.env.reset()
        for _ in range(length):
            inp = self._state_to_device(state)
            with torch.no_grad():
                out = self.agent(*inp)
            stage_probs = F.softmax(out["stage_logits"], dim=-1)
            stage_dist = torch.distributions.Categorical(probs=stage_probs + 1e-10)
            stage_a = stage_dist.sample()
            exec_dist = torch.distributions.Categorical(logits=out["exec_logits"])
            exec_a = exec_dist.sample()
            logp = stage_dist.log_prob(stage_a) + exec_dist.log_prob(exec_a)

            nxt, r, done, _ = self.env.step(int(stage_a.item()), int(exec_a.item()))
            buffer.append({
                "state": state, "stage_a": stage_a.item(), "exec_a": exec_a.item(),
                "logp": logp.detach(), "value": out["value"].detach(),
                "reward": r, "done": done,
                "teacher_stage_logits": out.get("teacher_stage_logits", None),
                "teacher_exec_logits": out.get("teacher_exec_logits", None),
            })
            state = self.env.reset() if done else nxt
        return buffer, state

    # ------------------------------------------------------------------
    def _update(self, buffer: list[dict], last_state) -> dict[str, float]:
        cfg = self.cfg
        dcfg = cfg["distillation"]
        tcfg = cfg["train"]
        device = self.device

        rewards = torch.tensor([b["reward"] for b in buffer], device=device)
        dones = torch.tensor([b["done"] for b in buffer], device=device)
        values = torch.cat([b["value"].view(1) for b in buffer])
        with torch.no_grad():
            bootstrap = self.agent(*self._state_to_device(last_state))["value"].view(1)
        vals_full = torch.cat([values, bootstrap])
        adv, ret = gae_advantages(rewards, vals_full, dones, tcfg["gamma"], tcfg["gae_lambda"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pi_meter, v_meter, kl_meter = AverageMeter(), AverageMeter(), AverageMeter()
        for _ in range(tcfg["ppo_epochs"]):
            new_logps, new_vals, stage_kl_list, exec_kl_list = [], [], [], []
            for b in buffer:
                out = self.agent(*self._state_to_device(b["state"]))
                stage_dist = torch.distributions.Categorical(logits=out["stage_logits"])
                exec_dist = torch.distributions.Categorical(logits=out["exec_logits"])
                logp = stage_dist.log_prob(torch.tensor(b["stage_a"], device=device)) + \
                       exec_dist.log_prob(torch.tensor(b["exec_a"], device=device))
                new_logps.append(logp)
                new_vals.append(out["value"].view(1))
                if b["teacher_stage_logits"] is not None:
                    # align shape: teacher and student logits both (N,)
                    t_stage = b["teacher_stage_logits"]
                    s_stage = out["stage_logits"]
                    k = min(t_stage.size(0), s_stage.size(0))
                    stage_kl_list.append(rl_policy_distillation_loss(
                        s_stage[:k].unsqueeze(0), t_stage[:k].unsqueeze(0),
                        temperature=dcfg["temperature"],
                    ))
                    exec_kl_list.append(rl_policy_distillation_loss(
                        out["exec_logits"].unsqueeze(0),
                        b["teacher_exec_logits"].unsqueeze(0),
                        temperature=dcfg["temperature"],
                    ))
            new_logp = torch.stack(new_logps)
            new_val = torch.cat(new_vals)
            old_logp = torch.stack([b["logp"] for b in buffer])
            pi_loss = ppo_surrogate_loss(new_logp, old_logp, adv, tcfg["clip_ratio"])
            v_loss = F.mse_loss(new_val, ret)
            if stage_kl_list:
                kl = torch.stack(stage_kl_list).mean() + torch.stack(exec_kl_list).mean()
            else:
                kl = torch.tensor(0.0, device=device)
            loss = pi_loss + tcfg["value_coef"] * v_loss + dcfg["beta"] * kl

            self.opt_student.zero_grad(set_to_none=True)
            if self.opt_teacher is not None:
                self.opt_teacher.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.agent.parameters() if p.requires_grad], tcfg["grad_clip"])
            self.opt_student.step()
            if self.opt_teacher is not None:
                self.opt_teacher.step()

            pi_meter.update(pi_loss.item()); v_meter.update(v_loss.item()); kl_meter.update(float(kl))
        return {"pi": pi_meter.avg, "v": v_meter.avg, "kl": kl_meter.avg}

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_jct = float("inf")
        for ep in tqdm(range(cfg["train"]["episodes"])):
            buf, last_s = self._rollout(cfg["train"]["rollout_len"])
            stats = self._update(buf, last_s)
            if ep % cfg["train"]["log_interval"] == 0:
                tqdm.write(f"ep {ep} | π={stats['pi']:.3f} v={stats['v']:.3f} kl={stats['kl']:.3f}")
            if ep % cfg["train"]["eval_interval"] == 0 and ep > 0:
                jct = self.evaluate()
                tqdm.write(f"eval JCT={jct:.2f}")
                if jct < best_jct:
                    best_jct = jct
                    torch.save({"agent": self.agent.state_dict(), "cfg": cfg, "ep": ep, "jct": jct}, ckpt_dir / "best.pt")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, episodes: int = 5) -> float:
        self.agent.eval()
        jcts = []
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                out = self.agent(*self._state_to_device(state))
                stage_a = int(torch.argmax(out["stage_logits"]).item())
                exec_a = int(torch.argmax(out["exec_logits"]).item())
                state, _, done, _ = self.env.step(stage_a, exec_a)
            jcts.append(self.env.jct())
        return float(np.mean(jcts))
