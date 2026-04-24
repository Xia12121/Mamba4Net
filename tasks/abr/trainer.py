"""PPO trainer with RL policy distillation (Eqs. 6-8) for ABR."""

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
    ABRScalarEncoder,
    BitratePolicyHead,
    HybridMambaStudent,
    ModalityProjector,
    TeacherLLM,
)
from models.hybrid_mamba import StudentConfig
from models.teacher import TeacherConfig
from utils import AverageMeter, build_optimizer, count_parameters

from .env import ABREnv


# ---------------------------------------------------------------------------
class ABRAgent(nn.Module):
    def __init__(self, cfg: dict[str, Any], with_teacher: bool = True) -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        stu_cfg = cfg["student"]
        head_cfg = cfg["head"]

        self.encoder = ABRScalarEncoder(
            enc_cfg["throughput_dim"], enc_cfg["delay_dim"], enc_cfg["chunk_size_dim"],
            enc_cfg.get("buffer_dim", 1), enc_cfg["hidden"],
        )
        dims = {"abr_tokens": self.encoder.out_dim}
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
        self.student_head = BitratePolicyHead(stu_cfg["embed_dim"], head_cfg["num_bitrates"])

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
            self.teacher_head = BitratePolicyHead(self.teacher.hidden_size, head_cfg["num_bitrates"])
        else:
            self.teacher = None

    # ------------------------------------------------------------------
    def _encode(self, state_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self.encoder(
            state_batch["throughput"], state_batch["delay"],
            state_batch["chunk_sizes"], state_batch["buffer"],
        )
        return {"abr_tokens": tokens}

    def forward(self, state_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self._encode(state_batch)
        h_s = self.student(self.projector_student(feats))
        s_out = self.student_head(h_s)
        out = {"student_logits": s_out["logits"], "student_value": s_out["value"], "student_probs": s_out["probs"]}
        if self.teacher is not None:
            e_t = self.projector_teacher(feats)
            t_out = self.teacher(inputs_embeds=e_t, output_hidden_states=True)
            t_head = self.teacher_head(t_out["last_hidden_state"])
            out.update({"teacher_logits": t_head["logits"], "teacher_value": t_head["value"]})
        return out


# ---------------------------------------------------------------------------
def _state_to_tensor(state, device) -> dict[str, torch.Tensor]:
    return {
        "throughput": torch.as_tensor(state.past_throughput, device=device).unsqueeze(0).float(),
        "delay": torch.as_tensor(state.past_delay, device=device).unsqueeze(0).float(),
        "chunk_sizes": torch.as_tensor(state.chunk_sizes, device=device).unsqueeze(0).float(),
        "buffer": torch.as_tensor([[state.buffer]], device=device).float(),
    }


# ---------------------------------------------------------------------------
class ABRTrainer:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.agent = ABRAgent(cfg, with_teacher=True).to(self.device)
        if cfg.get("cwr", {}).get("enabled", False):
            info = cwr_initialise(self.agent.student, self.agent.teacher.attention_weights, rank=cfg["cwr"].get("rank", 64))
            print(f"[CWR] warm-started: {info}")

        self.env = ABREnv(cfg["data"].get("traces"))

        student_params, teacher_params = self._param_groups()
        self.opt_student = build_optimizer(student_params, cfg["train"])
        self.opt_teacher = build_optimizer(teacher_params, {**cfg["train"], "lr": cfg["train"]["teacher_lr"]}) if teacher_params else None
        print(f"[params] student trainable={count_parameters(self.agent.student):,}")

    # ------------------------------------------------------------------
    def _param_groups(self):
        teacher, student = [], []
        for n, p in self.agent.named_parameters():
            if not p.requires_grad:
                continue
            (teacher if "teacher" in n else student).append(p)
        return student, teacher

    # ------------------------------------------------------------------
    def _rollout(self, length: int) -> dict[str, torch.Tensor]:
        states, actions, logprobs, values, rewards, dones = [], [], [], [], [], []
        teacher_logits = []
        state = self.env.reset()
        self.agent.eval()
        for _ in range(length):
            s_tensor = _state_to_tensor(state, self.device)
            with torch.no_grad():
                out = self.agent(s_tensor)
            dist = torch.distributions.Categorical(logits=out["student_logits"])
            a = dist.sample()
            logp = dist.log_prob(a)
            nxt, r, done, _ = self.env.step(int(a.item()))
            states.append(s_tensor)
            actions.append(a)
            logprobs.append(logp)
            values.append(out["student_value"])
            teacher_logits.append(out["teacher_logits"].detach() if "teacher_logits" in out else None)
            rewards.append(r)
            dones.append(done)
            state = self.env.reset() if done else nxt

        # Bootstrap value
        with torch.no_grad():
            bootstrap = self.agent(_state_to_tensor(state, self.device))["student_value"]
        values_with_bootstrap = torch.cat(values + [bootstrap], dim=0)
        rewards_t = torch.tensor(rewards, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        advantages, returns = gae_advantages(
            rewards_t, values_with_bootstrap.detach(), dones_t,
            gamma=self.cfg["train"]["gamma"], lam=self.cfg["train"]["gae_lambda"],
        )

        return {
            "states": states,
            "actions": torch.stack(actions).squeeze(-1),
            "old_logprobs": torch.stack(logprobs).squeeze(-1).detach(),
            "advantages": advantages.detach(),
            "returns": returns.detach(),
            "teacher_logits": [tl for tl in teacher_logits if tl is not None],
        }

    # ------------------------------------------------------------------
    def _update(self, rollout: dict[str, Any]) -> dict[str, float]:
        cfg = self.cfg
        dcfg = cfg["distillation"]
        tcfg = cfg["train"]
        all_states = rollout["states"]
        actions = rollout["actions"]
        old_logp = rollout["old_logprobs"]
        adv = rollout["advantages"]
        ret = rollout["returns"]
        teacher_logits_list = rollout["teacher_logits"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pi_loss_m, v_loss_m, kl_m = AverageMeter(), AverageMeter(), AverageMeter()
        self.agent.train()
        for _ in range(tcfg["ppo_epochs"]):
            # Recompute logits/values in a single forward per state
            new_logp_list, new_val_list, new_logits_list = [], [], []
            for s in all_states:
                out = self.agent(s)
                new_logits_list.append(out["student_logits"])
                new_val_list.append(out["student_value"])
            new_logits = torch.cat(new_logits_list, dim=0)
            new_vals = torch.cat(new_val_list, dim=0)
            dist = torch.distributions.Categorical(logits=new_logits)
            new_logp = dist.log_prob(actions)

            pi_loss = ppo_surrogate_loss(new_logp, old_logp, adv, tcfg["clip_ratio"])
            v_loss = F.mse_loss(new_vals, ret)
            ent = dist.entropy().mean()

            # RL policy distillation — Eqs. 7-8
            if teacher_logits_list and len(teacher_logits_list) == new_logits.size(0):
                tl = torch.cat(teacher_logits_list, dim=0)
                kl = rl_policy_distillation_loss(new_logits, tl, temperature=dcfg["temperature"])
            else:
                kl = torch.tensor(0.0, device=self.device)

            loss = pi_loss + tcfg["value_coef"] * v_loss - tcfg["entropy_coef"] * ent + dcfg["beta"] * kl

            self.opt_student.zero_grad(set_to_none=True)
            if self.opt_teacher is not None:
                self.opt_teacher.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.agent.parameters() if p.requires_grad],
                tcfg["grad_clip"],
            )
            self.opt_student.step()
            if self.opt_teacher is not None:
                self.opt_teacher.step()

            pi_loss_m.update(pi_loss.item())
            v_loss_m.update(v_loss.item())
            kl_m.update(float(kl))

        return {"pi": pi_loss_m.avg, "v": v_loss_m.avg, "kl": kl_m.avg}

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_qoe = -float("inf")
        for ep in tqdm(range(cfg["train"]["episodes"])):
            rollout = self._rollout(cfg["train"]["rollout_len"])
            stats = self._update(rollout)
            if ep % cfg["train"]["log_interval"] == 0:
                tqdm.write(f"ep {ep} | π={stats['pi']:.3f} v={stats['v']:.3f} kl={stats['kl']:.3f}")
            if ep % cfg["train"]["eval_interval"] == 0 and ep > 0:
                qoe = self.evaluate()
                tqdm.write(f"eval QoE = {qoe:.3f}")
                if qoe > best_qoe:
                    best_qoe = qoe
                    torch.save({"agent": self.agent.state_dict(), "cfg": cfg, "ep": ep, "qoe": qoe}, ckpt_dir / "best.pt")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, episodes: int = 10) -> float:
        self.agent.eval()
        total = 0.0
        for _ in range(episodes):
            s = self.env.reset()
            ep_r = 0.0
            done = False
            while not done:
                out = self.agent(_state_to_tensor(s, self.device))
                a = int(torch.argmax(out["student_logits"], dim=-1).item())
                s, r, done, _ = self.env.step(a)
                ep_r += r
            total += ep_r
        return total / episodes
