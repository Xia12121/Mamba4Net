"""Supervised distillation trainer for the Viewport Prediction task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from distillation import cwr_initialise, supervised_distillation_loss
from models import (
    HybridMambaStudent,
    ModalityProjector,
    TeacherLLM,
    TimeSeriesEncoder,
    ViewportHead,
    ViTEncoder,
)
from models.hybrid_mamba import StudentConfig
from models.teacher import TeacherConfig
from utils import AverageMeter, build_optimizer, build_scheduler, count_parameters

from .dataset import ViewportDataset, collate_viewport


# ---------------------------------------------------------------------------
class ViewportAgent(nn.Module):
    """Wraps encoders + projector + student + head. A matching teacher
    replica is created lazily only when ``with_teacher=True``."""

    def __init__(self, cfg: dict[str, Any], with_teacher: bool = True) -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        stu_cfg = cfg["student"]
        head_cfg = cfg["head"]

        # Modality encoders
        self.ts_encoder = TimeSeriesEncoder(enc_cfg["ts_input_dim"], enc_cfg["ts_hidden"])
        self.vit_encoder = ViTEncoder(enc_cfg["vit_name"]) if enc_cfg.get("use_vit", False) else None

        dims = {"past_viewports": self.ts_encoder.out_dim}
        if self.vit_encoder is not None:
            dims["frames"] = self.vit_encoder.out_dim

        # Student projector + backbone
        self.projector_student = ModalityProjector(dims, stu_cfg["embed_dim"], enc_cfg["proj_dropout"])
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
        self.student_head = ViewportHead(
            stu_cfg["embed_dim"], head_cfg["output_dim"], head_cfg["future_horizon"]
        )

        # Teacher replica (shares encoders, has its own projector/head)
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
            self.projector_teacher = ModalityProjector(dims, self.teacher.hidden_size, enc_cfg["proj_dropout"])
            self.teacher_head = ViewportHead(
                self.teacher.hidden_size, head_cfg["output_dim"], head_cfg["future_horizon"]
            )
        else:
            self.teacher = None

    # ------------------------------------------------------------------
    def _encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = {"past_viewports": self.ts_encoder(batch["past_viewports"])}
        if self.vit_encoder is not None and "frames" in batch:
            feats["frames"] = self.vit_encoder(batch["frames"])
        return feats

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self._encode(batch)
        h_s = self.student(self.projector_student(feats))
        y_s = self.student_head(h_s)

        out: dict[str, torch.Tensor] = {"student_pred": y_s, "student_hidden": h_s}
        if self.teacher is not None:
            with torch.no_grad() if not self.training else torch.enable_grad():
                e_t = self.projector_teacher(feats)
                t_out = self.teacher(inputs_embeds=e_t, output_hidden_states=True)
                y_t = self.teacher_head(t_out["last_hidden_state"])
            out.update({"teacher_pred": y_t, "teacher_hidden": t_out["last_hidden_state"]})
        return out


# ---------------------------------------------------------------------------
class ViewportTrainer:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.agent = ViewportAgent(cfg, with_teacher=True).to(self.device)

        if cfg.get("cwr", {}).get("enabled", False):
            tw = self.agent.teacher.attention_weights
            info = cwr_initialise(self.agent.student, tw, rank=cfg["cwr"].get("rank", 64))
            print(f"[CWR] warm-started student blocks: {info}")

        # Data
        data_cfg = cfg["data"]
        enc_cfg = cfg["encoder"]
        self.train_set = ViewportDataset(
            data_cfg["root"], "train",
            window=enc_cfg["ts_context"], horizon=cfg["head"]["future_horizon"],
            use_vit=enc_cfg.get("use_vit", False),
        )
        self.val_set = ViewportDataset(
            data_cfg["root"], "val",
            window=enc_cfg["ts_context"], horizon=cfg["head"]["future_horizon"],
            use_vit=enc_cfg.get("use_vit", False),
        )
        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg["train"]["batch_size"],
            shuffle=True, collate_fn=collate_viewport, num_workers=2, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg["train"]["batch_size"],
            collate_fn=collate_viewport, num_workers=2,
        )

        # Two optimizers: one for the (student, student_head, encoders,
        # student_projector), one for the teacher LoRA parameters.
        student_params, teacher_params = self._param_groups()
        self.opt_student = build_optimizer(student_params, cfg["train"])
        teacher_train_cfg = {**cfg["train"], "lr": cfg["train"]["teacher_lr"]}
        self.opt_teacher = build_optimizer(teacher_params, teacher_train_cfg) if teacher_params else None

        total_steps = len(self.train_loader) * cfg["train"]["epochs"]
        self.sched_student = build_scheduler(self.opt_student, total_steps, cfg["train"])
        self.sched_teacher = build_scheduler(self.opt_teacher, total_steps, cfg["train"]) if self.opt_teacher else None

        print(f"[params] student={count_parameters(self.agent.student):,} "
              f"head={count_parameters(self.agent.student_head):,}")

    # ------------------------------------------------------------------
    def _param_groups(self) -> tuple[list, list]:
        teacher_params, student_params = [], []
        for name, p in self.agent.named_parameters():
            if not p.requires_grad:
                continue
            (teacher_params if "teacher" in name else student_params).append(p)
        return student_params, teacher_params

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        for epoch in range(cfg["train"]["epochs"]):
            self._run_epoch(epoch)
            val_mae = self.evaluate()
            print(f"[epoch {epoch:03d}] val MAE = {val_mae:.4f}")
            if val_mae < best_val:
                best_val = val_mae
                torch.save(
                    {"agent": self.agent.state_dict(), "cfg": cfg, "epoch": epoch, "val": val_mae},
                    ckpt_dir / "best.pt",
                )

    # ------------------------------------------------------------------
    def _run_epoch(self, epoch: int) -> None:
        cfg = self.cfg["distillation"]
        meter_loss = AverageMeter()
        self.agent.train()
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {epoch}")):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.agent(batch)

            loss_dict = supervised_distillation_loss(
                student_out=out["student_pred"],
                teacher_out=out.get("teacher_pred"),
                target=batch["future_viewports"],
                base_loss_fn=F.l1_loss,
                alpha=cfg["alpha"],
                temperature=cfg["temperature"],
                task="regression",
            )
            loss = loss_dict["loss"]
            self.opt_student.zero_grad(set_to_none=True)
            if self.opt_teacher is not None:
                self.opt_teacher.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.agent.parameters() if p.requires_grad],
                self.cfg["train"]["grad_clip"],
            )
            self.opt_student.step()
            self.sched_student.step()
            if self.opt_teacher is not None:
                self.opt_teacher.step()
                self.sched_teacher.step()
            meter_loss.update(loss.item())

            if step % self.cfg["train"]["log_interval"] == 0:
                tqdm.write(
                    f"step {step:5d} | loss {meter_loss.avg:.4f} | "
                    f"base {loss_dict['base']:.4f} | distill {loss_dict['distill']:.4f}"
                )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> float:
        self.agent.eval()
        meter = AverageMeter()
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.agent(batch)
            mae = F.l1_loss(out["student_pred"], batch["future_viewports"])
            meter.update(mae.item(), n=batch["future_viewports"].size(0))
        return meter.avg
