"""Two-Phase Training Engine for FusionMatch Contrastive Learning."""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from .losses import InfoNCELoss, HardNegativeMiner
from .metrics import compute_pairwise_f1, compute_precision_recall_at_k
from ..utils.io import save_checkpoint


class FusionMatchTrainer:
    """Orchestrates Two-Phase Training:
    
    1. Phase 1 (Warm-Up, Epochs 1-5):
       - Backbone is completely frozen.
       - Gated fusion module and projection head are trained at LR 2e-5.
    2. Phase 2 (Fine-Tuning, Epochs 6-15):
       - Last 2 transformer blocks of vision and text backbones are unfrozen.
       - Discriminative learning rates (2e-6 for backbone, 2e-5 for heads).
       - In-category hard-negative mining actively supplements in-batch negatives.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        default_cfg = {
            "warmup_epochs": 5,
            "finetune_epochs": 10,
            "total_epochs": 15,
            "lr_head": 2e-5,
            "lr_backbone": 2e-6,
            "weight_decay": 0.01,
            "temperature": 0.07,
            "hard_negative_start_epoch": 3,
            "unfreeze_last_n_blocks": 2,
            "embed_dim": 256,
            "checkpoint_dir": "artifacts/checkpoints",
            "val_threshold": 0.70,
            "max_grad_norm": 1.0,
        }
        if config:
            default_cfg.update(config)
        self.cfg = default_cfg

        self.criterion = InfoNCELoss(temperature=self.cfg["temperature"])
        self.miner = HardNegativeMiner(
            embed_dim=self.cfg["embed_dim"],
            device=self.device,
        )
        self.use_amp = self.device.startswith("cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = GradScaler(enabled=self.use_amp)
        self.best_val_f1 = -1.0
        self.history: List[Dict[str, Any]] = []

    def _build_optimizer(self, phase: str) -> torch.optim.Optimizer:
        """Builds AdamW optimizer with phase-specific parameter groups and learning rates."""
        if phase == "warmup":
            # Train only fusion and projection parameters
            params = [
                p for n, p in self.model.named_parameters()
                if "encoder.backbone" not in n and p.requires_grad
            ]
            return torch.optim.AdamW(
                params,
                lr=self.cfg["lr_head"],
                weight_decay=self.cfg["weight_decay"],
            )
        else:
            # Fine-tuning: discriminative learning rate groups
            head_params = [
                p for n, p in self.model.named_parameters()
                if "encoder.backbone" not in n and p.requires_grad
            ]
            backbone_params = [
                p for n, p in self.model.named_parameters()
                if "encoder.backbone" in n and p.requires_grad
            ]
            return torch.optim.AdamW(
                [
                    {"params": head_params, "lr": self.cfg["lr_head"]},
                    {"params": backbone_params, "lr": self.cfg["lr_backbone"]},
                ],
                weight_decay=self.cfg["weight_decay"],
            )

    def _run_epoch(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        phase: str,
        mine_hard_negatives: bool,
    ) -> float:
        """Runs a single training epoch with mixed precision and gradient clipping."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            num_batches += 1
            optimizer.zero_grad()

            # Handle pair sampler format or standard batch
            if "positive_pixel_values" in batch:
                # Pair / Triplet format
                anc_pixels = batch["anchor_pixel_values"].to(self.device)
                anc_ids = batch["anchor_input_ids"].to(self.device)
                anc_mask = batch.get("anchor_attention_mask")
                if anc_mask is not None:
                    anc_mask = anc_mask.to(self.device)
                anc_qv = batch.get("anchor_q_v")
                anc_qt = batch.get("anchor_q_t")

                pos_pixels = batch["positive_pixel_values"].to(self.device)
                pos_ids = batch["positive_input_ids"].to(self.device)
                pos_mask = batch.get("positive_attention_mask")
                if pos_mask is not None:
                    pos_mask = pos_mask.to(self.device)
                pos_qv = batch.get("positive_q_v")
                pos_qt = batch.get("positive_q_t")

                sku_ids = batch.get("anchor_sku_id", batch.get("sku_ids", batch.get("sku_id", [f"sku_{i}" for i in range(anc_pixels.size(0))])))
            else:
                # Standard batch
                anc_pixels = batch["pixel_values"].to(self.device)
                anc_ids = batch["input_ids"].to(self.device)
                anc_mask = batch.get("attention_mask")
                if anc_mask is not None:
                    anc_mask = anc_mask.to(self.device)
                anc_qv = batch.get("q_v")
                anc_qt = batch.get("q_t")
                pos_pixels, pos_ids, pos_mask, pos_qv, pos_qt = (
                    anc_pixels, anc_ids, anc_mask, anc_qv, anc_qt
                )
                sku_ids = batch.get("sku_ids", batch.get("sku_id", [f"sku_{i}" for i in range(anc_pixels.size(0))])))

            device_type = "cuda" if self.device.startswith("cuda") else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                anchor_emb, _ = self.model(
                    pixel_values=anc_pixels,
                    input_ids=anc_ids,
                    attention_mask=anc_mask,
                    q_v=anc_qv,
                    q_t=anc_qt,
                )
                positive_emb, _ = self.model(
                    pixel_values=pos_pixels,
                    input_ids=pos_ids,
                    attention_mask=pos_mask,
                    q_v=pos_qv,
                    q_t=pos_qt,
                )

                hard_neg_emb = None
                if mine_hard_negatives and self.miner.current_size >= 16:
                    hard_neg_emb = self.miner.mine(anchor_emb.detach(), sku_ids)

                loss = self.criterion(anchor_emb, positive_emb, hard_neg_emb)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
                optimizer.step()

            # Update negative miner bank
            self.miner.update(positive_emb.detach(), sku_ids)
            running_loss += loss.item()

            if self.device.startswith("cuda") and num_batches % 50 == 0:
                torch.cuda.empty_cache()

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluates model performance on the validation dataset."""
        self.model.eval()
        all_embeddings = []
        all_sku_ids = []

        for batch in self.val_loader:
            if "anchor_pixel_values" in batch:
                # Pair sampler format
                anc_p = batch["anchor_pixel_values"].to(self.device)
                anc_ids = batch["anchor_input_ids"].to(self.device)
                anc_mask = batch.get("anchor_attention_mask")
                if anc_mask is not None:
                    anc_mask = anc_mask.to(self.device)
                anc_qv = batch.get("anchor_q_v")
                anc_qt = batch.get("anchor_q_t")
                anc_skus = batch.get("anchor_sku_id", [])

                anc_emb, _ = self.model(
                    pixel_values=anc_p,
                    input_ids=anc_ids,
                    attention_mask=anc_mask,
                    q_v=anc_qv,
                    q_t=anc_qt,
                )
                all_embeddings.append(anc_emb.cpu())
                all_sku_ids.extend(anc_skus)

                # Also evaluate positive pairs
                if "positive_pixel_values" in batch:
                    pos_p = batch["positive_pixel_values"].to(self.device)
                    pos_ids = batch["positive_input_ids"].to(self.device)
                    pos_mask = batch.get("positive_attention_mask")
                    if pos_mask is not None:
                        pos_mask = pos_mask.to(self.device)
                    pos_qv = batch.get("positive_q_v")
                    pos_qt = batch.get("positive_q_t")
                    pos_skus = batch.get("positive_sku_id", anc_skus)

                    pos_emb, _ = self.model(
                        pixel_values=pos_p,
                        input_ids=pos_ids,
                        attention_mask=pos_mask,
                        q_v=pos_qv,
                        q_t=pos_qt,
                    )
                    all_embeddings.append(pos_emb.cpu())
                    all_sku_ids.extend(pos_skus)

            elif "pixel_values" in batch:
                # Standard flat batch format
                pixels = batch["pixel_values"].to(self.device)
                ids = batch["input_ids"].to(self.device)
                mask = batch.get("attention_mask")
                if mask is not None:
                    mask = mask.to(self.device)
                qv = batch.get("q_v")
                qt = batch.get("q_t")

                emb, _ = self.model(
                    pixel_values=pixels,
                    input_ids=ids,
                    attention_mask=mask,
                    q_v=qv,
                    q_t=qt,
                )
                all_embeddings.append(emb.cpu())
                skus = batch.get("sku_ids", batch.get("sku_id", []))
                all_sku_ids.extend(skus)

        if not all_embeddings:
            return {"val_f1": 0.0, "p@5": 0.0, "r@5": 0.0}

        embeddings = torch.cat(all_embeddings, dim=0)
        val_f1 = compute_pairwise_f1(embeddings, all_sku_ids, threshold=self.cfg["val_threshold"])
        p_at_5, r_at_5 = compute_precision_recall_at_k(embeddings, all_sku_ids, k=5)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "val_f1": val_f1,
            "p@5": p_at_5,
            "r@5": r_at_5,
        }

    def train_warmup_phase(
        self,
        epochs: int = 5,
        checkpoint_name: str = "warmup_best.pt",
    ) -> List[Dict[str, Any]]:
        """Executes Phase 1 Warm-up training with frozen backbone (Safe for quick Colab runs)."""
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"  STARTING STAGE 1: WARM-UP TRAINING (Epochs 1 - {epochs})")
        print(f"  Mode: Backbone Frozen | Trainable: Gated Fusion & Heads")
        print(f"=======================================================\n")

        self.model.encoder._set_trainable(
            freeze_vision=True,
            freeze_text=True,
            unfreeze_last_n_blocks=0,
        )
        optimizer = self._build_optimizer("warmup")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            mine_negs = epoch > self.cfg["hard_negative_start_epoch"]
            train_loss = self._run_epoch(epoch, optimizer, phase="warmup", mine_hard_negatives=mine_negs)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            epoch_log = {
                "epoch": epoch,
                "phase": "warmup",
                "train_loss": train_loss,
                "elapsed_s": elapsed,
                **val_metrics,
            }
            self.history.append(epoch_log)

            print(
                f"[Warm-Up {epoch:02d}/{epochs:02d}] "
                f"Loss: {train_loss:.4f} | Val F1: {val_metrics['val_f1']:.4f} | "
                f"P@5: {val_metrics['p@5']:.4f} | R@5: {val_metrics['r@5']:.4f} ({elapsed:.1f}s)"
            )

            if val_metrics["val_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_f1"]
                save_checkpoint(
                    self.model.state_dict(),
                    ckpt_dir / checkpoint_name,
                    metadata={"epoch": epoch, "phase": "warmup", "val_f1": self.best_val_f1},
                )
                save_checkpoint(
                    self.model.state_dict(),
                    ckpt_dir / "best.pt",
                    metadata={"epoch": epoch, "phase": "warmup", "val_f1": self.best_val_f1},
                )

        print(f"\nStage 1 Warm-Up Complete! Checkpoint saved to: {ckpt_dir / checkpoint_name}")
        return self.history

    def train_finetune_phase(
        self,
        start_epoch: int = 6,
        end_epoch: int = 10,
        resume_from: Optional[Union[str, Path]] = None,
        checkpoint_name: str = "finetune_best.pt",
    ) -> List[Dict[str, Any]]:
        """Executes Phase 2 Fine-Tuning with selective block unfreezing and checkpoint resumption."""
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if resume_from and Path(resume_from).exists():
            from ..utils.io import load_checkpoint
            ckpt = load_checkpoint(resume_from, device=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            if "metadata" in ckpt and "val_f1" in ckpt["metadata"]:
                self.best_val_f1 = max(self.best_val_f1, ckpt["metadata"]["val_f1"])
            print(f"Resumed model weights from: {resume_from} (Best F1 so far: {self.best_val_f1:.4f})")

        print(f"\n=================================================================")
        print(f"  STARTING STAGE 2: FINE-TUNING (Epochs {start_epoch} - {end_epoch})")
        print(f"  Mode: Unfreeze Last {self.cfg['unfreeze_last_n_blocks']} Blocks | In-Category Hard Negative Mining")
        print(f"=================================================================\n")

        self.model.encoder._set_trainable(
            freeze_vision=False,
            freeze_text=False,
            unfreeze_last_n_blocks=self.cfg["unfreeze_last_n_blocks"],
        )
        optimizer = self._build_optimizer("finetune")

        for epoch in range(start_epoch, end_epoch + 1):
            t0 = time.time()
            train_loss = self._run_epoch(epoch, optimizer, phase="finetune", mine_hard_negatives=True)
            val_metrics = self.validate()
            elapsed = time.time() - t0

            epoch_log = {
                "epoch": epoch,
                "phase": "finetune",
                "train_loss": train_loss,
                "elapsed_s": elapsed,
                **val_metrics,
            }
            self.history.append(epoch_log)

            print(
                f"[Fine-Tune {epoch:02d}/{end_epoch:02d}] "
                f"Loss: {train_loss:.4f} | Val F1: {val_metrics['val_f1']:.4f} | "
                f"P@5: {val_metrics['p@5']:.4f} | R@5: {val_metrics['r@5']:.4f} ({elapsed:.1f}s)"
            )

            if val_metrics["val_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_f1"]
                save_checkpoint(
                    self.model.state_dict(),
                    ckpt_dir / checkpoint_name,
                    metadata={"epoch": epoch, "phase": "finetune", "val_f1": self.best_val_f1},
                )
                save_checkpoint(
                    self.model.state_dict(),
                    ckpt_dir / "best.pt",
                    metadata={"epoch": epoch, "phase": "finetune", "val_f1": self.best_val_f1},
                )

        print(f"\nStage Fine-Tuning Complete! Checkpoint saved to: {ckpt_dir / checkpoint_name}")
        return self.history

    def fit(self, epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Executes the full two-phase training lifecycle sequentially."""
        warmup_epochs = self.cfg["warmup_epochs"]
        total_epochs = epochs or self.cfg["total_epochs"]

        self.train_warmup_phase(epochs=warmup_epochs, checkpoint_name="warmup_best.pt")
        if total_epochs > warmup_epochs:
            self.train_finetune_phase(
                start_epoch=warmup_epochs + 1,
                end_epoch=total_epochs,
                checkpoint_name="finetune_best.pt",
            )
        return self.history
