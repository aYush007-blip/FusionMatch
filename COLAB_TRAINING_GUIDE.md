# Google Colab Multi-Stage Training & Metrics Guide (T4 GPU)

To avoid **GPU timeout, memory leaks, or Colab free-tier compute unit exhaustion**, the training lifecycle is divided into **3 modular, independently executable stages** with **Google Drive checkpoint resumption**.

📁 **Master Notebook**: [`notebooks/FusionMatch_Colab_Master.ipynb`](file:///c:/Users/HP/Desktop/Project/Deduplication/notebooks/FusionMatch_Colab_Master.ipynb)

---

## 1. Why Multi-Stage Training is Safer on Colab Free Tier

1. **Zero Progress Loss**: If your browser closes or Colab disconnects, each stage has already saved its best weights directly to your Google Drive (`MyDrive/FusionMatch/artifacts/checkpoints/`).
2. **Flexible Multi-Session Execution**: You can run Stage 1 today (15 min), Stage 2 tomorrow (30 min), and Stage 3 later without starting over!
3. **Periodic VRAM Flushing**: `torch.cuda.empty_cache()` is called automatically between batches and validation steps to prevent CUDA Out-of-Memory (OOM).

---

## 2. The 3 Modular Training Stages

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Warm-Up Training (Epochs 1–5)                                       │
│ • Backbone: Frozen                                                          │
│ • Trainable: Gated Fusion + Projection Head (~1.7M params)                  │
│ • Duration: ~15 minutes on T4 GPU                                           │
│ • Checkpoint Saved: /content/drive/MyDrive/.../checkpoints/warmup_best.pt   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Resumes from warmup_best.pt)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Fine-Tuning Chunk 1 (Epochs 6–10)                                  │
│ • Backbone: Unfreezes last 2 transformer blocks (~30M params)                │
│ • Hard-Negative Mining: Active                                              │
│ • Duration: ~30 minutes on T4 GPU                                           │
│ • Checkpoint Saved: /content/drive/MyDrive/.../checkpoints/finetune_phase1.pt│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Resumes from finetune_phase1.pt)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Fine-Tuning Chunk 2 (Epochs 11–15, Full Convergence)               │
│ • Full convergence with discriminative learning rates                       │
│ • Duration: ~30 minutes on T4 GPU                                           │
│ • Final Checkpoint Saved: /content/drive/MyDrive/.../checkpoints/best.pt    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Execution in the Master Notebook

### Step 1–3: Environment, Workspace & Fast S3 Download
- Mounts Google Drive (`/content/drive/`).
- Downloads dataset directly to `/content/data/raw/` (Colab local NVMe SSD) in **~45 seconds** and extracts in **~1.5 minutes**.

### Step 4: Stratified Partitioning
- Generates 80/10/10 zero-leakage SKU split (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`).

### Step 5A: Run Stage 1 (Epochs 1–5)
```python
trainer = FusionMatchTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config={"lr_head": 2e-5, "lr_backbone": 2e-6, "temperature": 0.07, "checkpoint_dir": str(checkpoint_dir)}, device=device)
history_warmup = trainer.train_warmup_phase(epochs=5, checkpoint_name="warmup_best.pt")
```
*Run time: ~15 minutes.* Saves `warmup_best.pt` to Drive.

### Step 5B: Run Stage 2 (Epochs 6–10)
```python
history_finetune1 = trainer.train_finetune_phase(start_epoch=6, end_epoch=10, resume_from=checkpoint_dir / "warmup_best.pt", checkpoint_name="finetune_phase1_best.pt")
```
*Run time: ~30 minutes.* Automatically unfreezes last 2 transformer blocks and mines in-category hard negatives. Saves `finetune_phase1_best.pt` to Drive.

### Step 5C: Run Stage 3 (Epochs 11–15)
```python
history_finetune2 = trainer.train_finetune_phase(start_epoch=11, end_epoch=15, resume_from=checkpoint_dir / "finetune_phase1_best.pt", checkpoint_name="best.pt")
```
*Run time: ~30 minutes.* Fine-tunes to final convergence. Saves `best.pt` to Drive.

---

## 4. Post-Training Steps (Run anytime on CPU or GPU)

- **Step 6**: Plot and save `training_curves.png` and `training_history.csv` to Drive.
- **Step 7**: Evaluate Pairwise F1 and Precision/Recall on the unseen Test split.
- **Step 8**: Build FAISS compressed `IndexIVFPQ` ($\sim 1.2\text{ MB}$) and benchmark `nprobe`.
- **Step 9**: Calibrate category-specific decision thresholds $\to$ `thresholds.json`.
- **Step 10**: Export dynamic INT8 quantized ONNX model $\to$ `fusion_match_int8.onnx`.
- **Step 11**: Generate `final_summary_report.csv` on your Drive.
