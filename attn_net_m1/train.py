"""
train.py -- Baseline training for attn_net_m1

Trains an Attention U-Net with bottleneck CBAM from random initialisation.

Training schedule
-----------------
  Optimiser : AdamW (lr=5e-5, weight_decay=1e-4)
  Schedule  : 10-epoch LinearLR warmup -> CosineAnnealingLR
  Epochs    : 300  (early-stop with patience=20 validation checks)
  Batch     : 2 patches per step, 400 steps per epoch
  Loss      : 0.3 * CE + 0.5 * Dice + 0.2 * ET_Dice

Usage
-----
    python train.py [--data_root PATH] [--save_dir PATH]
"""

import os
import argparse
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm

from .config  import TRAIN_CONFIG, MODEL_CONFIG
from .model   import AttentionUNet3D
from .dataset import Brain3DDataset, Brain3DValDataset
from .losses  import BaselineCriterion
from .metrics import dice_per_class
from .utils   import get_case_dirs, make_logger, validate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_training(cfg: dict) -> None:
    device = torch.device(cfg["device"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    log_path = os.path.join(cfg["save_dir"], "training_log.txt")
    log, log_f = make_logger(log_path)

    log("=" * 80)
    log("Attention U-Net M1 (Bottleneck CBAM) -- Baseline Training")
    log(f"Start: {datetime.datetime.now()}")
    log("=" * 80)
    for k, v in cfg.items():
        log(f"  {k}: {v}")
    log("=" * 80)

    # -- Data ----------------------------------------------------------------
    train_dirs, val_dirs = get_case_dirs(cfg["data_root"])
    val_subset_dirs      = val_dirs[:cfg["val_subset_size"]]

    log(f"Train cases: {len(train_dirs)} | Val cases: {len(val_dirs)}")
    log(f"Validating on {len(val_subset_dirs)} cases per interval")

    train_ds = Brain3DDataset(train_dirs, cfg["patch_size"],
                              cfg=cfg, augment=True)
    val_ds   = Brain3DValDataset(val_subset_dirs, cfg=cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2,
    )

    # -- Model ---------------------------------------------------------------
    model = AttentionUNet3D(**MODEL_CONFIG).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- Optimiser & schedule ------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=cfg["betas"],
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg["warmup_epochs"],
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"], eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg["warmup_epochs"]],
    )
    scaler    = GradScaler()
    criterion = BaselineCriterion(cfg["ce_class_weights"], device)

    best_val_dice_avg = 0.0
    best_epoch        = 0
    patience_counter  = 0

    log("=" * 80)
    log("Starting training...")
    log("=" * 80)

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        log(f"\n{'='*80}")
        log(f"Epoch {epoch+1}/{cfg['epochs']} | {datetime.datetime.now()}")
        log(f"{'='*80}")

        # ---- Train ---------------------------------------------------------
        model.train()
        train_loss         = 0.0
        train_dice_batches = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for i, batch in enumerate(pbar):
            if i >= cfg["patches_per_epoch"]:
                break

            img = batch["image"].to(device)
            seg = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(img)
                loss   = criterion(logits, seg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            with torch.no_grad():
                pred        = logits.argmax(dim=1)
                dice_scores = dice_per_class(pred, seg, cfg["out_channels"])
                train_dice_batches.append([d.item() for d in dice_scores])

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n_batches        = min(len(train_loader), cfg["patches_per_epoch"])
        train_loss      /= n_batches
        train_dice_mean  = np.mean(train_dice_batches, axis=0)

        log(f"TRAIN | Loss: {train_loss:.4f}")
        log(f"TRAIN | Dice per class: {train_dice_mean.tolist()}")
        log(f"TRAIN | Mean Dice (FG): {float(np.mean(train_dice_mean[1:])):.4f}")

        # ---- Validation ----------------------------------------------------
        if (epoch + 1) % cfg["val_interval"] == 0 or \
                epoch == cfg["epochs"] - 1:

            dice_avg, val_metrics = validate(
                model, val_loader, cfg, device, log)

            if dice_avg > best_val_dice_avg + cfg["min_delta"]:
                best_val_dice_avg = dice_avg
                best_epoch        = epoch + 1
                patience_counter  = 0
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice_avg":         dice_avg,
                    "val_metrics":          val_metrics,
                    "config":               cfg,
                }, os.path.join(cfg["save_dir"], cfg["save_name"]))
                log(f"NEW BEST | Avg Dice: {dice_avg:.4f}")
            else:
                patience_counter += 1
                log(f"No improvement: {patience_counter}/{cfg['patience']}")

            if patience_counter >= cfg["patience"]:
                log(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()
        log(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        log(f"Epoch time: {time.time()-epoch_start:.1f}s")
        log(f"Best Avg Dice: {best_val_dice_avg:.4f} at epoch {best_epoch}")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(cfg["save_dir"],
                            f"checkpoint_epoch_{epoch+1}.pth"))

    log("=" * 80)
    log(f"Training complete: {datetime.datetime.now()}")
    log(f"Best Avg Dice: {best_val_dice_avg:.4f} at epoch {best_epoch}")
    log("=" * 80)
    log_f.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Attention U-Net M1 (Bottleneck CBAM).")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--save_dir",  default=None)
    args = parser.parse_args()

    cfg = dict(TRAIN_CONFIG)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.save_dir:
        cfg["save_dir"] = args.save_dir

    run_training(cfg)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
