"""
finetune.py -- Boundary-aware fine-tuning for attn_net_m1

Loads a pre-trained bottleneck CBAM checkpoint and fine-tunes it with
boundary-aware supervision.

Fine-tuning schedule
--------------------
  Optimiser : AdamW (lr=5e-6, weight_decay=1e-4)
  Schedule  : ReduceLROnPlateau (mode=max, patience=8, factor=0.5,
                                  min_lr=1e-7)
  Epochs    : 80  (early-stop with patience=20 validation checks)
  Batch     : 2 patches per step, 400 steps per epoch
  Loss      : 0.25*CE_boundary + 0.40*Dice + 0.15*ET_Dice + 0.20*TC_Dice
            + 0.20 * L_disc  (SA discriminability at bottleneck)

Usage
-----
    python finetune.py [--checkpoint_path PATH] [--save_dir PATH]
"""

import os
import argparse
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config  import FINETUNE_CONFIG, MODEL_CONFIG
from .model   import AttentionUNet3D
from .dataset import Brain3DDataset, Brain3DValDataset
from .losses  import BoundaryAwareCriterion
from .metrics import dice_per_class
from .utils   import get_case_dirs, make_logger, validate, log_sa_saturation


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------
def run_finetuning(cfg: dict) -> None:
    device = torch.device(cfg["device"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    log_path = os.path.join(cfg["save_dir"], "finetune_log.txt")
    log, log_f = make_logger(log_path)

    log("=" * 80)
    log("Attention U-Net M1 (Bottleneck CBAM) -- Boundary-Aware Fine-Tuning")
    log(f"Start: {datetime.datetime.now()}")
    log("=" * 80)
    log(f"Checkpoint:      {cfg['checkpoint_path']}")
    log(f"Save dir:        {cfg['save_dir']}")
    log(f"LR:              {cfg['lr']}")
    log(f"Epochs:          {cfg['epochs']}")
    log(f"lambda_disc:     {cfg['lambda_disc']}")
    log(f"disc_margin:     {cfg['disc_margin']}")
    log(f"boundary_weight: {cfg['boundary_weight']}")
    log(f"CBAM level supervised: bottleneck (8^3 for 128^3 patches)")
    log(f"Scheduler: ReduceLROnPlateau "
        f"(patience={cfg['rlrop_patience']}, factor={cfg['rlrop_factor']})")
    log("=" * 80)

    # -- Data ----------------------------------------------------------------
    train_dirs, val_dirs = get_case_dirs(cfg["data_root"])
    val_subset_dirs      = val_dirs[:cfg["val_subset_size"]]

    log(f"Train cases: {len(train_dirs)} | Val cases: {len(val_dirs)}")

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
    ckpt  = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
        f"val_dice_avg={ckpt.get('val_dice_avg', 0.0):.4f}")

    # -- Optimiser & schedule ------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=cfg["betas"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg["rlrop_factor"],
        patience=cfg["rlrop_patience"],
        min_lr=cfg["rlrop_min_lr"],
        verbose=True,
    )
    scaler    = GradScaler()
    criterion = BoundaryAwareCriterion(
        lambda_disc=cfg["lambda_disc"],
        boundary_weight=cfg["boundary_weight"],
        disc_margin=cfg["disc_margin"],
        min_boundary_voxels=cfg["min_boundary_voxels"],
        boundary_erosion_iters=cfg["boundary_erosion_iters"],
    )

    best_dice_avg     = ckpt.get("val_dice_avg", 0.0)
    best_epoch        = 0
    patience_counter  = 0
    disc_loss_history = []
    sa_diag_logged    = False

    log("\nStarting fine-tuning...\n")

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        log(f"\n{'='*80}")
        log(f"Epoch {epoch+1}/{cfg['epochs']} | {datetime.datetime.now()}")
        log(f"{'='*80}")

        # ---- Train ---------------------------------------------------------
        model.train()
        train_loss         = 0.0
        train_disc_loss    = 0.0
        train_seg_loss     = 0.0
        train_tc_loss      = 0.0
        train_dice_batches = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
        for i, batch in enumerate(pbar):
            if i >= cfg["patches_per_epoch"]:
                break

            img = batch["image"].to(device)
            seg = batch["label"].to(device)
            optimizer.zero_grad()

            with autocast():
                attn_out  = model(img, return_attention=True)
                logits    = attn_out["out"]
                attn_dict = {"cbam_bottleneck": attn_out["cbam_bottleneck"]}
                loss, comps = criterion(logits, seg, attn_dict)

            if not sa_diag_logged:
                log_sa_saturation(attn_dict, log)
                sa_diag_logged = True

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss      += loss.item()
            train_disc_loss += comps["disc_loss"]
            train_seg_loss  += comps["seg_loss"]
            train_tc_loss   += comps["tc_dice"]

            with torch.no_grad():
                pred        = logits.argmax(dim=1)
                dice_scores = dice_per_class(pred, seg, cfg["out_channels"])
                train_dice_batches.append([d.item() for d in dice_scores])

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "disc": f"{comps['disc_loss']:.4f}",
                "tc":   f"{comps['tc_dice']:.4f}",
            })

        n_batches        = min(len(train_loader), cfg["patches_per_epoch"])
        train_loss      /= n_batches
        train_disc_loss /= n_batches
        train_seg_loss  /= n_batches
        train_tc_loss   /= n_batches
        train_dice_mean  = np.mean(train_dice_batches, axis=0)
        disc_loss_history.append(train_disc_loss)

        log(f"TRAIN | Total loss:   {train_loss:.4f}")
        log(f"TRAIN | Seg loss:     {train_seg_loss:.4f}")
        log(f"TRAIN | Disc loss:    {train_disc_loss:.4f}  "
            f"(weighted: {cfg['lambda_disc']*train_disc_loss:.4f})")
        log(f"TRAIN | TC dice loss: {train_tc_loss:.4f}")
        log(f"TRAIN | Dice/class:   {train_dice_mean.tolist()}")
        log(f"TRAIN | Mean FG Dice: {float(np.mean(train_dice_mean[1:])):.4f}")

        # Disc loss trend (every 10 epochs)
        if epoch > 0 and (epoch + 1) % 10 == 0:
            recent    = disc_loss_history[-10:]
            trend     = recent[-1] - recent[0]
            direction = ("decreasing" if trend < 0
                         else "flat/increasing - consider raising lambda_disc")
            log(f"DISC  | 10-epoch trend: {trend:+.4f}  ({direction})")

        # ---- Validation ----------------------------------------------------
        if (epoch + 1) % cfg["val_interval"] == 0 or \
                epoch == cfg["epochs"] - 1:

            dice_avg, val_metrics = validate(
                model, val_loader, cfg, device, log)
            scheduler.step(dice_avg)

            if dice_avg > best_dice_avg + cfg["min_delta"]:
                best_dice_avg    = dice_avg
                best_epoch       = epoch + 1
                patience_counter = 0
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice_avg":         dice_avg,
                    "val_metrics":          val_metrics,
                    "lambda_disc":          cfg["lambda_disc"],
                    "boundary_weight":      cfg["boundary_weight"],
                    "disc_margin":          cfg["disc_margin"],
                    "config":               cfg,
                }, os.path.join(cfg["save_dir"], cfg["save_name"]))
                log(f"NEW BEST | Avg Dice: {dice_avg:.4f}")
            else:
                patience_counter += 1
                log(f"No improvement: {patience_counter}/{cfg['patience']}")

            if patience_counter >= cfg["patience"]:
                log(f"Early stopping at epoch {epoch+1}")
                break

        log(f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        log(f"Epoch time: {time.time()-epoch_start:.1f}s")
        log(f"Best Avg Dice: {best_dice_avg:.4f} at epoch {best_epoch}")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(cfg["save_dir"],
                            f"checkpoint_epoch_{epoch+1}.pth"))

    log("=" * 80)
    log(f"Fine-tuning complete: {datetime.datetime.now()}")
    log(f"Best Avg Dice: {best_dice_avg:.4f} at epoch {best_epoch}")
    log("=" * 80)
    log_f.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary-aware fine-tuning for Attention U-Net M1.")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--save_dir",        default=None)
    parser.add_argument("--data_root",       default=None)
    args = parser.parse_args()

    cfg = dict(FINETUNE_CONFIG)
    if args.checkpoint_path:
        cfg["checkpoint_path"] = args.checkpoint_path
    if args.save_dir:
        cfg["save_dir"] = args.save_dir
    if args.data_root:
        cfg["data_root"] = args.data_root

    run_finetuning(cfg)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
