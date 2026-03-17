"""
utils.py -- Shared utilities for attn_net_m1

Functions
---------
get_case_dirs       -- collect and split patient directories
make_logger         -- create a dual console+file logger
log_sa_saturation   -- diagnostic for CBAM spatial-attention saturation
validate            -- sliding-window validation loop
"""

import os
import datetime
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from .metrics import dice_per_class, brats_wt_tc_et


# ---------------------------------------------------------------------------
# Data split helper
# ---------------------------------------------------------------------------
def get_case_dirs(data_root: str, val_fraction: float = 0.2) -> tuple:
    """
    Collect all patient directories under data_root and perform a
    deterministic (sorted) train / validation split.

    Parameters
    ----------
    data_root     : str   -- root directory containing per-patient folders
    val_fraction  : float -- fraction of cases reserved for validation

    Returns
    -------
    (train_dirs, val_dirs) : tuple of list[str]
    """
    case_dirs = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    val_count  = max(1, int(val_fraction * len(case_dirs)))
    train_dirs = case_dirs[:-val_count]
    val_dirs   = case_dirs[-val_count:]
    return train_dirs, val_dirs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def make_logger(log_path: str) -> Callable:
    """
    Create a simple logger that writes to both stdout and a log file.

    Parameters
    ----------
    log_path : str -- file path for the log output

    Returns
    -------
    log : callable  -- call log(message) to write a string
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="ascii")

    def log(msg: str) -> None:
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    return log, log_f


# ---------------------------------------------------------------------------
# SA saturation diagnostic
# ---------------------------------------------------------------------------
def log_sa_saturation(attn_dict: dict, log: Callable) -> None:
    """
    Print statistics for the bottleneck CBAM spatial-attention map to
    detect saturation (std < 0.01 indicates a collapsed attention map).

    Parameters
    ----------
    attn_dict : dict  -- output of model(x, return_attention=True)
                         expected key: "cbam_bottleneck" -> {"sa": Tensor}
    log       : callable
    """
    log("SA Saturation Diagnostic (bottleneck)")
    sa       = attn_dict["cbam_bottleneck"]["sa"].detach()
    saturated = " *** SATURATED ***" if sa.std() < 0.01 else ""
    log(
        f"  bottleneck SA | min={sa.min():.4f}  max={sa.max():.4f}  "
        f"mean={sa.mean():.4f}  std={sa.std():.5f}{saturated}"
    )


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
def validate(model: nn.Module, val_loader: DataLoader,
             cfg: dict, device: torch.device,
             log: Callable) -> tuple:
    """
    Sliding-window validation loop.

    Computes:
      - Mean cross-entropy + Dice + ET-Dice loss
      - Per-class Dice (4 classes)
      - BraTS ET / TC / WT Dice and their average

    Parameters
    ----------
    model      : nn.Module
    val_loader : DataLoader  -- yields {"image", "label"}
    cfg        : dict        -- must contain val_patch_size, val_overlap,
                                out_channels
    device     : torch.device
    log        : callable

    Returns
    -------
    dice_avg    : float         -- (ET + TC + WT) / 3
    val_metrics : dict[str, float]
    """
    model.eval()

    dice_loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, include_background=False,
        lambda_dice=1.0, lambda_ce=0.0,
    )
    ce_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5, 2.0, 3.0], device=device)
    )

    def _simple_criterion(logits, target):
        ce   = ce_fn(logits, target)
        dice = dice_loss_fn(logits, target.unsqueeze(1))
        pred = torch.softmax(logits, dim=1)
        et_p = pred[:, 3]
        et_g = (target == 3).float()
        inter = (et_p * et_g).sum()
        union = et_p.sum() + et_g.sum()
        et_d  = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
        return 0.3 * ce + 0.5 * dice + 0.2 * et_d

    val_loss         = 0.0
    val_dice_batches = []
    val_wt_tc_et     = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Val")
        for batch in pbar:
            img = batch["image"].to(device)
            seg = batch["label"].to(device)

            logits = sliding_window_inference(
                inputs=img,
                roi_size=cfg["val_patch_size"],
                sw_batch_size=6,
                predictor=model,
                overlap=cfg["val_overlap"],
            )

            val_loss += _simple_criterion(logits, seg).item()

            pred   = logits.argmax(dim=1).cpu().numpy()[0]
            pred_t = torch.tensor(pred, device=device).unsqueeze(0)

            dice_scores = dice_per_class(pred_t, seg, cfg["out_channels"])
            val_dice_batches.append([d.item() for d in dice_scores])

            et, tc, wt = brats_wt_tc_et(pred, seg.cpu().numpy()[0])
            val_wt_tc_et.append([et, tc, wt])

    val_loss      /= len(val_loader)
    val_dice_mean  = np.mean(val_dice_batches, axis=0)
    et, tc, wt     = np.mean(val_wt_tc_et, axis=0)
    dice_avg       = float((et + tc + wt) / 3)

    log(f"VAL   | Loss: {val_loss:.4f}")
    log(f"VAL   | Dice per class: {val_dice_mean.tolist()}")
    log(f"VAL   | ET: {et:.4f} | TC: {tc:.4f} | WT: {wt:.4f}")
    log(f"VAL   | Avg Dice (ET+TC+WT)/3: {dice_avg:.4f}")

    return dice_avg, {"et": float(et), "tc": float(tc),
                      "wt": float(wt), "avg": dice_avg}
