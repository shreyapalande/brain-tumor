"""
losses.py -- Loss functions for attn_net_m2

Baseline loss (used in train.py)
---------------------------------
  L = 0.3 * CE + 0.5 * Dice + 0.2 * ET_Dice

  Shared with attn_net_m1 -- re-exported here for standalone use.

Boundary-aware loss (used in finetune.py)
-----------------------------------------
  L = seg_loss + lambda_disc * mean(L_disc_enc2, L_disc_enc3, L_disc_enc4)

  seg_loss = 0.25 * CE_boundary + 0.40 * Dice
           + 0.15 * ET_Dice     + 0.20 * TC_Dice

  L_disc is applied at three encoder levels (enc2/3/4), each with its own
  spatial-attention map.  Boundary masks are computed by downsampling the
  label to each encoder resolution first, then eroding -- this preserves
  boundary voxels at coarser resolutions.

Difference from attn_net_m1
----------------------------
attn_net_m1.BoundaryAwareCriterion supervises a single bottleneck SA map
via attn_dict["cbam_bottleneck"].
MultiLevelBoundaryAwareCriterion supervises three encoder SA maps
via attn_dict["cbam2"], attn_dict["cbam3"], attn_dict["cbam4"].

References
----------
Woo et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
"""

from monai.losses import DiceCELoss

# Boundary helper functions are identical across all variants -- re-export.
from ..attn_net_m1.losses import (
    BaselineCriterion,
    compute_boundary_mask_3d,
    get_boundary_at_resolution,
    boundary_weighted_ce,
    sa_discriminability_loss,
)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Multi-level boundary-aware criterion (encoder CBAM levels)
# ---------------------------------------------------------------------------
class MultiLevelBoundaryAwareCriterion(nn.Module):
    """
    Boundary-aware loss for multi-level encoder (or decoder) CBAM.

    Total loss = seg_loss + lambda_disc * mean(L_disc over CBAM levels)

    seg_loss = 0.25 * CE_boundary
             + 0.40 * Dice
             + 0.15 * ET_Dice
             + 0.20 * TC_Dice

    SA maps are accessed from attn_dict via the keys listed in
    `cbam_levels` (default: ["cbam2", "cbam3", "cbam4"]).  Each key
    must map to a dict {"ca": ..., "sa": Tensor (B,1,d,h,w)}.

    Parameters
    ----------
    lambda_disc            : float -- weight for discriminability loss
    boundary_weight        : float -- CE weight for boundary voxels
    disc_margin            : float -- hinge margin for SA gap
    min_boundary_voxels    : int   -- skip level if too few boundary voxels
    boundary_erosion_iters : int   -- erosion iterations for boundary mask
    cbam_levels            : list[str] -- attn_dict keys to supervise
    """

    def __init__(self,
                 lambda_disc: float = 0.2,
                 boundary_weight: float = 6.0,
                 disc_margin: float = 0.30,
                 min_boundary_voxels: int = 10,
                 boundary_erosion_iters: int = 1,
                 cbam_levels: list = None) -> None:
        super().__init__()
        self.lambda_disc         = lambda_disc
        self.boundary_weight     = boundary_weight
        self.disc_margin         = disc_margin
        self.min_boundary_voxels = min_boundary_voxels
        self.erosion_iters       = boundary_erosion_iters
        self.cbam_levels         = cbam_levels or ["cbam2", "cbam3", "cbam4"]

        self.dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, include_background=False,
            lambda_dice=1.0, lambda_ce=0.0,
            smooth_nr=1e-5, smooth_dr=1e-5,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                attn_dict: dict) -> tuple:
        """
        Parameters
        ----------
        logits    : FloatTensor  (B, C, D, H, W)
        target    : LongTensor   (B, D, H, W)
        attn_dict : dict  -- keys from self.cbam_levels, each -> {"sa": ...}

        Returns
        -------
        total : scalar loss tensor
        comps : dict of float loss components for logging
        """
        # 1. Boundary-weighted CE at full patch resolution
        bnd_full, _ = get_boundary_at_resolution(
            target, tuple(target.shape[1:]), self.erosion_iters)
        ce = boundary_weighted_ce(logits, target,
                                  bnd_full.to(logits.device),
                                  self.boundary_weight)

        # 2. Dice (foreground classes only)
        dice = self.dice_loss(logits, target.unsqueeze(1))

        # 3. ET Dice
        pred    = torch.softmax(logits, dim=1)
        et_p    = pred[:, 3]
        et_gt   = (target == 3).float()
        inter   = (et_p * et_gt).sum()
        union   = et_p.sum() + et_gt.sum()
        et_dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)

        # 4. TC Dice
        tc_p     = pred[:, 1] + pred[:, 3]
        tc_gt    = ((target == 1) | (target == 3)).float()
        inter_tc = (tc_p * tc_gt).sum()
        union_tc  = tc_p.sum() + tc_gt.sum()
        tc_dice  = 1.0 - (2.0 * inter_tc + 1e-5) / (union_tc + 1e-5)

        seg_loss = (0.25 * ce + 0.40 * dice
                    + 0.15 * et_dice + 0.20 * tc_dice)

        # 5. Multi-level SA discriminability loss
        disc_total    = torch.tensor(0.0, device=logits.device)
        n_disc_levels = 0

        for level_key in self.cbam_levels:
            sa_map        = attn_dict[level_key]["sa"]
            _, _, d, h, w = sa_map.shape

            bnd_down, bgd_down = get_boundary_at_resolution(
                target, (d, h, w), self.erosion_iters)

            l_disc        = sa_discriminability_loss(
                sa_map, bnd_down, bgd_down,
                min_voxels=self.min_boundary_voxels,
                margin=self.disc_margin,
            )
            disc_total    = disc_total + l_disc
            n_disc_levels += 1

        if n_disc_levels > 0:
            disc_total = disc_total / n_disc_levels

        total = seg_loss + self.lambda_disc * disc_total

        return total, {
            "seg_loss":  seg_loss.item(),
            "disc_loss": disc_total.item(),
            "ce":        ce.item(),
            "dice":      dice.item(),
            "et_dice":   et_dice.item(),
            "tc_dice":   tc_dice.item(),
        }


__all__ = [
    "BaselineCriterion",
    "MultiLevelBoundaryAwareCriterion",
    "compute_boundary_mask_3d",
    "get_boundary_at_resolution",
    "boundary_weighted_ce",
    "sa_discriminability_loss",
]
