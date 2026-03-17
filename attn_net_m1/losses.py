"""
losses.py -- Loss functions for attn_net_m1

Baseline loss (used in train.py)
---------------------------------
  L = 0.3 * CE + 0.5 * Dice + 0.2 * ET_Dice

  where CE uses class weights [1.0, 1.5, 2.0, 3.0] to counter class
  imbalance, and ET_Dice explicitly penalises errors on the small
  Enhancing Tumour region.

Boundary-aware loss (used in finetune.py)
-----------------------------------------
  L = seg_loss + lambda_disc * L_disc

  seg_loss = 0.25 * CE_boundary + 0.40 * Dice
           + 0.15 * ET_Dice     + 0.20 * TC_Dice

  L_disc is a hinge-based spatial-attention discriminability loss that
  encourages the bottleneck CBAM spatial-attention map to assign higher
  activation to tumour-boundary voxels than background voxels.

References
----------
Woo et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
from monai.losses import DiceCELoss


# ---------------------------------------------------------------------------
# Boundary mask utilities
# ---------------------------------------------------------------------------
def compute_boundary_mask_3d(seg_np_batch: np.ndarray,
                              erosion_iters: int = 1) -> tuple:
    """
    Compute per-volume tumour-boundary and background masks.

    The boundary is defined as the set of tumour voxels that are removed
    by a single binary erosion, i.e. the outer shell of the tumour region.

    Parameters
    ----------
    seg_np_batch : ndarray  (B, D, H, W)  -- integer label batch
    erosion_iters : int                   -- erosion iterations

    Returns
    -------
    boundary   : ndarray  (B, D, H, W) bool
    background : ndarray  (B, D, H, W) bool
    """
    struct = ndi.generate_binary_structure(3, 1)
    B      = seg_np_batch.shape[0]
    boundary_list, background_list = [], []

    for b in range(B):
        s     = seg_np_batch[b]
        tumor = (s > 0).astype(np.uint8)
        bg    = (s == 0)

        if tumor.sum() == 0:
            boundary_list.append(np.zeros_like(tumor, dtype=bool))
            background_list.append(bg)
            continue

        eroded   = ndi.binary_erosion(tumor, structure=struct,
                                      iterations=erosion_iters)
        boundary = tumor.astype(bool) & (~eroded)
        boundary_list.append(boundary)
        background_list.append(bg)

    return np.stack(boundary_list), np.stack(background_list)


def get_boundary_at_resolution(seg_tensor: torch.Tensor,
                                target_size: tuple,
                                erosion_iters: int = 1) -> tuple:
    """
    Down-sample a label tensor to target_size (nearest-neighbour),
    then compute boundary and background masks.

    Used to align boundary supervision with the spatial attention map
    resolution inside the model.

    Parameters
    ----------
    seg_tensor  : LongTensor  (B, D, H, W)
    target_size : tuple (D', H', W')
    erosion_iters : int

    Returns
    -------
    bnd_mask : BoolTensor  (B, D', H', W')
    bgd_mask : BoolTensor  (B, D', H', W')
    """
    seg_f    = seg_tensor.float().unsqueeze(1)
    seg_down = F.interpolate(seg_f, size=target_size,
                             mode="nearest").squeeze(1)
    seg_np   = seg_down.long().cpu().numpy()
    bnd_np, bgd_np = compute_boundary_mask_3d(seg_np, erosion_iters)
    return torch.from_numpy(bnd_np), torch.from_numpy(bgd_np)


# ---------------------------------------------------------------------------
# Boundary-weighted cross-entropy
# ---------------------------------------------------------------------------
def boundary_weighted_ce(logits: torch.Tensor,
                          target: torch.Tensor,
                          boundary_mask: torch.Tensor,
                          boundary_weight: float = 6.0) -> torch.Tensor:
    """
    Cross-entropy loss with elevated weight on tumour-boundary voxels.

    Boundary voxels receive weight = boundary_weight; all other voxels
    receive weight = 1.

    Parameters
    ----------
    logits         : FloatTensor  (B, C, D, H, W)
    target         : LongTensor   (B, D, H, W)
    boundary_mask  : BoolTensor   (B, D, H, W)  -- True at boundary voxels
    boundary_weight : float       -- weight multiplier for boundary voxels
    """
    weight_map = torch.ones_like(target, dtype=torch.float32)
    weight_map[boundary_mask] = boundary_weight

    log_probs    = F.log_softmax(logits, dim=1)
    per_voxel_ce = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return (per_voxel_ce * weight_map).sum() / (weight_map.sum() + 1e-8)


# ---------------------------------------------------------------------------
# SA discriminability loss
# ---------------------------------------------------------------------------
def sa_discriminability_loss(sa_map: torch.Tensor,
                              boundary_mask: torch.Tensor,
                              background_mask: torch.Tensor,
                              min_voxels: int = 10,
                              margin: float = 0.15) -> torch.Tensor:
    """
    Hinge-based loss that encourages the spatial-attention map to assign
    higher activation to tumour-boundary voxels than to background voxels.

    L_disc = ReLU(margin - (mean_bnd_SA - mean_bgd_SA))

    Parameters
    ----------
    sa_map          : FloatTensor  (B, 1, D', H', W')
    boundary_mask   : BoolTensor   (B, D', H', W')
    background_mask : BoolTensor   (B, D', H', W')
    min_voxels      : int  -- skip samples with too few boundary / bg voxels
    margin          : float -- desired gap between boundary and bg activations
    """
    sa     = sa_map.squeeze(1)   # (B, D', H', W')
    B      = sa.shape[0]
    losses = []

    for b in range(B):
        bnd = boundary_mask[b].to(sa.device)
        bgd = background_mask[b].to(sa.device)

        if bnd.sum().item() < min_voxels or bgd.sum().item() < min_voxels:
            continue

        mean_bnd = sa[b][bnd].mean()
        mean_bgd = sa[b][bgd].mean()
        gap      = mean_bnd - mean_bgd
        losses.append(F.relu(margin - gap))

    if not losses:
        return torch.tensor(0.0, requires_grad=False, device=sa_map.device)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Baseline criterion
# ---------------------------------------------------------------------------
class BaselineCriterion(nn.Module):
    """
    L = 0.3 * CE + 0.5 * Dice + 0.2 * ET_Dice

    Parameters
    ----------
    ce_class_weights : list[float]  -- per-class CE weights [BG,NCR,ED,ET]
    device           : str or torch.device
    """

    def __init__(self, ce_class_weights: list, device) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(ce_class_weights, device=device)
        )
        self.dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, include_background=False,
            lambda_dice=1.0, lambda_ce=0.0,
            smooth_nr=1e-5, smooth_dr=1e-5,
        )

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        ce   = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target.unsqueeze(1))

        pred  = torch.softmax(logits, dim=1)
        et_p  = pred[:, 3]
        et_gt = (target == 3).float()
        inter = (et_p * et_gt).sum()
        union = et_p.sum() + et_gt.sum()
        et_dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)

        return 0.3 * ce + 0.5 * dice + 0.2 * et_dice


# ---------------------------------------------------------------------------
# Boundary-aware criterion
# ---------------------------------------------------------------------------
class BoundaryAwareCriterion(nn.Module):
    """
    Total loss = seg_loss + lambda_disc * L_disc_bottleneck

    seg_loss = 0.25 * CE_boundary
             + 0.40 * Dice
             + 0.15 * ET_Dice
             + 0.20 * TC_Dice

    L_disc supervises the spatial-attention map of the bottleneck CBAM
    (passed in via attn_dict["cbam_bottleneck"]["sa"]).

    Parameters
    ----------
    lambda_disc            : float
    boundary_weight        : float
    disc_margin            : float
    min_boundary_voxels    : int
    boundary_erosion_iters : int
    """

    def __init__(self, lambda_disc: float = 0.2,
                 boundary_weight: float = 6.0,
                 disc_margin: float = 0.30,
                 min_boundary_voxels: int = 10,
                 boundary_erosion_iters: int = 1) -> None:
        super().__init__()
        self.lambda_disc         = lambda_disc
        self.boundary_weight     = boundary_weight
        self.disc_margin         = disc_margin
        self.min_boundary_voxels = min_boundary_voxels
        self.erosion_iters       = boundary_erosion_iters

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
        attn_dict : dict with key "cbam_bottleneck" -> {"ca": ..., "sa": ...}

        Returns
        -------
        total  : scalar loss tensor
        comps  : dict of float loss components for logging
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
        tc_p    = pred[:, 1] + pred[:, 3]
        tc_gt   = ((target == 1) | (target == 3)).float()
        inter_tc = (tc_p * tc_gt).sum()
        union_tc  = tc_p.sum() + tc_gt.sum()
        tc_dice = 1.0 - (2.0 * inter_tc + 1e-5) / (union_tc + 1e-5)

        seg_loss = (0.25 * ce + 0.40 * dice
                    + 0.15 * et_dice + 0.20 * tc_dice)

        # 5. SA discriminability loss at bottleneck resolution
        sa_map        = attn_dict["cbam_bottleneck"]["sa"]
        _, _, d, h, w = sa_map.shape
        bnd_down, bgd_down = get_boundary_at_resolution(
            target, (d, h, w), self.erosion_iters)

        disc_loss = sa_discriminability_loss(
            sa_map, bnd_down, bgd_down,
            min_voxels=self.min_boundary_voxels,
            margin=self.disc_margin,
        )

        total = seg_loss + self.lambda_disc * disc_loss

        return total, {
            "seg_loss":  seg_loss.item(),
            "disc_loss": disc_loss.item(),
            "ce":        ce.item(),
            "dice":      dice.item(),
            "et_dice":   et_dice.item(),
            "tc_dice":   tc_dice.item(),
        }
