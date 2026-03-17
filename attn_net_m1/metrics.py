"""
metrics.py -- Segmentation evaluation metrics for BraTS

Volumetric metrics
------------------
  dice_binary      -- Dice Similarity Coefficient for binary masks
  iou_binary       -- Intersection over Union for binary masks
  dice_per_class   -- per-class Dice (4 classes, torch tensors)
  brats_wt_tc_et   -- BraTS standard Dice for ET / TC / WT regions
  post_process     -- remove small connected-component noise

Boundary / surface metrics
--------------------------
  extract_boundary -- erode-based surface extraction
  boundary_f1      -- boundary F1 score (BF score)
  boundary_iou     -- boundary Intersection over Union
  hd95             -- 95th-percentile Hausdorff distance (voxel units)
  asd              -- average symmetric surface distance (voxel units)

BraTS region definitions
------------------------
  WT (Whole Tumour)  : labels {1, 2, 3}
  TC (Tumour Core)   : labels {1, 3}
  ET (Enhancing Tumour): label {3}
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
import torch


# ---------------------------------------------------------------------------
# Volumetric metrics (numpy)
# ---------------------------------------------------------------------------
def dice_binary(pred: np.ndarray, gt: np.ndarray,
                eps: float = 1e-5) -> float:
    """Dice Similarity Coefficient for two boolean / binary arrays."""
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (union + eps))


def iou_binary(pred: np.ndarray, gt: np.ndarray,
               eps: float = 1e-5) -> float:
    """Intersection over Union for two boolean / binary arrays."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def brats_wt_tc_et(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    Compute BraTS-standard Dice for ET, TC, and WT regions.

    Returns
    -------
    (dice_et, dice_tc, dice_wt) : tuple of float
    """
    dice_et = dice_binary(pred == 3,
                          gt == 3)
    dice_tc = dice_binary((pred == 1) | (pred == 3),
                          (gt  == 1) | (gt  == 3))
    dice_wt = dice_binary(pred > 0,
                          gt  > 0)
    return dice_et, dice_tc, dice_wt


def dice_per_class(pred: torch.Tensor, gt: torch.Tensor,
                   n_cls: int = 4) -> list:
    """
    Per-class Dice Similarity Coefficient (torch tensors).

    Parameters
    ----------
    pred  : LongTensor  (B, D, H, W) or (D, H, W)
    gt    : LongTensor  same shape
    n_cls : int -- number of classes

    Returns
    -------
    list of float, length n_cls
    """
    out = []
    for c in range(n_cls):
        p     = (pred == c)
        g     = (gt   == c)
        inter = (p & g).sum()
        union = p.sum() + g.sum()
        out.append((2 * inter + 1e-5) / (union + 1e-5))
    return out


def post_process(pred: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    Remove connected components smaller than min_size voxels from each
    foreground class.

    Parameters
    ----------
    pred     : ndarray (D, H, W)  -- integer label map
    min_size : int                -- minimum component size in voxels

    Returns
    -------
    ndarray (D, H, W) with small components zeroed out.
    """
    out = np.zeros_like(pred)
    for cls in range(1, 4):
        mask           = (pred == cls)
        labeled, n     = ndi.label(mask)
        for i in range(1, n + 1):
            if np.sum(labeled == i) >= min_size:
                out[labeled == i] = cls
    return out


# ---------------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------------
def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract the surface voxels of a binary mask via morphological erosion.

    Returns a boolean array of the same shape where True indicates a
    surface voxel (present in mask but removed by erosion).
    """
    structure = np.ones((3, 3, 3))
    eroded    = ndi.binary_erosion(mask, structure=structure)
    return np.logical_xor(mask, eroded)


def boundary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Boundary F1 score (BF score) between two binary segmentations.

    Returns 1.0 when both masks are empty.
    """
    b_pred = extract_boundary(pred)
    b_gt   = extract_boundary(gt)
    tp = np.logical_and(b_pred, b_gt).sum()
    fp = np.logical_and(b_pred, ~b_gt).sum()
    fn = np.logical_and(~b_pred, b_gt).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn))


def boundary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Boundary Intersection over Union between two binary segmentations.

    Returns 1.0 when both boundary sets are empty.
    """
    b_pred = extract_boundary(pred)
    b_gt   = extract_boundary(gt)
    inter  = np.logical_and(b_pred, b_gt).sum()
    union  = np.logical_or(b_pred, b_gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def hd95(pred: np.ndarray, gt: np.ndarray,
         percentile: int = 95) -> float:
    """
    95th-percentile Hausdorff distance in voxel units.

    Computed symmetrically between the boundary point sets of pred and gt.
    Returns 0.0 when either boundary is empty.
    """
    pts_pred = np.array(np.where(extract_boundary(pred))).T
    pts_gt   = np.array(np.where(extract_boundary(gt))).T
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 0.0
    d_pg = cdist(pts_pred, pts_gt).min(axis=1)
    d_gp = cdist(pts_gt, pts_pred).min(axis=1)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), percentile))


def asd(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Average Symmetric Surface Distance in voxel units.

    Returns 0.0 when either boundary is empty.
    """
    pts_pred = np.array(np.where(extract_boundary(pred))).T
    pts_gt   = np.array(np.where(extract_boundary(gt))).T
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return 0.0
    d_pg = cdist(pts_pred, pts_gt).min(axis=1)
    d_gp = cdist(pts_gt, pts_pred).min(axis=1)
    return float(np.concatenate([d_pg, d_gp]).mean())
