"""
dataset.py -- PyTorch Dataset classes for BraTS 3-D patch training

Classes
-------
Brain3DDataset    -- training dataset with on-the-fly patch extraction and
                     augmentation.  Implements tumour-biased cropping to
                     increase the frequency of informative patches.
Brain3DValDataset -- validation / inference dataset that returns full
                     volumes (no augmentation, no cropping).

Both datasets expect a list of patient directories, each containing:
    image.npy  -- float32 FLAIR volume (H x W x D)
    label.npy  -- uint8  segmentation mask (H x W x D)

Label convention
----------------
  0 - Background
  1 - Necrotic Core (NCR)
  2 - Peritumoral Edema (ED)
  3 - Enhancing Tumor (ET)
"""

import os
import random

import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_volume(case_dir: str, cfg: dict) -> tuple:
    """
    Load image and label from a case directory, then transpose to
    (C, D, H, W) and (D, H, W) respectively and normalise the image.
    """
    img = np.load(os.path.join(case_dir, "image.npy"))  # (H, W, D)
    seg = np.load(os.path.join(case_dir, "label.npy"))  # (H, W, D)

    # Ensure channel dimension exists
    if img.ndim == 3:
        img = img[np.newaxis, ...]           # (1, H, W, D)

    # Transpose to (C, D, H, W)
    img = np.transpose(img, (0, 3, 1, 2))   # (C, D, H, W)
    seg = np.transpose(seg, (2, 0, 1))       # (D, H, W)

    # Select FLAIR channel
    if cfg["use_only_flair"] and img.shape[0] > 1:
        img = img[cfg["flair_channel"]:cfg["flair_channel"] + 1]
    elif img.shape[0] > 1:
        img = img[:1]

    # Percentile clipping + z-score normalisation
    ch = img[0]
    p_low, p_high = np.percentile(ch, [0.5, 99.5])
    ch = np.clip(ch, p_low, p_high)
    if ch.std() > 1e-8:
        ch = (ch - ch.mean()) / ch.std()
    img[0] = ch

    return img, seg


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
class Brain3DDataset(Dataset):
    """
    Random-patch 3-D training dataset with tumour-biased cropping.

    For each sample the dataset:
      1. Loads the full volume and normalises.
      2. Attempts up to 30 crops targeting a random tumour class
         (NCR / ED / ET drawn from a prior favouring ED and ET).
      3. Falls back to any tumour-containing crop, then to a random crop.
      4. Optionally applies intensity and spatial augmentation.

    Parameters
    ----------
    case_dirs  : list[str]  -- paths to per-patient directories
    patch_size : tuple[int] -- (D, H, W) patch dimensions
    cfg        : dict       -- must contain "use_only_flair", "flair_channel"
    augment    : bool       -- enable augmentation (default True)
    """

    def __init__(self, case_dirs: list, patch_size: tuple,
                 cfg: dict, augment: bool = True) -> None:
        self.case_dirs  = case_dirs
        self.patch      = patch_size
        self.cfg        = cfg
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.case_dirs)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def apply_augmentation(self, img: np.ndarray,
                            seg: np.ndarray) -> tuple:
        """
        On-the-fly augmentation applied to a single patch pair.

        Transformations (each applied independently at random):
          - Random axis flips (x3 axes)
          - Random 90-degree rotation in the axial plane
          - Intensity scale jitter in [0.8, 1.2]
          - Intensity shift jitter in [-0.2, 0.2]
          - Additive Gaussian noise (sigma=0.05)
          - Gaussian blur (sigma in [0.5, 1.5])
          - Gamma correction (gamma in [0.8, 1.2])
        """
        # Spatial flips
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            seg = np.flip(seg, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            seg = np.flip(seg, axis=1).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=3).copy()
            seg = np.flip(seg, axis=2).copy()

        # Axial-plane 90-degree rotations
        if random.random() > 0.5:
            k = random.randint(1, 3)
            img = np.rot90(img, k, axes=(2, 3)).copy()
            seg = np.rot90(seg, k, axes=(1, 2)).copy()

        # Intensity augmentations
        if random.random() > 0.5:
            img = img * random.uniform(0.8, 1.2)
        if random.random() > 0.5:
            img = img + random.uniform(-0.2, 0.2)
        if random.random() > 0.3:
            img = img + np.random.normal(0, 0.05, img.shape)
        if random.random() > 0.3:
            sigma = random.uniform(0.5, 1.5)
            img[0] = ndi.gaussian_filter(img[0], sigma=sigma)
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.sign(img) * np.power(np.abs(img), gamma)

        return img, seg

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------
    def random_crop_3d(self, img: np.ndarray,
                       seg: np.ndarray) -> tuple:
        """Extract a random 3-D patch from (C, D, H, W) image."""
        C, D, H, W = img.shape
        pd, ph, pw = self.patch
        d0 = random.randint(0, max(0, D - pd))
        h0 = random.randint(0, max(0, H - ph))
        w0 = random.randint(0, max(0, W - pw))
        return (img[:, d0:d0+pd, h0:h0+ph, w0:w0+pw],
                seg[d0:d0+pd, h0:h0+ph, w0:w0+pw])

    def _tumour_biased_crop(self, img: np.ndarray,
                             seg: np.ndarray) -> tuple:
        """
        Tumour-biased patch sampling strategy.

        1. Draw a target class from {1, 2, 3} with p=[0.25, 0.35, 0.40].
        2. Try up to 30 crops containing the target class.
        3. Fall back: try 15 crops with any tumour (sum > 100 voxels).
        4. Final fallback: fully random crop.
        """
        target = np.random.choice([1, 2, 3], p=[0.25, 0.35, 0.40])

        for _ in range(30):
            ic, sc = self.random_crop_3d(img, seg)
            if np.any(sc == target):
                return ic, sc

        for _ in range(15):
            ic, sc = self.random_crop_3d(img, seg)
            if np.sum(sc > 0) > 100:
                return ic, sc

        return self.random_crop_3d(img, seg)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        img, seg = _load_volume(self.case_dirs[idx], self.cfg)
        img_crop, seg_crop = self._tumour_biased_crop(img, seg)

        if self.augment:
            img_crop, seg_crop = self.apply_augmentation(img_crop, seg_crop)

        return {
            "image": torch.tensor(img_crop.copy(), dtype=torch.float32),
            "label": torch.tensor(seg_crop.copy(), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Validation / inference dataset
# ---------------------------------------------------------------------------
class Brain3DValDataset(Dataset):
    """
    Full-volume validation dataset (no augmentation, no cropping).

    Returns the entire normalised volume so that sliding-window inference
    can be applied at test time.

    Parameters
    ----------
    case_dirs : list[str] -- paths to per-patient directories
    cfg       : dict      -- must contain "use_only_flair", "flair_channel"
    """

    def __init__(self, case_dirs: list, cfg: dict) -> None:
        self.case_dirs = case_dirs
        self.cfg       = cfg

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> dict:
        img, seg = _load_volume(self.case_dirs[idx], self.cfg)
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "label": torch.tensor(seg, dtype=torch.long),
        }
