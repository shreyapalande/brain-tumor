"""
attn_net_m2 -- Attention U-Net with CBAM at encoder levels enc2, enc3, enc4.

FLAIR-only input, 4-class BraTS segmentation.
Filters: (48, 96, 192, 384, 768).

Modules
-------
  model   : AttentionUNet3D  -- encoder CBAM architecture
  config  : TRAIN_CONFIG, FINETUNE_CONFIG, TEST_CONFIG, MODEL_CONFIG
  losses  : BaselineCriterion, MultiLevelBoundaryAwareCriterion
  utils   : get_case_dirs, make_logger, validate, log_sa_saturation
  train   : run_training    -- from-scratch training entry point
  finetune: run_finetuning  -- boundary-aware fine-tuning entry point
  test    : run_test        -- evaluation entry point

Shared modules (dataset, metrics) are imported from attn_net_m1.
"""

from .model  import AttentionUNet3D
from .config import TRAIN_CONFIG, FINETUNE_CONFIG, TEST_CONFIG, MODEL_CONFIG
from .losses import BaselineCriterion, MultiLevelBoundaryAwareCriterion

__all__ = [
    "AttentionUNet3D",
    "TRAIN_CONFIG",
    "FINETUNE_CONFIG",
    "TEST_CONFIG",
    "MODEL_CONFIG",
    "BaselineCriterion",
    "MultiLevelBoundaryAwareCriterion",
]
