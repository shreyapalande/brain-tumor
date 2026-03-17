"""
attn_net_m3 -- Attention U-Net with CBAM at decoder levels dec4, dec3, dec2, dec1.

FLAIR-only input, 4-class BraTS segmentation.
Filters: (48, 96, 192, 384, 768).

Modules
-------
  model   : AttentionUNet3D  -- decoder CBAM architecture
  config  : TRAIN_CONFIG, FINETUNE_CONFIG, TEST_CONFIG, MODEL_CONFIG
  utils   : get_case_dirs, make_logger, validate, log_sa_saturation
  train   : run_training    -- from-scratch training entry point
  finetune: run_finetuning  -- boundary-aware fine-tuning entry point
  test    : run_test        -- evaluation entry point

Shared modules (dataset, metrics, losses) are imported from
attn_net_m1 and attn_net_m2 respectively.

Note on key aliasing
--------------------
The forward pass with return_attention=True aliases decoder CBAM maps
as cbam4/cbam3/cbam2 (matching attn_net_m2 encoder keys) so that
MultiLevelBoundaryAwareCriterion from attn_net_m2 can be reused
without modification.
"""

from .model  import AttentionUNet3D
from .config import TRAIN_CONFIG, FINETUNE_CONFIG, TEST_CONFIG, MODEL_CONFIG

__all__ = [
    "AttentionUNet3D",
    "TRAIN_CONFIG",
    "FINETUNE_CONFIG",
    "TEST_CONFIG",
    "MODEL_CONFIG",
]
