"""
attn_net_m1 -- Attention U-Net with CBAM at the bottleneck (enc5).

Public API
----------
  AttentionUNet3D       -- model class
  TRAIN_CONFIG          -- baseline training hyperparameters
  FINETUNE_CONFIG       -- boundary fine-tuning hyperparameters
  TEST_CONFIG           -- evaluation settings
  MODEL_CONFIG          -- architecture hyperparameters
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
