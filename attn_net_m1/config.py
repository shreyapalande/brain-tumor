"""
config.py -- Hyperparameter configuration for attn_net_m1

attn_net_m1: Attention U-Net with CBAM placed at the bottleneck (enc5).
CBAM resolution: (D/16, H/16, W/16)  -- 8^3 for 128^3 patches.

Three configs are provided:
  TRAIN_CONFIG    -- baseline training from scratch
  FINETUNE_CONFIG -- boundary-aware fine-tuning from a pretrained checkpoint
  TEST_CONFIG     -- inference and evaluation settings
"""

# ---------------------------------------------------------------------------
# Shared base (common to training and fine-tuning)
# ---------------------------------------------------------------------------
_BASE = {
    "data_root":       r"C:\Users\palande.3\brats\data_npy",
    "patch_size":      (128, 128, 128),
    "batch_size":      2,
    "num_workers":     4,
    "device":          "cuda",
    "out_channels":    4,
    "val_interval":    5,
    "flair_channel":   0,
    "use_only_flair":  True,
    "patience":        20,
    "min_delta":       0.001,
    "val_patch_size":  (96, 96, 96),
    "val_subset_size": 30,
    "val_overlap":     0.4,
}

# ---------------------------------------------------------------------------
# Baseline training (train.py)
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    **_BASE,
    "save_dir":          r"C:\Users\palande.3\brats\attn_net_m1",
    "save_name":         "best_model_attn_net_m1.pth",

    # Optimiser
    "lr":                5e-5,
    "weight_decay":      1e-4,
    "betas":             (0.9, 0.999),

    # Schedule: LinearLR warmup -> CosineAnnealingLR
    "epochs":            300,
    "warmup_epochs":     10,

    # Epoch budget
    "patches_per_epoch": 400,

    # Class weights for CE loss  [BG, NCR, ED, ET]
    "ce_class_weights":  [1.0, 1.5, 2.0, 3.0],
}

# ---------------------------------------------------------------------------
# Boundary-aware fine-tuning (finetune.py)
# ---------------------------------------------------------------------------
FINETUNE_CONFIG = {
    **_BASE,
    "checkpoint_path": r"C:\Users\palande.3\brats\attn_net_m1\best_model_attn_net_m1.pth",
    "save_dir":        r"C:\Users\palande.3\brats\attn_net_m1_boundary",
    "save_name":       "best_model_attn_net_m1_boundary.pth",

    # Optimiser
    "lr":              5e-6,
    "weight_decay":    1e-4,
    "betas":           (0.9, 0.999),

    # Schedule: ReduceLROnPlateau (metric = val Avg Dice)
    "epochs":            80,
    "patches_per_epoch": 400,
    "rlrop_patience":    8,
    "rlrop_factor":      0.5,
    "rlrop_min_lr":      1e-7,

    # Boundary loss
    "lambda_disc":            0.2,
    "boundary_weight":        6.0,
    "disc_margin":            0.30,
    "min_boundary_voxels":    10,
    "boundary_erosion_iters": 1,
}

# ---------------------------------------------------------------------------
# Test / evaluation (test.py)
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "data_root":    r"C:\Users\palande.3\brats\dataset",
    "test_list":    r"C:\Users\palande.3\brats\dataset_split\patients_missing_in_data_npy.txt",
    "model_path":   r"C:\Users\palande.3\brats\attn_net_m1\best_model_attn_net_m1.pth",
    "save_dir":     r"C:\Users\palande.3\brats\attn_net_m1\visualizations",

    "device":       "cuda",
    "out_channels": 4,
    "patch_size":   (96, 96, 96),
    "overlap":      0.2,
    "sw_batch_size": 4,
}

# ---------------------------------------------------------------------------
# Model architecture (shared across all scripts)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "in_ch":     1,
    "out_ch":    4,
    "filters":   (48, 96, 192, 384, 768),
}
