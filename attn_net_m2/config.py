"""
config.py -- Hyperparameter configuration for attn_net_m2

attn_net_m2: Attention U-Net with CBAM at encoder levels enc2, enc3, enc4.
FLAIR-only input, 4-class output (BG / NCR / ED / ET),
filters = (48, 96, 192, 384, 768).

Three configs are provided:
  TRAIN_CONFIG    -- from-scratch training
  FINETUNE_CONFIG -- boundary-aware fine-tuning from attn_net_m2 checkpoint
  TEST_CONFIG     -- inference and evaluation settings
  MODEL_CONFIG    -- architecture hyperparameters
"""

# ---------------------------------------------------------------------------
# Shared base
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
# Training (train.py)
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    **_BASE,
    "save_dir":        r"C:\Users\palande.3\brats\attn_net_m2",
    "save_name":       "best_model_attn_net_m2.pth",

    # Optimiser
    "lr":              5e-5,
    "weight_decay":    1e-4,
    "betas":           (0.9, 0.999),
    "warmup_epochs":   10,

    # Schedule
    "epochs":            300,
    "patches_per_epoch": 400,

    # Loss
    "ce_class_weights": [1.0, 1.5, 2.0, 3.0],
}

# ---------------------------------------------------------------------------
# Fine-tuning (finetune.py)
# ---------------------------------------------------------------------------
FINETUNE_CONFIG = {
    **_BASE,
    # Paths
    "checkpoint_path": r"C:\Users\palande.3\brats\attn_net_m2\best_model_attn_net_m2.pth",
    "save_dir":        r"C:\Users\palande.3\brats\attn_net_m2_boundary",
    "save_name":       "best_model_attn_net_m2_boundary.pth",

    # Optimiser
    "lr":           5e-6,
    "weight_decay": 1e-4,
    "betas":        (0.9, 0.999),

    # Schedule: ReduceLROnPlateau
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

    # Which encoder CBAM levels to supervise
    "cbam_levels": ["cbam2", "cbam3", "cbam4"],
}

# ---------------------------------------------------------------------------
# Test / evaluation (test.py)
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "data_root":     r"C:\Users\palande.3\brats\dataset",
    "test_list":     r"C:\Users\palande.3\brats\dataset_split\patients_missing_in_data_npy.txt",
    "model_path":    r"C:\Users\palande.3\brats\attn_net_m2\best_model_attn_net_m2.pth",
    "save_dir":      r"C:\Users\palande.3\brats\attn_net_m2\visualizations",

    "device":        "cuda",
    "out_channels":  4,
    "patch_size":    (96, 96, 96),
    "overlap":       0.2,
    "sw_batch_size": 4,
}

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "in_ch":   1,
    "out_ch":  4,
    "filters": (48, 96, 192, 384, 768),
}
