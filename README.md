# Attention U-Net Variants for 3D Brain Tumor Segmentation

This repository contains three variants of 3D Attention U-Net for BraTS-style tumor segmentation, plus boundary-aware fine-tuning code.

## Project structure

- `attn_net_m1/` - Attention U-Net with one CBAM module at the bottleneck (`enc5`).
- `attn_net_m1_boundary/` - boundary fine-tuning wrapper that reuses `attn_net_m1` modules.
- `attn_net_m2/` - Attention U-Net with CBAM at encoder levels `enc2`, `enc3`, `enc4`.
- `attn_net_m3/` - Attention U-Net with CBAM at decoder levels with aliasing for boundary loss.

Each folder includes:

- `model.py` - model architecture
- `config.py` - train/finetune/test config dictionaries
- `train.py` - training script
- `finetune.py` - boundary-aware fine-tuning script
- `test.py` - inference/evaluation script
- `utils.py` - dataset and helper utilities (shared or variant-specific)

## Model differences

1. **attn_net_m1**: CBAM at the bottleneck only.
2. **attn_net_m2**: CBAM at encoder levels (`enc2`, `enc3`, `enc4`).
3. **attn_net_m3**: CBAM at decoder levels (`dec4`, `dec3`, `dec2`) with supervised attention maps aliased as `cbam4`, `cbam3`, `cbam2`.

## Boundary-aware fine-tuning

The `finetune.py` scripts use boundary-augmented cross-entropy and SA discriminability loss on CBAM spatial attention maps:

- `attn_net_m1`: `BoundaryAwareCriterion` on `cbam_bottleneck`.
- `attn_net_m2`: `MultiLevelBoundaryAwareCriterion` on `cbam2`/`cbam3`/`cbam4`.
- `attn_net_m3`: `MultiLevelBoundaryAwareCriterion` with decoder-level CBAM maps aliased to the same keys.

## How to run

1. Set `data_root` and paths in the model config file for the variant.
2. Train from scratch:
    ```bash
    python -m github.attn_net_m2.train
    ```
3. Fine-tune boundary-aware:
    ```bash
    python -m github.attn_net_m2.finetune --checkpoint_path <model.pth> --save_dir <out>
    ```
4. Test/evaluation:
    ```bash
    python -m github.attn_net_m2.test --model_path <best_model.pth>
    ```

## Notes

- The code uses mixed precision (`autocast`, `GradScaler`) in fine-tuning.
- The boundary-aware criterion computes boundary masks at attention resolutions using nearest neighbor downsampling and erosion.
- For m3, `dec1` CBAM is intentionally not supervised in boundary discrimination due to high-resolution noisy gradients.
