"""
test.py -- Evaluation of attn_net_m1 on the BraTS test split

Loads raw NIfTI files from the test set, runs sliding-window inference,
and reports the full suite of BraTS segmentation metrics:

  Volumetric : Dice (ET / TC / WT), Avg Dice, mIoU (FG classes)
  Boundary   : Boundary F1, Boundary IoU, HD95, ASD

Saves a result summary to <save_dir>/test_results.txt and produces
axial visualisations for the first MAX_VIZ patients.

Usage
-----
    python test.py [--model_path PATH] [--data_root PATH]
                   [--test_list PATH]  [--save_dir PATH]
"""

import os
import argparse

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config  import TEST_CONFIG, MODEL_CONFIG
from .model   import AttentionUNet3D
from .metrics import (brats_wt_tc_et, iou_binary,
                      boundary_f1, boundary_iou, hd95, asd)


# ---------------------------------------------------------------------------
# BraTS colour map: BG / NCR / ED / ET
# ---------------------------------------------------------------------------
BRATS_CMAP = ListedColormap([
    "black",    # 0 Background
    "#1f77b4",  # 1 NCR  (blue)
    "#ffdf00",  # 2 ED   (yellow)
    "#d62728",  # 3 ET   (red)
])

MAX_VIZ = 5   # number of patients for which axial slices are saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_normalise_flair(flair_path: str) -> np.ndarray:
    """
    Load a FLAIR NIfTI volume, reorient to RAS canonical, and apply
    percentile clipping followed by z-score normalisation.

    Returns a float32 ndarray of shape (D, H, W) in axial-first order.
    """
    img = nib.load(flair_path)
    vol = img.get_fdata().astype(np.float32)         # (H, W, D)
    vol = np.transpose(vol, (2, 0, 1))               # -> (D, H, W)
    p_low, p_high = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, p_low, p_high)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol


def pick_tumour_slice(gt_dhw: np.ndarray) -> int:
    """Return the axial index of the median tumour-containing slice."""
    D           = gt_dhw.shape[0]
    tumor_slices = np.where(gt_dhw.sum(axis=(1, 2)) > 0)[0]
    if len(tumor_slices) > 0:
        return int(tumor_slices[len(tumor_slices) // 2])
    return D // 2


def save_visualisation(flair: np.ndarray, gt: np.ndarray,
                        pred: np.ndarray, pid: str,
                        save_dir: str) -> None:
    """Save a 3-panel axial visualisation (FLAIR | GT | Prediction)."""
    z = pick_tumour_slice(gt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(flair[z], cmap="gray")
    axes[0].set_title(f"{pid} | FLAIR (z={z})")
    axes[1].imshow(gt[z],   cmap=BRATS_CMAP, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred[z], cmap=BRATS_CMAP, vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{pid}_slice{z}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def run_test(cfg: dict) -> None:
    device = torch.device(cfg["device"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Load model
    model = AttentionUNet3D(**MODEL_CONFIG).to(device)
    ckpt  = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    # Load test patient list
    with open(cfg["test_list"]) as fh:
        test_patients = [l.strip() for l in fh if l.strip()]
    print(f"Testing on {len(test_patients)} patients")

    all_results = []
    visualized  = 0

    with torch.no_grad():
        for pid in tqdm(test_patients, desc="Testing"):
            case_dir   = os.path.join(cfg["data_root"], pid)
            flair_path = os.path.join(case_dir, f"{pid}-t2f.nii.gz")
            seg_path   = os.path.join(case_dir, f"{pid}-seg.nii.gz")

            if not os.path.exists(flair_path) or not os.path.exists(seg_path):
                print(f"  [SKIP] Missing files for {pid}")
                continue

            flair_raw = nib.load(flair_path).get_fdata().astype(np.float32)
            seg_raw   = nib.load(seg_path).get_fdata().astype(np.int64)

            # Transpose to (D, H, W) axial-first
            flair_np = np.transpose(flair_raw, (2, 0, 1))
            seg      = np.transpose(seg_raw,   (2, 0, 1))

            # Normalise FLAIR for inference
            p_low, p_high = np.percentile(flair_np, [0.5, 99.5])
            flair_norm    = np.clip(flair_np, p_low, p_high)
            flair_norm    = ((flair_norm - flair_norm.mean())
                             / (flair_norm.std() + 1e-8))

            # Inference: (1, 1, D, H, W)
            inp = (torch.tensor(flair_norm, dtype=torch.float32)
                   .unsqueeze(0).unsqueeze(0).to(device))

            logits = sliding_window_inference(
                inputs=inp,
                roi_size=cfg["patch_size"],
                sw_batch_size=cfg["sw_batch_size"],
                predictor=model,
                overlap=cfg["overlap"],
            )
            pred = logits.argmax(dim=1).cpu().numpy()[0]

            # -- Visualisation -----------------------------------------------
            if visualized < MAX_VIZ:
                save_visualisation(flair_np, seg, pred, pid, cfg["save_dir"])
                visualized += 1

            # -- Volumetric metrics ------------------------------------------
            dice_et, dice_tc, dice_wt = brats_wt_tc_et(pred, seg)
            dice_avg = (dice_et + dice_tc + dice_wt) / 3.0

            miou_fg = float(np.mean([
                iou_binary(pred == c, seg == c) for c in [1, 2, 3]
            ]))

            # -- Boundary metrics --------------------------------------------
            bf_et  = boundary_f1(pred == 3,              seg == 3)
            bf_tc  = boundary_f1((pred==1)|(pred==3),    (seg==1)|(seg==3))
            bf_wt  = boundary_f1(pred > 0,               seg > 0)
            bf_avg = (bf_et + bf_tc + bf_wt) / 3.0

            bi_et  = boundary_iou(pred == 3,             seg == 3)
            bi_tc  = boundary_iou((pred==1)|(pred==3),   (seg==1)|(seg==3))
            bi_wt  = boundary_iou(pred > 0,              seg > 0)
            bi_avg = (bi_et + bi_tc + bi_wt) / 3.0

            h_et   = hd95(pred == 3,                     seg == 3)
            h_tc   = hd95((pred==1)|(pred==3),           (seg==1)|(seg==3))
            h_wt   = hd95(pred > 0,                      seg > 0)
            h_avg  = (h_et + h_tc + h_wt) / 3.0

            a_et   = asd(pred == 3,                      seg == 3)
            a_tc   = asd((pred==1)|(pred==3),            (seg==1)|(seg==3))
            a_wt   = asd(pred > 0,                       seg > 0)
            a_avg  = (a_et + a_tc + a_wt) / 3.0

            all_results.append([
                dice_et, dice_tc, dice_wt, dice_avg, miou_fg,
                bf_et,   bf_tc,   bf_wt,   bf_avg,
                bi_et,   bi_tc,   bi_wt,   bi_avg,
                h_et,    h_tc,    h_wt,    h_avg,
                a_et,    a_tc,    a_wt,    a_avg,
            ])

    # -- Summary -------------------------------------------------------------
    results = np.array(all_results)
    m       = results.mean(axis=0)

    lines = [
        "=" * 60,
        "FINAL TEST RESULTS -- Attention U-Net M1 (Bottleneck CBAM)",
        "=" * 60,
        f"ET Dice        : {m[0]:.4f}",
        f"TC Dice        : {m[1]:.4f}",
        f"WT Dice        : {m[2]:.4f}",
        f"Avg Dice       : {m[3]:.4f}",
        f"mIoU (FG)      : {m[4]:.4f}",
        "",
        f"Boundary F1    ET / TC / WT / Avg : "
        f"{m[5]:.4f} / {m[6]:.4f} / {m[7]:.4f} / {m[8]:.4f}",
        f"Boundary IoU   ET / TC / WT / Avg : "
        f"{m[9]:.4f} / {m[10]:.4f} / {m[11]:.4f} / {m[12]:.4f}",
        f"HD95 (voxels)  ET / TC / WT / Avg : "
        f"{m[13]:.2f} / {m[14]:.2f} / {m[15]:.2f} / {m[16]:.2f}",
        f"ASD  (voxels)  ET / TC / WT / Avg : "
        f"{m[17]:.2f} / {m[18]:.2f} / {m[19]:.2f} / {m[20]:.2f}",
        "=" * 60,
    ]

    for line in lines:
        print(line)

    result_path = os.path.join(cfg["save_dir"], "test_results.txt")
    with open(result_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {result_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Attention U-Net M1 on BraTS test split.")
    parser.add_argument("--model_path",  default=None)
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--test_list",   default=None)
    parser.add_argument("--save_dir",    default=None)
    args = parser.parse_args()

    cfg = dict(TEST_CONFIG)
    if args.model_path: cfg["model_path"] = args.model_path
    if args.data_root:  cfg["data_root"]  = args.data_root
    if args.test_list:  cfg["test_list"]  = args.test_list
    if args.save_dir:   cfg["save_dir"]   = args.save_dir

    run_test(cfg)


if __name__ == "__main__":
    main()
