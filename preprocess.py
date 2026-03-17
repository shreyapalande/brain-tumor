"""
preprocess.py -- BraTS 2024 NIfTI to NPY preprocessing

Converts raw BraTS NIfTI data into per-patient NPY archives ready for
3-D training.  For each patient the script:
  1. Loads the FLAIR (t2f) volume and reorients to RAS canonical space.
  2. Applies percentile-based intensity normalisation ([0, 1] range).
  3. Saves image.npy  (float32, shape H x W x D) and
           label.npy  (uint8,   shape H x W x D)
     into  <output_dir>/<patient_id>/ .

Label convention (BraTS 2024 GLI):
  0 - Background
  1 - Necrotic Core (NCR)
  2 - Peritumoral Edema (ED)
  3 - Enhancing Tumor (ET)

Usage
-----
    python preprocess.py \\
        --data_dir   /path/to/brats/dataset \\
        --output_dir /path/to/data_npy \\
        [--overwrite]
"""

import os
import re
import argparse
import logging
from collections import defaultdict

import numpy as np
import nibabel as nib
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient discovery
# ---------------------------------------------------------------------------
def collect_patients(data_dir: str) -> list:
    """
    Return one entry per unique patient (first scan index only).

    BraTS folder names follow the pattern  BraTS-GLI-<pid>-<scan_idx>.
    When multiple longitudinal scans exist for the same patient, only
    the lowest scan index is kept, matching the standard single-timepoint
    BraTS protocol.
    """
    unique: defaultdict = defaultdict(list)

    for entry in os.listdir(data_dir):
        m = re.match(r"(BraTS-GLI-\d+)-(\d+)", entry)
        if m:
            pid, scan_idx = m.groups()
            unique[pid].append((int(scan_idx), entry))

    patients = []
    for pid, scans in unique.items():
        patients.append(sorted(scans)[0][1])   # first scan only

    return sorted(patients)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def load_canonical(path: str) -> np.ndarray:
    """Load a NIfTI file and reorient to RAS canonical space."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata()


def normalise_flair(volume: np.ndarray) -> np.ndarray:
    """
    Percentile-based min-max normalisation to [0, 1].

    Clips at the 0.5th and 99.5th percentile before scaling to
    suppress outlier intensities common in clinical MRI.
    """
    vol = np.nan_to_num(volume.astype(np.float32))
    p_low  = np.percentile(vol, 0.5)
    p_high = np.percentile(vol, 99.5)
    vol    = np.clip(vol, p_low, p_high)
    denom  = p_high - p_low + 1e-8
    return (vol - p_low) / denom


def process_label(volume: np.ndarray) -> np.ndarray:
    """Round floating-point label values and cast to uint8."""
    vol = np.nan_to_num(volume)
    return np.round(vol).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def preprocess(data_dir: str, output_dir: str, overwrite: bool = False) -> None:
    patients = collect_patients(data_dir)
    log.info("Unique patients found: %d", len(patients))

    os.makedirs(output_dir, exist_ok=True)
    skipped = 0
    processed = 0
    missing = 0

    for patient in tqdm(patients, desc="Preprocessing"):
        out_patient_dir = os.path.join(output_dir, patient)
        image_out = os.path.join(out_patient_dir, "image.npy")
        label_out = os.path.join(out_patient_dir, "label.npy")

        if not overwrite and os.path.exists(image_out) and os.path.exists(label_out):
            skipped += 1
            continue

        flair_path = os.path.join(data_dir, patient, f"{patient}-t2f.nii.gz")
        seg_path   = os.path.join(data_dir, patient, f"{patient}-seg.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            log.warning("Missing files for %s -- skipping", patient)
            missing += 1
            continue

        # Load and process
        flair_vol = load_canonical(flair_path)    # (H, W, D)
        seg_vol   = load_canonical(seg_path)       # (H, W, D)

        image = normalise_flair(flair_vol)          # float32, [0, 1]
        label = process_label(seg_vol)              # uint8,   {0,1,2,3}

        os.makedirs(out_patient_dir, exist_ok=True)
        np.save(image_out, image)
        np.save(label_out, label)
        processed += 1

    log.info(
        "Done -- processed: %d | skipped (already exist): %d | missing files: %d",
        processed, skipped, missing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 2024 NIfTI data to NPY format."
    )
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\palande.3\brats\dataset",
        help="Root directory containing BraTS patient subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        default=r"C:\Users\palande.3\brats\data_npy",
        help="Directory where per-patient NPY archives are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate NPY files even if they already exist.",
    )
    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
