"""
utils.py -- Utilities for attn_net_m2

All generic utilities (get_case_dirs, make_logger, validate) are
re-exported from attn_net_m1.utils.

This module provides a variant-specific log_sa_saturation that labels
the three encoder CBAM levels (enc2/3/4) with their correct spatial
resolutions.
"""

from typing import Callable

import torch

# Re-export all shared utilities
from ..attn_net_m1.utils import get_case_dirs, make_logger, validate


# ---------------------------------------------------------------------------
# SA saturation diagnostic -- encoder levels
# ---------------------------------------------------------------------------
def log_sa_saturation(attn_dict: dict, log: Callable) -> None:
    """
    Print statistics for encoder CBAM spatial-attention maps to detect
    saturation (std < 0.01 indicates a collapsed attention map that
    produces negligible discriminability-loss gradients).

    Parameters
    ----------
    attn_dict : dict  -- output of model(x, return_attention=True)
                         expected keys: "cbam2", "cbam3", "cbam4"
                         each -> {"sa": Tensor (B,1,d,h,w)}
    log       : callable
    """
    log("SA Saturation Diagnostic (encoder levels)")
    res_labels = {
        "cbam2": "enc2 (D/2)",
        "cbam3": "enc3 (D/4)",
        "cbam4": "enc4 (D/8)",
    }
    for key in ["cbam2", "cbam3", "cbam4"]:
        sa = attn_dict[key]["sa"].detach()
        saturated = " *** SATURATED ***" if sa.std() < 0.01 else ""
        log(
            f"  {res_labels[key]} SA | "
            f"min={sa.min():.4f}  max={sa.max():.4f}  "
            f"mean={sa.mean():.4f}  std={sa.std():.5f}{saturated}"
        )


__all__ = [
    "get_case_dirs",
    "make_logger",
    "validate",
    "log_sa_saturation",
]
