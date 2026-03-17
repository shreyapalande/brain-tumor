"""
utils.py -- Utilities for attn_net_m3

All generic utilities (get_case_dirs, make_logger, validate) are
re-exported from attn_net_m1.utils.

This module provides a variant-specific log_sa_saturation that labels
the three supervised decoder CBAM levels with their correct names and
spatial resolutions (dec4/3/2, aliased as cbam4/3/2 in the attn_dict).
"""

from typing import Callable

import torch

# Re-export all shared utilities
from ..attn_net_m1.utils import get_case_dirs, make_logger, validate


# ---------------------------------------------------------------------------
# SA saturation diagnostic -- decoder levels
# ---------------------------------------------------------------------------
def log_sa_saturation(attn_dict: dict, log: Callable) -> None:
    """
    Print statistics for decoder CBAM spatial-attention maps to detect
    saturation (std < 0.01 indicates a collapsed attention map that
    produces negligible discriminability-loss gradients).

    dec1 (cbam1 / full resolution) is intentionally excluded from
    supervision and is not checked here.

    Parameters
    ----------
    attn_dict : dict  -- output of model(x, return_attention=True)
                         expected keys: "cbam4", "cbam3", "cbam2"
                         (aliased from dec4, dec3, dec2 respectively)
                         each -> {"sa": Tensor (B,1,d,h,w)}
    log       : callable
    """
    log("SA Saturation Diagnostic (decoder levels)")
    res_labels = {
        "cbam4": "dec4 (D/8)",
        "cbam3": "dec3 (D/4)",
        "cbam2": "dec2 (D/2)",
    }
    for key in ["cbam4", "cbam3", "cbam2"]:
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
