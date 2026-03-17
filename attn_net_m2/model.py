"""
model.py -- 3-D Attention U-Net with encoder CBAM (attn_net_m2)

This model uses CBAM at encoder levels enc2, enc3, and enc4. The conv
blocks run down to enc5, then the decoder reconstructs the segmentation
with attention gates on the skip links.

Compared to m1 (bottleneck CBAM), this one adds multi-scale CBAM earlier
in the encoder so the skip features are already refined.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CBAM: Channel Attention
# ---------------------------------------------------------------------------
class ChannelAttention3D(nn.Module):
    """Channel attention via global average- and max-pooling MLP."""

    def __init__(self, in_planes: int, ratio: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


# ---------------------------------------------------------------------------
# CBAM: Spatial Attention
# ---------------------------------------------------------------------------
class SpatialAttention3D(nn.Module):
    """Spatial attention via channel-wise pooling and a 3-D convolution."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv    = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# ---------------------------------------------------------------------------
# CBAM: Combined
# ---------------------------------------------------------------------------
class CBAM3D(nn.Module):
    """
    CBAM3D applies channel then spatial attention sequentially.

    Supports return_attention to expose raw attention maps for loss
    supervision or visualisation.
    """

    def __init__(self, in_planes: int, ratio: int = 8,
                 kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        ca_map = self.ca(x)
        x      = x * ca_map
        sa_map = self.sa(x)
        x      = x * sa_map
        if return_attention:
            return x, {"ca": ca_map, "sa": sa_map}
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks
# ---------------------------------------------------------------------------
class ConvBlock3D(nn.Module):
    """Double 3x3x3 convolution with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate3D(nn.Module):
    """Additive soft attention gate for skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv3d(F_g, F_int, 1, bias=False),
                                   nn.BatchNorm3d(F_int))
        self.W_x  = nn.Sequential(nn.Conv3d(F_l, F_int, 1, bias=False),
                                   nn.BatchNorm3d(F_int))
        self.psi  = nn.Sequential(nn.Conv3d(F_int, 1, 1, bias=True),
                                   nn.BatchNorm3d(1),
                                   nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class AttentionUNet3D(nn.Module):
    """
    3-D Attention U-Net with CBAM modules at encoder levels 2, 3, and 4.

    CBAM placement
    --------------
    enc2 -> cbam2  (D/2  x H/2  x W/2)
    enc3 -> cbam3  (D/4  x H/4  x W/4)
    enc4 -> cbam4  (D/8  x H/8  x W/8)

    The CBAM-refined features serve as skip connections to the decoder and
    are also available for multi-scale SA discriminability supervision.

    Parameters
    ----------
    in_ch   : int  -- input modalities (1 for FLAIR-only)
    out_ch  : int  -- segmentation classes (4 for BraTS)
    filters : tuple[int, ...] -- feature widths at each encoder level

    Forward return
    --------------
    If return_attention is False (default):
        logits : Tensor  (B, out_ch, D, H, W)
    If return_attention is True:
        dict with keys
            "out"   -- logits tensor
            "cbam2" -- {"ca": ..., "sa": ...}  enc2 attention maps
            "cbam3" -- {"ca": ..., "sa": ...}  enc3 attention maps
            "cbam4" -- {"ca": ..., "sa": ...}  enc4 attention maps
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 4,
                 filters: tuple = (48, 96, 192, 384, 768)) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = filters

        # Encoder
        self.enc1 = ConvBlock3D(in_ch, f1)
        self.enc2 = ConvBlock3D(f1, f2)
        self.enc3 = ConvBlock3D(f2, f3)
        self.enc4 = ConvBlock3D(f3, f4)
        self.enc5 = ConvBlock3D(f4, f5)
        self.pool = nn.MaxPool3d(2)

        # Encoder CBAM modules
        self.cbam2 = CBAM3D(f2)   # after enc2, D/2
        self.cbam3 = CBAM3D(f3)   # after enc3, D/4
        self.cbam4 = CBAM3D(f4)   # after enc4, D/8

        # Decoder
        self.up4  = nn.ConvTranspose3d(f5, f4, 2, stride=2)
        self.att4 = AttentionGate3D(F_g=f4, F_l=f4, F_int=f3)
        self.dec4 = ConvBlock3D(f5, f4)

        self.up3  = nn.ConvTranspose3d(f4, f3, 2, stride=2)
        self.att3 = AttentionGate3D(F_g=f3, F_l=f3, F_int=f2)
        self.dec3 = ConvBlock3D(f4, f3)

        self.up2  = nn.ConvTranspose3d(f3, f2, 2, stride=2)
        self.att2 = AttentionGate3D(F_g=f2, F_l=f2, F_int=f1)
        self.dec2 = ConvBlock3D(f3, f2)

        self.up1  = nn.ConvTranspose3d(f2, f1, 2, stride=2)
        self.att1 = AttentionGate3D(F_g=f1, F_l=f1, F_int=f1 // 2)
        self.dec1 = ConvBlock3D(f2, f1)

        # Segmentation head
        self.out = nn.Conv3d(f1, out_ch, 1)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Crop x to match the spatial size of ref."""
        return x[..., :ref.shape[2], :ref.shape[3], :ref.shape[4]]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Encoder + CBAM at enc2, enc3, enc4
        e1 = self.enc1(x)

        e2_raw = self.enc2(self.pool(e1))
        if return_attention:
            e2, att2 = self.cbam2(e2_raw, return_attention=True)
        else:
            e2 = self.cbam2(e2_raw)

        e3_raw = self.enc3(self.pool(e2))
        if return_attention:
            e3, att3 = self.cbam3(e3_raw, return_attention=True)
        else:
            e3 = self.cbam3(e3_raw)

        e4_raw = self.enc4(self.pool(e3))
        if return_attention:
            e4, att4 = self.cbam4(e4_raw, return_attention=True)
        else:
            e4 = self.cbam4(e4_raw)

        e5 = self.enc5(self.pool(e4))

        # Decoder
        d4 = self.up4(e5)
        e4 = self.att4(d4, self._match_size(e4, d4))
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, self._match_size(e3, d3))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, self._match_size(e2, d2))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, self._match_size(e1, d1))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.out(d1)

        if return_attention:
            return {
                "out":   logits,
                "cbam2": att2,   # enc2, D/2
                "cbam3": att3,   # enc3, D/4
                "cbam4": att4,   # enc4, D/8
            }
        return logits
