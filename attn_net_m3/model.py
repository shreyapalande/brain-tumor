"""
model.py -- 3-D Attention U-Net with decoder CBAM (attn_net_m3)

Architecture overview
---------------------
Encoder : 5 x ConvBlock3D with MaxPool3d downsampling (no CBAM).
Decoder : 4 upsampling stages, each with an Attention Gate followed by
           a CBAM3D module applied to the decoder feature map after the
           double-convolution block.
           Levels supervised: dec4 (D/8), dec3 (D/4), dec2 (D/2).
           dec1 (full resolution) is NOT supervised -- gradients are too
           noisy at 128^3 resolution.
Output  : 1 x 1 x 1 convolution to <out_ch> class logits.

Difference from attn_net_m2
----------------------------
attn_net_m2 places CBAM after encoder convolution blocks (enc2/3/4).
attn_net_m3 places CBAM after decoder convolution blocks (dec1/2/3/4).
Placing CBAM in the decoder means attention is applied to semantically
richer, upsampled features rather than raw encoder activations.

Forward return key aliasing
----------------------------
To share the BoundaryAwareCriterion with attn_net_m2, the attention
dict uses keys "cbam2/3/4" mapped to decoder levels dec2/3/4:
    cbam4 -> dec4 attention  (D/8  x H/8  x W/8,  same res as enc4)
    cbam3 -> dec3 attention  (D/4  x H/4  x W/4,  same res as enc3)
    cbam2 -> dec2 attention  (D/2  x H/2  x W/2,  same res as enc2)

Reference
---------
Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
MIDL 2018.

Woo et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
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
    3-D Attention U-Net with CBAM modules at decoder levels 4, 3, 2, and 1.

    CBAM placement (after each decoder ConvBlock)
    ----------------------------------------------
    dec4 -> cbam_dec4  (D/8  x H/8  x W/8)   supervised
    dec3 -> cbam_dec3  (D/4  x H/4  x W/4)   supervised
    dec2 -> cbam_dec2  (D/2  x H/2  x W/2)   supervised
    dec1 -> cbam_dec1  (D    x H    x W)      NOT supervised (too large)

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
        dict with keys (aliased to match attn_net_m2 for shared criterion)
            "out"   -- logits tensor
            "cbam4" -- {"ca": ..., "sa": ...}  dec4 attention (D/8)
            "cbam3" -- {"ca": ..., "sa": ...}  dec3 attention (D/4)
            "cbam2" -- {"ca": ..., "sa": ...}  dec2 attention (D/2)
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 4,
                 filters: tuple = (48, 96, 192, 384, 768)) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = filters

        # Encoder (no CBAM)
        self.enc1 = ConvBlock3D(in_ch, f1)
        self.enc2 = ConvBlock3D(f1, f2)
        self.enc3 = ConvBlock3D(f2, f3)
        self.enc4 = ConvBlock3D(f3, f4)
        self.enc5 = ConvBlock3D(f4, f5)
        self.pool = nn.MaxPool3d(2)

        # Decoder with attention gates and post-conv CBAM
        self.up4       = nn.ConvTranspose3d(f5, f4, 2, stride=2)
        self.att4      = AttentionGate3D(F_g=f4, F_l=f4, F_int=f3)
        self.dec4      = ConvBlock3D(f5, f4)
        self.cbam_dec4 = CBAM3D(f4)   # D/8 -- supervised

        self.up3       = nn.ConvTranspose3d(f4, f3, 2, stride=2)
        self.att3      = AttentionGate3D(F_g=f3, F_l=f3, F_int=f2)
        self.dec3      = ConvBlock3D(f4, f3)
        self.cbam_dec3 = CBAM3D(f3)   # D/4 -- supervised

        self.up2       = nn.ConvTranspose3d(f3, f2, 2, stride=2)
        self.att2      = AttentionGate3D(F_g=f2, F_l=f2, F_int=f1)
        self.dec2      = ConvBlock3D(f3, f2)
        self.cbam_dec2 = CBAM3D(f2)   # D/2 -- supervised

        self.up1       = nn.ConvTranspose3d(f2, f1, 2, stride=2)
        self.att1      = AttentionGate3D(F_g=f1, F_l=f1, F_int=f1 // 2)
        self.dec1      = ConvBlock3D(f2, f1)
        self.cbam_dec1 = CBAM3D(f1)   # D   -- NOT supervised (gradient too noisy)

        # Segmentation head
        self.out = nn.Conv3d(f1, out_ch, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Encoder (no CBAM)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Decoder stage 4 + CBAM
        d4 = self.up4(e5)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        if return_attention:
            d4, att4 = self.cbam_dec4(d4, return_attention=True)   # D/8
        else:
            d4 = self.cbam_dec4(d4)

        # Decoder stage 3 + CBAM
        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        if return_attention:
            d3, att3 = self.cbam_dec3(d3, return_attention=True)   # D/4
        else:
            d3 = self.cbam_dec3(d3)

        # Decoder stage 2 + CBAM
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        if return_attention:
            d2, att2 = self.cbam_dec2(d2, return_attention=True)   # D/2
        else:
            d2 = self.cbam_dec2(d2)

        # Decoder stage 1 + CBAM (not supervised)
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.cbam_dec1(d1)

        logits = self.out(d1)

        if return_attention:
            return {
                "out":   logits,
                # Keys aliased to match attn_net_m2 / BoundaryAwareCriterion
                "cbam4": att4,   # dec4, D/8
                "cbam3": att3,   # dec3, D/4
                "cbam2": att2,   # dec2, D/2
            }
        return logits
