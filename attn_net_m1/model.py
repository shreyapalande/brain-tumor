"""
model.py -- 3-D Attention U-Net with bottleneck CBAM (attn_net_m1)

Architecture overview
---------------------
Encoder : 5 x ConvBlock3D with MaxPool3d downsampling.
Bottleneck : enc5 features are refined by a single CBAM3D module
             (Convolutional Block Attention Module, Woo et al. 2018).
             Resolution: (D/16, H/16, W/16).
Decoder : 4 upsampling stages, each with an Attention Gate on the
          corresponding skip connection before concatenation.
Output  : 1 x 1 x 1 convolution to <out_ch> class logits.

Reference
---------
Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
MIDL 2018.

Woo et al. "CBAM: Convolutional Block Attention Module."
ECCV 2018.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CBAM: Channel Attention
# ---------------------------------------------------------------------------
class ChannelAttention3D(nn.Module):
    """
    Channel attention via global average- and max-pooling MLP.

    Parameters
    ----------
    in_planes : int  -- number of input feature-map channels
    ratio     : int  -- bottleneck ratio for the shared MLP
    """

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
    """
    Spatial attention via channel-wise pooling and a 3-D convolution.

    Parameters
    ----------
    kernel_size : int  -- convolution kernel size (3 or 7)
    """

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

    Supports an optional return_attention flag that also returns the
    raw channel- and spatial-attention maps for loss supervision.

    Parameters
    ----------
    in_planes   : int  -- input channel count
    ratio       : int  -- channel attention bottleneck ratio
    kernel_size : int  -- spatial attention kernel size
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
    """
    Additive soft attention gate for skip connections.

    Parameters
    ----------
    F_g   : int -- gating signal channels (from decoder path)
    F_l   : int -- skip connection channels (from encoder path)
    F_int : int -- intermediate channel count
    """

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
    3-D Attention U-Net with a single CBAM module at the bottleneck.

    Parameters
    ----------
    in_ch   : int            -- number of input modalities (1 for FLAIR-only)
    out_ch  : int            -- number of segmentation classes (4 for BraTS)
    filters : tuple[int, ...]-- feature-map widths at each encoder level
                                default: (48, 96, 192, 384, 768)

    Forward return
    --------------
    If return_attention is False (default):
        logits : Tensor  (B, out_ch, D, H, W)
    If return_attention is True:
        dict with keys
            "out"             -- logits tensor
            "cbam_bottleneck" -- dict {"ca": ..., "sa": ...}
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 4,
                 filters: tuple = (48, 96, 192, 384, 768)) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = filters

        # Encoder
        self.enc1 = ConvBlock3D(in_ch, f1)
        self.enc2 = ConvBlock3D(f1,    f2)
        self.enc3 = ConvBlock3D(f2,    f3)
        self.enc4 = ConvBlock3D(f3,    f4)
        self.enc5 = ConvBlock3D(f4,    f5)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck CBAM  (resolution: D/16 x H/16 x W/16)
        self.cbam_bottleneck = CBAM3D(f5)

        # Decoder stage 4  (upsampled from f5 -> f4)
        self.up4  = nn.ConvTranspose3d(f5, f4, 2, stride=2)
        self.att4 = AttentionGate3D(F_g=f4, F_l=f4, F_int=f3)
        self.dec4 = ConvBlock3D(f5, f4)

        # Decoder stage 3
        self.up3  = nn.ConvTranspose3d(f4, f3, 2, stride=2)
        self.att3 = AttentionGate3D(F_g=f3, F_l=f3, F_int=f2)
        self.dec3 = ConvBlock3D(f4, f3)

        # Decoder stage 2
        self.up2  = nn.ConvTranspose3d(f3, f2, 2, stride=2)
        self.att2 = AttentionGate3D(F_g=f2, F_l=f2, F_int=f1)
        self.dec2 = ConvBlock3D(f3, f2)

        # Decoder stage 1
        self.up1  = nn.ConvTranspose3d(f2, f1, 2, stride=2)
        self.att1 = AttentionGate3D(F_g=f1, F_l=f1, F_int=f1 // 2)
        self.dec1 = ConvBlock3D(f2, f1)

        # Segmentation head
        self.out = nn.Conv3d(f1, out_ch, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Bottleneck CBAM
        if return_attention:
            e5, att_bottleneck = self.cbam_bottleneck(e5, return_attention=True)
        else:
            e5 = self.cbam_bottleneck(e5)

        # Decoder
        d4 = self.up4(e5);  e4 = self.att4(d4, e4);  d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = self.up3(d4);  e3 = self.att3(d3, e3);  d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3);  e2 = self.att2(d2, e2);  d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2);  e1 = self.att1(d1, e1);  d1 = self.dec1(torch.cat([d1, e1], 1))

        logits = self.out(d1)

        if return_attention:
            return {"out": logits, "cbam_bottleneck": att_bottleneck}
        return logits
