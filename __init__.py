"""
github -- BraTS 3-D brain tumour segmentation research package

Model variants
--------------
  attn_net_m1          Attention U-Net, CBAM at bottleneck (enc5)
  attn_net_m1_boundary attn_net_m1 fine-tuned with boundary-aware loss

Shared utilities are housed in attn_net_m1 and re-used by all
boundary-variant packages via relative imports.
"""
