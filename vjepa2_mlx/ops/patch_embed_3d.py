"""3D tubelet patch embedding — MLX port of VJEPA2PatchEmbeddings3D.

Conv3d kernel = stride = (tubelet, patch, patch). Upstream input is (B,T,C,H,W),
permuted to (B,C,T,H,W) then Conv3d → flatten(2).transpose → (B, N, hidden) with
token order depth·H'·W' (row-major). MLX Conv3d is NDHWC, so we feed (B,T,H,W,C)
and reshape the (B,D',H',W',hidden) output to (B, D'·H'·W', hidden) — same order.

PyTorch Conv3d weight (O,I,kT,kH,kW) -> MLX (O,kT,kH,kW,I).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tubelet_size = config.tubelet_size
        self.patch_size = config.patch_size
        self.proj = nn.Conv3d(
            config.in_chans, config.hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def __call__(self, video_bthwc: mx.array) -> mx.array:
        """video_bthwc: [B, T, H, W, C] (frames repeated to >= tubelet upstream)."""
        x = self.proj(video_bthwc)                 # [B, D', H', W', hidden]
        B, D, H, W, C = x.shape
        return x.reshape(B, D * H * W, C)          # [B, N, hidden]
