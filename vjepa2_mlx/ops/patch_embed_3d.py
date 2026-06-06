"""3D tubelet patch embedding — MLX port of VJEPA2PatchEmbeddings3D.

Conv3d with kernel = stride = (tubelet_size, patch_size, patch_size) over the
video [N, T, H, W, C] (NDHWC in MLX), flattened to patch tokens. Images with
T < tubelet_size are repeated along time (per VJEPA2Embeddings). Parity in P2.

PyTorch Conv3d weight (O,I,kT,kH,kW) -> MLX (O,kT,kH,kW,I).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError("P2/P3: nn.Conv3d(in_chans, hidden, "
                                  "kernel=stride=(tubelet,patch,patch)) + flatten")

    def __call__(self, pixel_values_videos: mx.array) -> mx.array:
        raise NotImplementedError("P2/P3")
