"""HF state_dict -> MLX weights. Operates on numpy (no torch import at use).

Only the Conv3d patch-embed weight changes layout:
  Conv3d (O,I,kT,kH,kW) -> (O,kT,kH,kW,I)   transpose(0,2,3,4,1)
Linear weights are (out,in) in both torch and MLX -> identity. LayerNorm / bias
pass through. Keys already match the MLX module tree 1:1 (verified).

`prefix` filters a component (e.g. "encoder.") so each can be loaded alone.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def convert_state_dict(sd: dict[str, np.ndarray], prefix: str | None = None) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for k, v in sd.items():
        if prefix is not None and not k.startswith(prefix):
            continue
        v = np.asarray(v)
        if v.ndim == 5:  # Conv3d patch embed
            v = np.transpose(v, (0, 2, 3, 4, 1))
        out[k] = mx.array(v)
    return out
