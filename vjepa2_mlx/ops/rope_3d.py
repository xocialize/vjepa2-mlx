"""3D Rotary Position Embeddings — MLX port of VJEPA2RopeAttention's rotary path.

The crux op. V-JEPA2 applies RoPE over a 3D token grid (depth = frames/tubelet,
height, width); the head_dim (64) is split across the three axes. This is NOT in
config.json — the exact axis split, frequency base, and apply order are read from
transformers `modeling_vjepa2.py::VJEPA2RopeAttention.apply_rotary_embeddings`
in P1, then parity-locked in P2 (Gate A). mx.fast.rope handles 1D; the 3D
composition (per-axis rope on disjoint channel groups) is assembled here.
"""

from __future__ import annotations

import mlx.core as mx


def rope_3d(q: mx.array, k: mx.array, pos_ids: mx.array, head_dim: int):
    raise NotImplementedError("P2: port VJEPA2RopeAttention.apply_rotary_embeddings "
                              "(3D axis split + per-axis rotary); parity Gate A")
