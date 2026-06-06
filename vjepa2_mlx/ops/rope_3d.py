"""3D Rotary Position Embeddings — MLX port of VJEPA2RopeAttention's rotary path.

Verbatim translation of transformers `rotate_queries_or_keys` + `apply_rotary_
embeddings` + `get_position_ids`. The crux op (parity Gate A).

Per-axis split of head_dim (64): d_dim = h_dim = w_dim = 20 (= 2*((64//3)//2)),
remainder 4 passthrough. Position ids decompose the flattened token index over a
(depth, height, width) grid using config grid_size (16) — NOT the actual input
spatial size.

rotate() quirk to preserve exactly: cos/sin are concat-tiled (emb = [e, e]) so
index i uses freq[i mod d/2]; but the rotated half `y` is interleaved
(y[2j]=-x[2j+1], y[2j+1]=x[2j]). Match it, don't "fix" it.
"""

from __future__ import annotations

import mlx.core as mx


def rotate_queries_or_keys(x: mx.array, pos: mx.array) -> mx.array:
    """x: [..., N, D]; pos: [N] (encoder) or [B, 1, N] (predictor masks)."""
    D = x.shape[-1]
    omega = mx.arange(D // 2).astype(mx.float32) / (D / 2.0)
    omega = 1.0 / (10000.0**omega)                      # [D/2]
    freq = pos.astype(mx.float32)[..., None] * omega    # [..., N, D/2]
    emb_sin = mx.concatenate([mx.sin(freq), mx.sin(freq)], axis=-1)  # [N, D]
    emb_cos = mx.concatenate([mx.cos(freq), mx.cos(freq)], axis=-1)  # [N, D]

    y = x.reshape(*x.shape[:-1], D // 2, 2)
    y1 = y[..., 0]
    y2 = y[..., 1]
    y = mx.stack([-y2, y1], axis=-1).reshape(x.shape)
    return x * emb_cos + y * emb_sin


def apply_rotary_embeddings(qk: mx.array, pos_ids, dims=(20, 20, 20)) -> mx.array:
    """qk: [B, H, N, head_dim]; pos_ids = (d, h, w) each [N]."""
    d_dim, h_dim, w_dim = dims
    pos_d, pos_h, pos_w = pos_ids
    s = 0
    qkd = rotate_queries_or_keys(qk[..., s:s + d_dim], pos_d); s += d_dim
    qkh = rotate_queries_or_keys(qk[..., s:s + h_dim], pos_h); s += h_dim
    qkw = rotate_queries_or_keys(qk[..., s:s + w_dim], pos_w); s += w_dim
    parts = [qkd, qkh, qkw]
    if s < qk.shape[-1]:
        parts.append(qk[..., s:])
    return mx.concatenate(parts, axis=-1)


def get_position_ids(num_tokens: int | None = None, grid_size: int = 16, masks=None):
    """Decompose token index over (depth, height, width) using config grid_size.

    Encoder: masks=None -> ids = arange(num_tokens), returns [N] each.
    Predictor: masks = position_masks [B, N] (actual token indices to keep) ->
    returns [B, 1, N] each (broadcast over heads), matching upstream
    masks.unsqueeze(1).repeat(1, num_heads, 1) up to the redundant head repeat.
    """
    if masks is None:
        ids = mx.arange(num_tokens)
    else:
        ids = masks[:, None, :]
    tpf = grid_size * grid_size
    frame = ids // tpf
    height = (ids - tpf * frame) // grid_size
    width = (ids - tpf * frame) - grid_size * height
    return frame, height, width
