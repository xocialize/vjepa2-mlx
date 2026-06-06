"""V-JEPA2 — MLX port, isomorphic to transformers/models/vjepa2/modeling_vjepa2.py.

Class / attribute names kept 1:1 with upstream (key-compatible loading). Encoder
path translated + parity-locked in P3 (Gate B). Predictor / attentive pooler /
AC predictor are P4.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import VJEPA2Config
from ..ops.patch_embed_3d import PatchEmbed3D
from ..ops.rope_3d import apply_rotary_embeddings, get_position_ids


class VJEPA2MLP(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int, mlp_ratio: float):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class VJEPA2RopeAttention(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.grid_size = config.crop_size // config.patch_size
        self.dims = (
            2 * ((self.head_dim // 3) // 2),
            2 * ((self.head_dim // 3) // 2),
            2 * ((self.head_dim // 3) // 2),
        )
        self.query = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x):
        B, N, _ = x.shape

        def split(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = split(self.query(x))
        k = split(self.key(x))
        v = split(self.value(x))

        pos = get_position_ids(N, grid_size=self.grid_size)
        k = apply_rotary_embeddings(k, pos, dims=self.dims)
        q = apply_rotary_embeddings(q, pos, dims=self.dims)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(o)


class VJEPA2Layer(nn.Module):
    def __init__(self, config: VJEPA2Config, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2RopeAttention(config, hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size, mlp_ratio)

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VJEPA2Embeddings(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbed3D(config)

    def __call__(self, video_btchw: mx.array) -> mx.array:
        """video_btchw: [B, T, C, H, W] (upstream layout)."""
        T = video_btchw.shape[1]
        if T < self.config.tubelet_size:
            reps = self.config.tubelet_size
            video_btchw = mx.repeat(video_btchw, reps, axis=1)
        bthwc = video_btchw.transpose(0, 1, 3, 4, 2)   # [B,T,H,W,C]
        return self.patch_embeddings(bthwc)


class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.embeddings = VJEPA2Embeddings(config)
        self.layer = [
            VJEPA2Layer(config, config.hidden_size, config.num_attention_heads, config.mlp_ratio)
            for _ in range(config.num_hidden_layers)
        ]
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, video_btchw: mx.array) -> mx.array:
        h = self.embeddings(video_btchw)
        for layer in self.layer:
            h = layer(h)
        return self.layernorm(h)


class VJEPA2Model(nn.Module):
    """Bare encoder -> last_hidden_state (the embedding model)."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.encoder = VJEPA2Encoder(config)

    def __call__(self, video_btchw: mx.array) -> mx.array:
        return self.encoder(video_btchw)


# --- P4 stubs ---------------------------------------------------------------

class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P4")


class VJEPA2AttentivePooler(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P4")
