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
    def __init__(self, config: VJEPA2Config, hidden_size: int, mlp_ratio: float = 4.0):
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

    def __call__(self, x, position_mask=None):
        B, N, _ = x.shape

        def split(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = split(self.query(x))
        k = split(self.key(x))
        v = split(self.value(x))

        if position_mask is None:
            pos = get_position_ids(N, grid_size=self.grid_size)
        else:
            pos = get_position_ids(grid_size=self.grid_size, masks=position_mask)
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

    def __call__(self, x, position_mask=None):
        x = x + self.attention(self.norm1(x), position_mask=position_mask)
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


# --- Predictor (JEPA latent prediction) -------------------------------------

def apply_masks(tensor: mx.array, masks: list) -> mx.array:
    """tensor [B,N,D]; masks list of [B,n] index tensors. Gather + cat on batch."""
    out = []
    for mask in masks:
        idx = mx.broadcast_to(mask[..., None], (mask.shape[0], mask.shape[1], tensor.shape[-1]))
        out.append(mx.take_along_axis(tensor, idx, axis=1))
    return mx.concatenate(out, axis=0)


class VJEPA2PredictorEmbeddings(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.predictor_embeddings = nn.Linear(config.hidden_size, config.pred_hidden_size)
        self.num_mask_tokens = config.pred_num_mask_tokens
        self.mask_tokens = mx.zeros((self.num_mask_tokens, 1, 1, config.pred_hidden_size))

    def __call__(self, hidden_states, context_mask, target_mask, mask_index=1):
        B = hidden_states.shape[0]
        context = self.predictor_embeddings(hidden_states)
        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]                 # [1, 1, pred_hidden]
        max_patch_num = int(target_mask[0].max().item()) + 1
        target = mx.broadcast_to(target, (B, max_patch_num, target.shape[-1]))
        target = apply_masks(target, target_mask)
        context = mx.concatenate([context] * len(context_mask), axis=0)
        embeddings = mx.concatenate([context, target], axis=1)
        cm = mx.concatenate(list(context_mask), axis=0)
        tm = mx.concatenate(list(target_mask), axis=0)
        masks = mx.concatenate([cm, tm], axis=1)
        return embeddings, masks


class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.embeddings = VJEPA2PredictorEmbeddings(config)
        self.layer = [
            VJEPA2Layer(config, config.pred_hidden_size,
                        config.pred_num_attention_heads, config.pred_mlp_ratio)
            for _ in range(config.pred_num_hidden_layers)
        ]
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(config.pred_hidden_size, config.hidden_size)

    @staticmethod
    def _gather_rows(x, order):  # x [B,N,D], order [B,N] -> reordered rows
        idx = mx.broadcast_to(order[..., None], (*order.shape, x.shape[-1]))
        return mx.take_along_axis(x, idx, axis=1)

    def __call__(self, encoder_hidden_states, context_mask, target_mask):
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        N_ctxt = encoder_hidden_states.shape[1]
        hidden, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

        argsort = mx.argsort(position_masks, axis=1)
        position_masks = mx.take_along_axis(position_masks, argsort, axis=1)
        hidden = self._gather_rows(hidden, argsort)

        for layer in self.layer:
            hidden = layer(hidden, position_mask=position_masks)

        hidden = self.layernorm(hidden)
        reverse = mx.argsort(argsort, axis=1)
        hidden = self._gather_rows(hidden, reverse)
        hidden = hidden[:, N_ctxt:]
        return self.proj(hidden)


# --- Attentive pooler + classifier (classification checkpoints) -------------

class VJEPA2PoolerSelfAttention(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x):
        B, N, _ = x.shape

        def split(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        o = mx.fast.scaled_dot_product_attention(
            split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)), scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, self.num_heads * self.head_dim)
        return self.out_proj(o)


class VJEPA2PoolerCrossAttention(nn.Module):
    """Cross-attention — no output projection (matches upstream)."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, queries, keys, values):
        B, Q, E = queries.shape
        Nkv = keys.shape[1]
        q = self.q_proj(queries).reshape(B, Q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(keys).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(values).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return o.transpose(0, 2, 1, 3).reshape(B, Q, E)


class VJEPA2PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VJEPA2PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, config.hidden_size)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VJEPA2PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = VJEPA2PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, config.hidden_size)

    def __call__(self, queries, hidden_state):
        h = self.layer_norm1(hidden_state)
        x = queries + self.cross_attn(queries, h, h)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VJEPA2AttentivePooler(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.query_tokens = mx.zeros((1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
        self.self_attention_layers = [
            VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)
        ]

    def __call__(self, hidden_state):
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)
        B = hidden_state.shape[0]
        queries = mx.broadcast_to(self.query_tokens, (B, 1, self.query_tokens.shape[-1]))
        hidden_state = self.cross_attention_layer(queries, hidden_state)
        return hidden_state[:, 0]


class VJEPA2ForVideoClassification(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.vjepa2 = VJEPA2Model(config)
        self.pooler = VJEPA2AttentivePooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, video_btchw):
        h = self.vjepa2(video_btchw)
        return self.classifier(self.pooler(h))
