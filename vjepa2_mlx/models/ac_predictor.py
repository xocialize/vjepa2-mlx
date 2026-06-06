"""VisionTransformerPredictorAC — MLX port of the action-conditioned predictor.

Isomorphic to facebookresearch/vjepa2 src/models/ac_predictor.py + the ACBlock /
ACRoPEAttention / build_action_block_causal_attention_mask in
src/models/utils/modules.py. The world-model predictor: given encoder context
tokens + per-frame robot actions/states (7-DoF), predicts future latent states
with frame-causal attention and 3D-RoPE.

Reuses ops.rope_3d.rotate_queries_or_keys (the AC rope == the encoder rope, the
documented concat-tiled quirk). Fused qkv (unlike the encoder's separate q/k/v).
Parity-locked vs the upstream torch code (structural; real weights via the
11.76 GB vjepa2-ac-vitg.pt checkpoint for production parity).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..ops.rope_3d import rotate_queries_or_keys

NEG_INF = -1e9


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1) -> mx.array:
    """Frame-causal additive mask [N, N]: token in frame t1 attends to all
    tokens in frames t2 <= t1. Returns 0 (attend) / -inf (block)."""
    N_T = add_tokens + H * W
    N = T * N_T
    allow = [[False] * N for _ in range(N)]
    for t1 in range(T):
        for t2 in range(0, t1 + 1):
            for i in range(t1 * N_T, (t1 + 1) * N_T):
                for j in range(t2 * N_T, (t2 + 1) * N_T):
                    allow[i][j] = True
    m = mx.array(allow)
    return mx.where(m, mx.zeros((N, N)), mx.full((N, N), NEG_INF))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class ACRoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, grid_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.grid_size = grid_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = self.d_dim
        self.w_dim = self.d_dim

    def _separate_positions(self, ids, H, W):
        tpf = H * W
        frame = ids // tpf
        height = (ids - tpf * frame) // W
        width = (ids - tpf * frame) - W * height
        return frame * 1.0, height * 1.0, width * 1.0

    def _qkv(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)            # [3, B, heads, N, hd]
        return qkv[0], qkv[1], qkv[2]

    def __call__(self, x, attn_mask, T, H, W, action_tokens):
        B, N, C = x.shape
        ids = mx.arange(int(T * H * W))
        d_mask, h_mask, w_mask = self._separate_positions(ids, H, W)
        h_mask = h_mask * (self.grid_size / H)
        w_mask = w_mask * (self.grid_size / W)

        # -- action tokens: temporal-only RoPE --
        x4 = x.reshape(B, T, action_tokens + H * W, C)
        aq, ak, av = [], [], []
        for i in range(action_tokens):
            a = x4[:, :, i:i + 1, :].reshape(B, T, C)
            q, k, v = self._qkv(a)                    # [B, heads, T, hd]
            tpos = mx.arange(T)
            qd = rotate_queries_or_keys(q[..., :self.d_dim], tpos)
            kd = rotate_queries_or_keys(k[..., :self.d_dim], tpos)
            qq = mx.concatenate([qd, q[..., self.d_dim:]], axis=-1)
            kk = mx.concatenate([kd, k[..., self.d_dim:]], axis=-1)
            aq.append(qq.reshape(B, self.num_heads, T, 1, -1))
            ak.append(kk.reshape(B, self.num_heads, T, 1, -1))
            av.append(v.reshape(B, self.num_heads, T, 1, -1))
        aq = mx.concatenate(aq, axis=3).reshape(B, self.num_heads, T * action_tokens, -1)
        ak = mx.concatenate(ak, axis=3).reshape(B, self.num_heads, T * action_tokens, -1)
        av = mx.concatenate(av, axis=3).reshape(B, self.num_heads, T * action_tokens, -1)

        # -- frame (patch) tokens: full 3D RoPE --
        xp = x4[:, :, action_tokens:, :].reshape(B, T * H * W, C)
        q, k, v = self._qkv(xp)
        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], d_mask); s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], h_mask); s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], w_mask); s += self.w_dim
        qparts = [qd, qh, qw] + ([q[..., s:]] if s < self.head_dim else [])
        kparts = [kd, kh, kw] + ([k[..., s:]] if s < self.head_dim else [])
        q = mx.concatenate(qparts, axis=-1)
        k = mx.concatenate(kparts, axis=-1)

        # -- merge action + patch tokens back to interleaved per-frame order --
        def merge(tx, ta):
            tx = tx.reshape(B, self.num_heads, T, H * W, -1)
            ta = ta.reshape(B, self.num_heads, T, action_tokens, -1)
            return mx.concatenate([ta, tx], axis=3).reshape(B, self.num_heads, T * (action_tokens + H * W), -1)

        q, k, v = merge(q, aq), merge(k, ak), merge(v, av)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class ACBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, eps, grid_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = ACRoPEAttention(dim, num_heads, grid_size=grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def __call__(self, x, attn_mask, T, H, W, action_tokens):
        x = x + self.attn(self.norm1(x), attn_mask, T, H, W, action_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerPredictorAC(nn.Module):
    def __init__(self, img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2,
                 embed_dim=1408, predictor_embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4.0, action_embed_dim=7, use_extrinsics=False, eps=1e-6):
        super().__init__()
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.tubelet_size = tubelet_size
        self.use_extrinsics = use_extrinsics
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim)
        self.predictor_blocks = [
            ACBlock(predictor_embed_dim, num_heads, mlp_ratio, eps, self.grid_height)
            for _ in range(depth)
        ]
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=eps)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def __call__(self, x, actions, states, extrinsics=None):
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.shape
        HW = self.grid_height * self.grid_width
        T = N_ctxt // HW
        s = self.state_encoder(states)[:, :, None, :]
        a = self.action_encoder(actions)[:, :, None, :]
        x = x.reshape(B, T, HW, D)
        cond = 3 if self.use_extrinsics else 2
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics)[:, :, None, :]
            x = mx.concatenate([a, s, e, x], axis=2).reshape(B, T * (HW + 3), D)
        else:
            x = mx.concatenate([a, s, x], axis=2).reshape(B, T * (HW + 2), D)

        Nmask = x.shape[1]
        attn_mask = build_action_block_causal_attention_mask(
            T, self.grid_height, self.grid_width, add_tokens=cond)[:Nmask, :Nmask]

        for blk in self.predictor_blocks:
            x = blk(x, attn_mask, T, self.grid_height, self.grid_width, cond)

        x = x.reshape(B, T, cond + HW, D)[:, :, cond:, :].reshape(B, T * HW, D)
        x = self.predictor_norm(x)
        return self.predictor_proj(x)
