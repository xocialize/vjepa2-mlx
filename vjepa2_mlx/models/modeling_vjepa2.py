"""V-JEPA2 — MLX port, isomorphic to transformers/models/vjepa2/modeling_vjepa2.py.

Class names kept 1:1 with upstream for clean diffing (mlx-porting hard rule).
Skeleton only — full translation + parity is P3 (encoder, Gate B) / P4 (predictor,
AC predictor, attentive pooler, Gate C). Reference: HF transformers + the
secondary cross-check gaarutyunov/vjepa2-mlx.

Component map (upstream → here):
  VJEPA2PatchEmbeddings3D   -> ops/patch_embed_3d.PatchEmbed3D
  VJEPA2RopeAttention       -> ops/rope_3d + SDPA
  VJEPA2Layer / Encoder     -> VJEPA2Layer / VJEPA2Encoder
  VJEPA2Model               -> VJEPA2Model        (encoder -> hidden states)
  VJEPA2Predictor           -> VJEPA2Predictor    (JEPA latent prediction)
  VJEPA2AttentivePooler     -> VJEPA2AttentivePooler (+ classifier head)
  + action-conditioned predictor (world-model) for the AC checkpoint
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import VJEPA2Config


class VJEPA2Layer(nn.Module):
    """LayerNorm -> RoPE MHA -> LayerNorm -> MLP(gelu), pre-norm residual."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P3")

    def __call__(self, x, pos_ids):
        raise NotImplementedError("P3")


class VJEPA2Encoder(nn.Module):
    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P3: PatchEmbed3D + 24x VJEPA2Layer + final LN")

    def __call__(self, pixel_values_videos):
        raise NotImplementedError("P3")


class VJEPA2Model(nn.Module):
    """Bare encoder -> hidden states (the embedding model)."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        raise NotImplementedError("P3")


class VJEPA2Predictor(nn.Module):
    """JEPA latent-space predictor (pred_hidden=384, 12 layers)."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P4")


class VJEPA2AttentivePooler(nn.Module):
    """Cross-attention pooling + linear classifier head."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        raise NotImplementedError("P4")
