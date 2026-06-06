"""V-JEPA2 config — pinned from facebook/vjepa2-vitl-fpc64-256 config.json +
the HF transformers modeling_vjepa2.py (the authoritative reference).

CONFIRM gate #3: defaults match the trained config. NOTE 3D-RoPE settings are
NOT serialized in config.json — they live in VJEPA2RopeAttention (grid over
depth=frames/tubelet, height, width); pinned from modeling_vjepa2.py in P1.

Scope: FULL — encoder + attentive pooler/classifier + predictor + AC predictor.
Checkpoints for the classifier + AC variants are pinned in P1 (AC is ViT-g based).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VJEPA2Config:
    # --- encoder (ViT-L/16) ---
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    mlp_ratio: int = 4                 # intermediate = 1024*4 = 4096
    patch_size: int = 16
    tubelet_size: int = 2
    crop_size: int = 256
    frames_per_clip: int = 64
    in_chans: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    # --- predictor (JEPA latent prediction) ---
    pred_hidden_size: int = 384
    pred_num_hidden_layers: int = 12
    pred_num_attention_heads: int = 12
    pred_mlp_ratio: float = 4.0
    pred_num_mask_tokens: int = 10
    # --- provenance ---
    hf_repo: str = "facebook/vjepa2-vitl-fpc64-256"

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.mlp_ratio

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # 64

    @property
    def grid_depth(self) -> int:
        return self.frames_per_clip // self.tubelet_size      # 32

    @property
    def grid_hw(self) -> int:
        return self.crop_size // self.patch_size              # 16


# Checkpoint registry (encoder pinned; others confirmed in P1)
CHECKPOINTS: dict[str, str] = {
    "vitl-encoder": "facebook/vjepa2-vitl-fpc64-256",
    # "vitl-ssv2-classifier": "facebook/vjepa2-vitl-fpc16-256-ssv2",  # confirm P1
    # "ac-vitg": "facebook/vjepa2-ac-vitg-...",                       # confirm P1
}

DEFAULT = "vitl-encoder"
