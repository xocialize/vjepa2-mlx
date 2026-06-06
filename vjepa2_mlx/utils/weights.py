"""Model build + weight load (HF auto-download / local override). P5.

Resolution per repo name: weights_dir arg | $VJEPA2_MLX_WEIGHTS_DIR/<name> |
dist/<name> | HF mlx-community/<name>. Each dir: model.safetensors + config.json.

  build_encoder()    -> VJEPA2Model  (embeddings; loads encoder.* from base)
  build_predictor()  -> VJEPA2Predictor (JEPA world-model; predictor.* from base)
  build_classifier() -> VJEPA2ForVideoClassification (ssv2 repo)
"""

from __future__ import annotations

import json
import os

import mlx.core as mx

from ..config import VJEPA2Config
from ..models.modeling_vjepa2 import (
    VJEPA2ForVideoClassification,
    VJEPA2Model,
    VJEPA2Predictor,
)

WEIGHTS_DIR_ENV = "VJEPA2_MLX_WEIGHTS_DIR"
HF_ORG = "mlx-community"
BASE = "V-JEPA2-vitl-fpc64-256"
CLASSIFIER = "V-JEPA2-vitl-fpc16-256-ssv2"


def _resolve_dir(name: str, weights_dir: str | None) -> str:
    if weights_dir:
        return weights_dir
    env = os.environ.get(WEIGHTS_DIR_ENV)
    if env and os.path.isdir(os.path.join(env, name)):
        return os.path.join(env, name)
    local = os.path.join("dist", name)
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=f"{HF_ORG}/{name}")


def _load(name: str, weights_dir: str | None):
    wdir = _resolve_dir(name, weights_dir)
    weights = dict(mx.load(os.path.join(wdir, "model.safetensors")).items())
    with open(os.path.join(wdir, "config.json")) as f:
        raw = json.load(f)
    cfg = VJEPA2Config(**{k: v for k, v in raw.items()
                          if k in VJEPA2Config.__dataclass_fields__})
    return weights, cfg


def _finalize(model, weights):
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def build_encoder(weights_dir: str | None = None) -> VJEPA2Model:
    weights, cfg = _load(BASE, weights_dir)
    weights = {k: v for k, v in weights.items() if k.startswith("encoder.")}
    return _finalize(VJEPA2Model(cfg), weights)


def build_predictor(weights_dir: str | None = None) -> VJEPA2Predictor:
    weights, cfg = _load(BASE, weights_dir)
    weights = {k[len("predictor."):]: v for k, v in weights.items()
               if k.startswith("predictor.")}
    return _finalize(VJEPA2Predictor(cfg), weights)


def build_classifier(weights_dir: str | None = None) -> VJEPA2ForVideoClassification:
    weights, cfg = _load(CLASSIFIER, weights_dir)
    return _finalize(VJEPA2ForVideoClassification(cfg), weights)


AC = "V-JEPA2-AC-vitg"


def build_ac_encoder(weights_dir: str | None = None) -> VJEPA2Model:
    """ViT-g encoder from the action-conditioned world-model checkpoint."""
    from dataclasses import replace
    weights, cfg = _load(AC, weights_dir)
    cfg = replace(cfg, hidden_size=1408, num_hidden_layers=40,
                  num_attention_heads=22, mlp_ratio=48 / 11)
    weights = {k: v for k, v in weights.items() if k.startswith("encoder.")}
    return _finalize(VJEPA2Model(cfg), weights)


def build_ac_predictor(weights_dir: str | None = None):
    """Action-conditioned predictor (world-model) — ViT-g (embed 1408)."""
    from ..models.ac_predictor import VisionTransformerPredictorAC
    weights, _ = _load(AC, weights_dir)
    weights = {k: v for k, v in weights.items() if not k.startswith("encoder.")}
    m = VisionTransformerPredictorAC(img_size=(256, 256), embed_dim=1408,
                                     predictor_embed_dim=1024, depth=24,
                                     num_heads=16, mlp_ratio=4, eps=1e-6)
    return _finalize(m, weights)
