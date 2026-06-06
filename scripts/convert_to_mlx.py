"""P5: HF checkpoint -> dist/<name>/{model.safetensors, config.json}.

Converts via utils.convert (Conv3d 5D transpose; Linear/LN identity; keys match
the MLX tree). mx.eval all (lazy-zero guard). Ships fp16 by default (ViT-L ~326M,
fp32 ~1.3 GB; fp16 ~650 MB) — validated against the fp32 golden separately.

Targets:
  base       = facebook/vjepa2-vitl-fpc64-256  -> V-JEPA2-vitl-fpc64-256
               (encoder.* + predictor.* — the embedding model + JEPA world-model)
  classifier = facebook/vjepa2-vitl-fpc16-256-ssv2 -> V-JEPA2-vitl-fpc16-256-ssv2
               (vjepa2.encoder.* + pooler.* + classifier.*; drops unused predictor)

  python scripts/convert_to_mlx.py [base|classifier|all]
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

import mlx.core as mx
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vjepa2_mlx.config import VJEPA2Config  # noqa: E402
from vjepa2_mlx.utils.convert import convert_fair_encoder, convert_state_dict  # noqa: E402

DIST = "dist"


def _save(name: str, weights: dict, cfg: VJEPA2Config) -> None:
    outdir = os.path.join(DIST, name)
    os.makedirs(outdir, exist_ok=True)
    weights = {k: v.astype(mx.float16) for k, v in weights.items()}
    mx.eval(weights)
    st = os.path.join(outdir, "model.safetensors")
    mx.save_safetensors(st, weights, metadata={"format": "mlx"})
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"OK {name}: {len(weights)} tensors  {os.path.getsize(st)/1e6:.0f} MB")


def convert_base() -> None:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    _save("V-JEPA2-vitl-fpc64-256", convert_state_dict(sd), VJEPA2Config())


def convert_classifier() -> None:
    from transformers import VJEPA2ForVideoClassification
    m = VJEPA2ForVideoClassification.from_pretrained(
        "facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=torch.float32)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    weights = {k: v for k, v in convert_state_dict(sd).items()
               if not k.startswith("vjepa2.predictor.")}
    cfg = replace(VJEPA2Config(), num_labels=m.config.num_labels, frames_per_clip=16)
    _save("V-JEPA2-vitl-fpc16-256-ssv2", weights, cfg)


def convert_ac() -> None:
    """vjepa2-ac-vitg.pt (encoder + AC predictor) -> one fp16 safetensors.
    Encoder under encoder.* (fused-qkv split); AC predictor keys as-is."""
    ck = torch.load("weights/vjepa2-ac-vitg.pt", map_location="cpu", weights_only=False)
    clean = lambda sd: {k.replace("module.", "").replace("backbone.", ""): v.detach().numpy()
                        for k, v in sd.items()}
    weights = {}
    weights.update(convert_fair_encoder(clean(ck["encoder"])))
    weights.update(convert_state_dict(clean(ck["predictor"])))
    from dataclasses import replace as _replace
    cfg = _replace(VJEPA2Config(), hidden_size=1408, num_hidden_layers=40,
                   num_attention_heads=22, mlp_ratio=48 / 11,
                   hf_repo="facebook/vjepa2-ac-vitg")
    _save("V-JEPA2-AC-vitg", weights, cfg)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("base", "all"):
        convert_base()
    if which in ("classifier", "all"):
        convert_classifier()
    if which in ("ac",):
        convert_ac()


if __name__ == "__main__":
    main()
