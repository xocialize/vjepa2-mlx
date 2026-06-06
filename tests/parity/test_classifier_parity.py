"""P4b Gate C — attentive-pooler + classifier (SSv2) vs PyTorch, on mx.cpu fp32.

Encoder → 3 self-attn pooler layers → cross-attn pooling w/ query token →
linear head (174 SSv2 classes). The classification checkpoint's VJEPA2Model
carries the unused predictor (`vjepa2.predictor.*`) — filtered on load.
Needs goldens/vitl-ssv2-classifier.npz + torch/transformers.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

import mlx.core as mx

from vjepa2_mlx.config import VJEPA2Config
from vjepa2_mlx.models.modeling_vjepa2 import VJEPA2ForVideoClassification
from vjepa2_mlx.utils.convert import convert_state_dict

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, "goldens", "vitl-ssv2-classifier.npz")
REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def test_classifier_parity():
    if not os.path.exists(GOLDEN):
        pytest.skip("missing classifier golden")
    from transformers import VJEPA2ForVideoClassification as HF
    hf = HF.from_pretrained(REPO, dtype=torch.float32)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    weights = convert_state_dict(sd)
    weights = {k: v for k, v in weights.items() if not k.startswith("vjepa2.predictor.")}

    g = np.load(GOLDEN)
    cfg = replace(VJEPA2Config(), num_labels=int(g["num_labels"][0]), frames_per_clip=16)
    model = VJEPA2ForVideoClassification(cfg)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    with mx.stream(mx.cpu):
        logits = model(mx.array(g["input_video"]))
        mx.eval(logits)
    out = np.array(logits)
    ref = g["logits"]
    assert out.shape == ref.shape
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))
    print(f"\nclassifier logits: rel={rel:.3e} abs_max={np.max(np.abs(out - ref)):.3e}; "
          f"argmax mlx={int(out.argmax())} torch={int(ref.argmax())}")
    assert int(out.argmax()) == int(ref.argmax())
    assert rel < 1e-3
