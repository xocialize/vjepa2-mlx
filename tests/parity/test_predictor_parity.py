"""P4 Gate C — JEPA predictor vs PyTorch, on mx.cpu fp32.

Exercises the masked-prediction path: apply_masks, mask tokens, sort/unsort by
position, 12 predictor layers with RoPE driven by position masks, proj 384->1024.
Uses the golden encoder output + synthetic context/target masks. Needs goldens +
torch/transformers. Skips if absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

from vjepa2_mlx.config import VJEPA2Config
from vjepa2_mlx.models.modeling_vjepa2 import VJEPA2Predictor
from vjepa2_mlx.utils.convert import convert_state_dict

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, "goldens", "vitl-encoder.npz")


def test_predictor_parity():
    if not os.path.exists(GOLDEN):
        pytest.skip("missing goldens")
    from transformers import AutoModel
    hf = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    weights = convert_state_dict(sd, prefix="predictor.")
    weights = {k[len("predictor."):]: v for k, v in weights.items()}  # strip for standalone

    pred = VJEPA2Predictor(VJEPA2Config())
    pred.load_weights(list(weights.items()), strict=True)
    mx.eval(pred.parameters())

    g = np.load(GOLDEN)
    enc = mx.array(g["encoder_last_hidden"])
    ctx = [mx.array(g["pred_context_mask"])]
    tgt = [mx.array(g["pred_target_mask"])]
    ref = g["predictor_out"]

    with mx.stream(mx.cpu):
        out = pred(enc, ctx, tgt)
        mx.eval(out)
    out = np.array(out)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))
    abs_max = float(np.max(np.abs(out - ref)))
    print(f"\npredictor: rel={rel:.3e} abs_max={abs_max:.3e} (ref max |{np.max(np.abs(ref)):.2f}|)")
    assert rel < 1e-3
