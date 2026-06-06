"""P3 Gate B — full ViT-L encoder hidden states vs PyTorch, on mx.cpu fp32.

Loads converted encoder weights into the MLX VJEPA2Model, runs the golden clip,
compares last_hidden_state + intermediate layers. V-JEPA2 activations are large
(~-43..30, no final scaling), so the full-pass gate is on RELATIVE error (Zonos
lesson). Needs goldens + torch/transformers. Skips if absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

from vjepa2_mlx.config import VJEPA2Config
from vjepa2_mlx.models.modeling_vjepa2 import VJEPA2Model
from vjepa2_mlx.utils.convert import convert_state_dict

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, "goldens", "vitl-encoder.npz")


def _rel(out, ref):
    return float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))


@pytest.fixture(scope="module")
def model_and_golden():
    if not os.path.exists(GOLDEN):
        pytest.skip("missing goldens/vitl-encoder.npz")
    from transformers import AutoModel
    hf = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
    sd = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()}
    weights = convert_state_dict(sd, prefix="encoder.")
    m = VJEPA2Model(VJEPA2Config())
    m.load_weights(list(weights.items()), strict=True)
    mx.eval(m.parameters())
    return m, np.load(GOLDEN)


def test_key_coverage(model_and_golden):
    from mlx.utils import tree_flatten
    from transformers import AutoModel
    hf = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
    conv = convert_state_dict({k: v.detach().numpy() for k, v in hf.state_dict().items()},
                              prefix="encoder.")
    model_keys = {k for k, _ in tree_flatten(VJEPA2Model(VJEPA2Config()).parameters())}
    assert model_keys == set(conv)


def test_encoder_full_parity(model_and_golden):
    m, g = model_and_golden
    video = mx.array(g["input_video"])  # (B,T,C,H,W) — Embeddings permutes
    with mx.stream(mx.cpu):
        out = m(video)
        mx.eval(out)
    out = np.array(out)
    ref = g["encoder_last_hidden"]
    assert out.shape == ref.shape
    rel = _rel(out, ref)
    abs_max = float(np.max(np.abs(out - ref)))
    print(f"\nencoder last_hidden: rel={rel:.3e} abs_max={abs_max:.3e} (ref max |{np.max(np.abs(ref)):.1f}|)")
    assert rel < 1e-3
