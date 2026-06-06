"""P2 Gate A — 3D-RoPE + 3D patch embed vs PyTorch, on mx.cpu fp32.

The crux ops. RoPE is pure (golden qk/pos/out). Patch-embed loads the real
Conv3d weight from the cached HF checkpoint. Threshold <1e-4. Needs
goldens/vitl-encoder.npz + torch/transformers. Skips if absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

from vjepa2_mlx.config import VJEPA2Config
from vjepa2_mlx.ops.rope_3d import apply_rotary_embeddings, get_position_ids
from vjepa2_mlx.ops.patch_embed_3d import PatchEmbed3D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, "goldens", "vitl-encoder.npz")


def _g():
    if not os.path.exists(GOLDEN):
        pytest.skip("missing goldens/vitl-encoder.npz")
    return np.load(GOLDEN)


def test_rope_parity():
    g = _g()
    qk = mx.array(g["rope_qk"])
    pos = (mx.array(g["rope_pos_d"]), mx.array(g["rope_pos_h"]), mx.array(g["rope_pos_w"]))
    dims = tuple(int(x) for x in g["rope_dims"])
    with mx.stream(mx.cpu):
        out = apply_rotary_embeddings(qk, pos, dims=dims)
        mx.eval(out)
    err = float(np.max(np.abs(np.array(out) - g["rope_out"])))
    print(f"\n3D-RoPE max_abs={err:.3e}")
    assert err < 1e-4


def test_position_ids_match():
    g = _g()
    N = g["rope_pos_d"].shape[0]
    d, h, w = get_position_ids(N, grid_size=16)
    assert np.array_equal(np.array(d), g["rope_pos_d"])
    assert np.array_equal(np.array(h), g["rope_pos_h"])
    assert np.array_equal(np.array(w), g["rope_pos_w"])


def test_patch_embed_parity():
    g = _g()
    torch = pytest.importorskip("torch")
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32).eval()
    proj = m.encoder.embeddings.patch_embeddings.proj
    w = proj.weight.detach().numpy()   # (O,I,kT,kH,kW)
    b = proj.bias.detach().numpy()

    pe = PatchEmbed3D(VJEPA2Config())
    pe.proj.weight = mx.array(np.transpose(w, (0, 2, 3, 4, 1)))  # (O,kT,kH,kW,I)
    pe.proj.bias = mx.array(b)
    mx.eval(pe.parameters())

    video = g["input_video"]                       # (B,T,C,H,W)
    bthwc = mx.array(np.transpose(video, (0, 1, 3, 4, 2)))  # (B,T,H,W,C)
    with mx.stream(mx.cpu):
        out = pe(bthwc)
        mx.eval(out)
    err = float(np.max(np.abs(np.array(out) - g["patch_embed"])))
    print(f"\npatch_embed max_abs={err:.3e}")
    assert out.shape == g["patch_embed"].shape
    assert err < 1e-4
