"""P4c — AC (action-conditioned) predictor structural parity vs the upstream
facebookresearch/vjepa2 torch code, on mx.cpu fp32.

Validates the full AC port logic (token interleaving, action-token temporal
RoPE, patch 3D-RoPE, frame-causal mask, fused qkv, merge, SDPA, proj) with
injected random weights at a small config — no 11.76 GB checkpoint needed.
Real-weight (ViT-g) parity is a separate step gated on that download.

Requires torch + the cloned oracle at refs/vjepa2-orig (git clone
https://github.com/facebookresearch/vjepa2). Skips otherwise.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest

import mlx.core as mx

from vjepa2_mlx.models.ac_predictor import VisionTransformerPredictorAC
from vjepa2_mlx.utils.convert import convert_state_dict

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG = os.path.join(ROOT, "refs", "vjepa2-orig")


def test_ac_predictor_structural_parity():
    if not os.path.isdir(os.path.join(ORIG, "src")):
        pytest.skip("oracle repo not cloned (refs/vjepa2-orig)")
    import sys
    sys.path.insert(0, ORIG)
    from src.models.ac_predictor import VisionTransformerPredictorAC as TorchAC

    cfg = dict(img_size=(64, 64), patch_size=16, num_frames=4, tubelet_size=2,
               embed_dim=64, predictor_embed_dim=48, depth=2, num_heads=4,
               mlp_ratio=4, action_embed_dim=7, use_extrinsics=False)
    torch.manual_seed(0)
    tm = TorchAC(norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
                 qkv_bias=True, **cfg).eval()
    sd = {k: v.detach().numpy() for k, v in tm.state_dict().items()}

    mm = VisionTransformerPredictorAC(img_size=(64, 64), patch_size=16, num_frames=4,
                                      tubelet_size=2, embed_dim=64, predictor_embed_dim=48,
                                      depth=2, num_heads=4, mlp_ratio=4,
                                      action_embed_dim=7, eps=1e-6)
    mm.load_weights(list(convert_state_dict(sd).items()), strict=True)
    mx.eval(mm.parameters())

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 32, 64)).astype(np.float32)
    act = rng.standard_normal((1, 2, 7)).astype(np.float32)
    st = rng.standard_normal((1, 2, 7)).astype(np.float32)
    with torch.no_grad():
        yt = tm(torch.from_numpy(x), torch.from_numpy(act), torch.from_numpy(st)).numpy()
    with mx.stream(mx.cpu):
        ym = np.array(mm(mx.array(x), mx.array(act), mx.array(st)))
    err = float(np.max(np.abs(ym - yt)))
    print(f"\nAC predictor structural parity max_abs={err:.3e}")
    assert ym.shape == yt.shape
    assert err < 1e-4
