"""P5: validate the published fp16 dist/ artifacts via the build_* loaders.

Confirms materialization + that fp16 weights stay within tolerance vs the fp32
goldens (encoder rel, classifier argmax). Skips if dist/ or goldens/ absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "dist", "V-JEPA2-vitl-fpc64-256")
CLS = os.path.join(ROOT, "dist", "V-JEPA2-vitl-fpc16-256-ssv2")
GENC = os.path.join(ROOT, "goldens", "vitl-encoder.npz")
GCLS = os.path.join(ROOT, "goldens", "vitl-ssv2-classifier.npz")


def test_fp16_encoder_artifact():
    if not (os.path.isdir(BASE) and os.path.exists(GENC)):
        pytest.skip("missing base dist/golden")
    from vjepa2_mlx.utils.weights import build_encoder
    g = np.load(GENC)
    m = build_encoder(weights_dir=BASE)
    with mx.stream(mx.cpu):
        out = np.array(m(mx.array(g["input_video"])))
    ref = g["encoder_last_hidden"]
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))
    assert not np.allclose(out, 0.0)
    print(f"\nfp16 encoder artifact rel={rel:.3e}")
    assert rel < 1e-2


def test_fp16_predictor_artifact_loads():
    if not os.path.isdir(BASE):
        pytest.skip("missing base dist")
    from vjepa2_mlx.utils.weights import build_predictor
    m = build_predictor(weights_dir=BASE)
    assert m is not None  # predictor.* keys present + load strict


def test_fp16_classifier_artifact():
    if not (os.path.isdir(CLS) and os.path.exists(GCLS)):
        pytest.skip("missing classifier dist/golden")
    from vjepa2_mlx.utils.weights import build_classifier
    g = np.load(GCLS)
    m = build_classifier(weights_dir=CLS)
    with mx.stream(mx.cpu):
        out = np.array(m(mx.array(g["input_video"])))
    print(f"\nfp16 classifier argmax mlx={int(out.argmax())} torch={int(g['logits'].argmax())}")
    assert int(out.argmax()) == int(g["logits"].argmax())
