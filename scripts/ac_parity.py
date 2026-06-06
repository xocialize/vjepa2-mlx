"""P4c real-weight parity: ViT-g encoder + AC predictor vs the upstream torch
oracle, using the local vjepa2-ac-vitg.pt checkpoint (11.76 GB).

Builds the facebookresearch encoder (vit_giant_xformers) + ac_predictor with the
real weights as the oracle, runs a small 256² clip + synthetic 7-DoF actions,
then loads the same weights into the MLX VJEPA2Model (ViT-g cfg, fused-qkv split)
+ VisionTransformerPredictorAC and compares. Gate: relative error (large acts).

  python scripts/ac_parity.py     # needs refs/vjepa2-orig + weights/vjepa2-ac-vitg.pt
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, "refs/vjepa2-orig")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
from vjepa2_mlx.config import VJEPA2Config  # noqa: E402
from vjepa2_mlx.models.modeling_vjepa2 import VJEPA2Model  # noqa: E402
from vjepa2_mlx.models.ac_predictor import VisionTransformerPredictorAC  # noqa: E402
from vjepa2_mlx.utils.convert import convert_fair_encoder, convert_state_dict  # noqa: E402

CKPT = "weights/vjepa2-ac-vitg.pt"
SEED = 1234


def _clean(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


def main() -> None:
    from src.models.vision_transformer import vit_giant_xformers
    from src.models.ac_predictor import vit_ac_predictor

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc_sd = _clean(ck["encoder"]); pred_sd = _clean(ck["predictor"])
    rng = np.random.default_rng(SEED)

    # ---- torch oracle ----
    T_frames = 4
    enc = vit_giant_xformers(img_size=(256, 256), num_frames=T_frames, tubelet_size=2,
                             use_rope=True, use_silu=False, uniform_power=False).eval()
    enc.load_state_dict(enc_sd, strict=True)
    pred = vit_ac_predictor(img_size=(256, 256), num_frames=T_frames, tubelet_size=2,
                            embed_dim=enc.embed_dim, use_rope=True).eval()
    pred.load_state_dict(pred_sd, strict=True)

    video = rng.standard_normal((1, T_frames, 3, 256, 256)).astype(np.float32)  # (B,T,C,H,W) for MLX
    video_bcthw = np.transpose(video, (0, 2, 1, 3, 4))                          # (B,C,T,H,W) for torch
    actions = rng.standard_normal((1, T_frames // 2, 7)).astype(np.float32)
    states = rng.standard_normal((1, T_frames // 2, 7)).astype(np.float32)
    with torch.no_grad():
        enc_out = enc(torch.from_numpy(video_bcthw))     # [1, N, 1408]
        pred_out = pred(enc_out, torch.from_numpy(actions), torch.from_numpy(states))
    enc_out_np = enc_out.numpy(); pred_out_np = pred_out.numpy()
    print(f"oracle: encoder {enc_out_np.shape}, predictor {pred_out_np.shape}")

    # ---- MLX encoder ----
    vitg = replace(VJEPA2Config(), hidden_size=1408, num_hidden_layers=40,
                   num_attention_heads=22, mlp_ratio=48 / 11, frames_per_clip=T_frames)
    menc = VJEPA2Model(vitg)
    menc.load_weights(list(convert_fair_encoder(enc_sd).items()), strict=True)
    mx.eval(menc.parameters())
    with mx.stream(mx.cpu):
        m_enc = np.array(menc(mx.array(video)))
    print(f"ENCODER (ViT-g) rel={_rel(m_enc, enc_out_np):.3e}")

    # ---- MLX AC predictor ----
    mpred = VisionTransformerPredictorAC(img_size=(256, 256), num_frames=T_frames,
                                         tubelet_size=2, embed_dim=1408,
                                         predictor_embed_dim=1024, depth=24,
                                         num_heads=16, mlp_ratio=4, eps=1e-6)
    mpred.load_weights(list(convert_state_dict(pred_sd).items()), strict=True)
    mx.eval(mpred.parameters())
    with mx.stream(mx.cpu):
        m_pred = np.array(mpred(mx.array(enc_out_np), mx.array(actions), mx.array(states)))
    print(f"AC PREDICTOR (ViT-g) rel={_rel(m_pred, pred_out_np):.3e}")


if __name__ == "__main__":
    main()
