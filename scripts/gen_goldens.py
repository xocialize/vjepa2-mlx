"""P1: PyTorch (transformers) parity oracle for V-JEPA2 ViT-L.

Loads facebook/vjepa2-vitl-fpc64-256 (encoder + predictor ship together), runs a
small seeded clip on CPU fp32, and captures → goldens/vitl-encoder.npz:
  - input_video (B,T,C,H,W)
  - patch_embed: encoder.embeddings output (patch tokens)        [Gate A]
  - rope: seeded qk + pos_ids + apply_rotary_embeddings output   [Gate A]
  - layer0 / layer11 / final encoder last_hidden_state           [Gate B]

Uses a 4-frame, full-256² clip so the 3D-RoPE position grid matches config
(grid_size 16, depth = 4/tubelet 2 = 2 → 512 tokens). Calls the model's own
methods so the golden is exactly upstream.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel

SEED = 1234
OUT = "goldens"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32).eval()
    enc = model.encoder
    g: dict[str, np.ndarray] = {}

    # input: (B, T, C, H, W) — 4 frames at 256x256
    video = rng.standard_normal((1, 4, 3, 256, 256)).astype(np.float32)
    g["input_video"] = video
    vt = torch.from_numpy(video)

    with torch.no_grad():
        # --- patch embed (Gate A) ---
        patch = enc.embeddings(vt)            # (1, 512, 1024)
        g["patch_embed"] = patch.numpy()

        # --- 3D RoPE op (Gate A): seeded qk through the real attn0 ---
        attn0 = enc.layer[0].attention
        N = patch.shape[1]
        qk = rng.standard_normal((1, attn0.num_attention_heads, N, attn0.attention_head_size)).astype(np.float32)
        qkt = torch.from_numpy(qk)
        pos_ids = attn0.get_position_ids(torch.zeros(1, N, 1))   # (d,h,w) each [N]
        rot = attn0.apply_rotary_embeddings(qkt, pos_ids)
        g["rope_qk"] = qk
        g["rope_pos_d"] = np.asarray(pos_ids[0]).astype(np.int64)
        g["rope_pos_h"] = np.asarray(pos_ids[1]).astype(np.int64)
        g["rope_pos_w"] = np.asarray(pos_ids[2]).astype(np.int64)
        g["rope_out"] = rot.numpy()
        g["rope_dims"] = np.array([attn0.d_dim, attn0.h_dim, attn0.w_dim], dtype=np.int64)

        # --- per-layer + full encoder (Gate B) ---
        caps = {}
        h0 = enc.layer[0].register_forward_hook(lambda m, i, o: caps.__setitem__("l0", o[0].numpy()))
        h11 = enc.layer[11].register_forward_hook(lambda m, i, o: caps.__setitem__("l11", o[0].numpy()))
        out = enc(vt).last_hidden_state
        h0.remove(); h11.remove()
        g["layer0"] = caps["l0"]
        g["layer11"] = caps["l11"]
        g["encoder_last_hidden"] = out.numpy()

        # --- predictor (P4): synthetic context/target masks ---
        Ntok = out.shape[1]
        n_ctx = 3 * Ntok // 4
        ctx = [torch.arange(0, n_ctx).unsqueeze(0)]
        tgt = [torch.arange(n_ctx, Ntok).unsqueeze(0)]
        pred = model.predictor(out, ctx, tgt).last_hidden_state
        g["pred_context_mask"] = ctx[0].numpy().astype(np.int64)
        g["pred_target_mask"] = tgt[0].numpy().astype(np.int64)
        g["predictor_out"] = pred.numpy()

    np.savez(os.path.join(OUT, "vitl-encoder.npz"), **g)
    print("OK goldens/vitl-encoder.npz:")
    for k, v in g.items():
        print(f"   {k:22s} {tuple(v.shape)} {v.dtype}")
    lh = g["encoder_last_hidden"]
    print(f"   last_hidden range [{lh.min():.4f}, {lh.max():.4f}]")


if __name__ == "__main__":
    main()
