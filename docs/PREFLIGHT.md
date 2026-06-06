# V-JEPA2 → MLX — PREFLIGHT

**Working dir:** `DEV_INT/vjepa2-mlx` · **Remote (planned):** `xocialize/vjepa2-mlx`
**Publish (planned):** `mlx-community/V-JEPA2-*` · **Upstream:** [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
**License:** MIT ✅ · **Tier:** 2 · **Started:** 2026-06-05
**Reference:** HF `transformers/models/vjepa2/modeling_vjepa2.py` (authoritative) + `gaarutyunov/vjepa2-mlx` (secondary cross-check)

> **STATUS (2026-06-05, PAUSED at P3):** P0–P3 done, **encoder parity-locked** — 3D-RoPE 2.4e-6, patch-embed 3.3e-5 (Gate A), full 24-layer encoder rel 2.66e-5 (Gate B). 10 tests. Educational/tradeshow-app port. Skill-worthy notes captured in `LESSONS.md`. Resume at **P4** (predictor + AC predictor + attentive pooler). See the memory file `vjepa2-mlx-port.md` for the authoritative tracker.

---

## CONFIRM gates (mlx-porting)

1. **License** ✅ **MIT** — verified the actual `facebookresearch/vjepa2/LICENSE` (© Meta Platforms). Commercial-OK.
2. **Port status** — an **MLX-Python port exists** (`gaarutyunov/vjepa2-mlx`: encoder + predictor + AC + attentive-pooler + 3D-RoPE + convert + tests, MIT, pushed 2025-11) but **experimental, no parity claim, no published `mlx-community` weights, no Swift, ViT-L only, 1★**. Gaps = **Swift** (none) + **published verified weights** (none). → SeedVR2-style reframe.
3. **Config truth (pinned, `config.py`)** — ViT-L: hidden 1024 · 24 layers · 16 heads · mlp_ratio 4 (interm 4096, head_dim 64) · patch 16 · **tubelet 2** · crop 256 · fpc 64 · gelu · LN-eps 1e-6 · qkv_bias. Predictor: 384 / 12 / 12 / 10 mask tokens. **3D-RoPE is NOT in config.json** — read from `VJEPA2RopeAttention` in P1 (resolved-config trap). Grid: depth 32 (=64/2), H=W=16.
4. **Tier** — Tier 2 (single ViT encoder; + predictor/pooler heads).

## Scope & target (locked)

- **Target: Python-first → Swift.** Build a parity-verified MLX-Python port + publish `mlx-community/V-JEPA2-*` weights (the missing artifact + the oracle), THEN the MLX-Swift on-device port (the strategic deliverable; nothing exists in Swift).
- **Scope: FULL** — encoder + attentive-pooler/classifier + predictor + AC predictor (world-model). AC is ViT-g-based; its checkpoint pinned in P1.

## Crux MLX surfaces

1. **3D-tubelet patch embed** — `nn.Conv3d` kernel=stride=(tubelet,patch,patch), NDHWC; weight (O,I,kT,kH,kW)→(O,kT,kH,kW,I). `ops/patch_embed_3d.py`.
2. **3D-RoPE** — head_dim split across (depth,H,W) axes, per-axis rotary; assembled in `ops/rope_3d.py`. The main lesson candidate. `mx.fast.rope` exists for 1D; 3D composition is custom.
3. **Attentive pooler** — cross-attention pooling head.

Reuse: `mx.fast.scaled_dot_product_attention`, `nn.LayerNorm`, gelu; parity-on-`mx.cpu`, `mx.eval`-before-save, conv-layout discipline from the realesrgan/rife ports.

## Plan / phases

- **P0 — Scaffold** ✅ (this commit): repo isomorphic to `modeling_vjepa2.py`, config registry, op + model skeletons, PREFLIGHT, pyproject. (MLX has `nn.Conv3d` + `mx.fast.rope`.)
- **P1 — Oracle**: install `transformers`, download checkpoints (encoder `vjepa2-vitl-fpc64-256` + classifier + AC), **pin 3D-RoPE + all configs from source**, CPU fp32 goldens per component.
- **P2 — Gate A**: patch-embed-3D + 3D-RoPE parity vs torch (crux ops).
- **P3 — Gate B**: full ViT-L encoder hidden-states parity (<1e-2; per-layer <5e-3).
- **P4 — Gate C**: predictor + AC predictor + attentive pooler parity.
- **P5 — Convert + pipeline**: HF→safetensors per component, `build_model`, video/image embedding + prediction CLI (256/64fpc preprocess).
- **P6 — Quant**: ViT-L ~300M → fp16/int8 (worthwhile; ~1.2 GB fp32).
- **P7 — Publish**: `mlx-community/V-JEPA2-*` + GitHub + M5 benchmarks.
- **P8 — MLX-Swift**: net-new on-device encoder (this Python port = oracle). The strategic deliverable.

## Open items for P1

- Pin exact 3D-RoPE (axis split / base / order) from `modeling_vjepa2.py`.
- Confirm classifier checkpoint (`vjepa2-vitl-fpc16-256-ssv2`?) + AC checkpoint (`vjepa2-ac-vitg-*`) ids + whether predictor weights ship with the encoder repo or separately.
- Decide published repo naming (`V-JEPA2-ViT-L` vs `V-JEPA2-vitl-fpc64-256`).
