# vjepa2-mlx

Apple **MLX** port of Meta's [V-JEPA2](https://github.com/facebookresearch/vjepa2) —
video **Joint-Embedding Predictive Architecture** (ViT-L/16) for Apple Silicon. **MIT.**

Torch-free video/image **embedding extraction** (retrieval, classification, VLM
features), the **JEPA predictor** (masked latent-space world-model), and an **SSv2
action classifier** — all parity-locked to the PyTorch reference and HF auto-downloaded.

> **Status:** encoder + predictor + classifier ported & parity-locked; pipeline +
> fp16 weights ready. Action-conditioned (AC/robotics) predictor is a documented
> follow-up (separate ViT-g model, not in HF transformers). See [`docs/PREFLIGHT.md`](docs/PREFLIGHT.md).

## Usage

```bash
pip install vjepa2-mlx

# video / image -> 1024-d embedding (.npy)
vjepa2-mlx -i clip.mp4 --task embed -o emb.npy

# SSv2 action classification (top-5)
vjepa2-mlx -i clip.mp4 --task classify
```

```python
from vjepa2_mlx.pipeline_mlx import embed_video
emb = embed_video("clip.mp4", num_frames=16)   # (1024,) mean-pooled token embedding

from vjepa2_mlx.utils.weights import build_predictor
predictor = build_predictor()                  # JEPA latent-space world-model
```

Weights auto-download from `mlx-community/V-JEPA2-*` on first use.

## Components & parity (vs PyTorch, mx.cpu fp32)

| Component | Parity |
|---|---|
| 3D-tubelet Conv3d patch embed | 3.3e-5 |
| 3D-RoPE (per-axis depth/H/W) | 2.4e-6 |
| **ViT-L encoder** (24 layers) | rel 2.66e-5 |
| **JEPA predictor** (masked) | rel 1.67e-6 |
| **Attentive pooler + classifier** | logits rel 2.43e-6, argmax ✓ |

Shipped **fp16** (~650 MB): encoder rel 4.7e-3, classifier argmax matches.

## Benchmarks (M5 Max, fp16, GPU)

| Task | Frames | Latency |
|---|---|---|
| encoder embed | 8 / 16 / 32 | 34 / 66 / 147 ms |
| classify (SSv2) | 16 | 71 ms |

See [`docs/REPORT.md`](docs/REPORT.md).

## How it works

ViT-L/16 with a **3D-tubelet** Conv3d patch embed and **3D-RoPE** (head_dim split
across depth/height/width). The encoder, the masked JEPA predictor (mask tokens +
context/target masks + position-mask-driven RoPE), and the attentive-pooler classifier
are translated 1:1 from HF `transformers` and parity-locked. Two crux ops (3D-RoPE,
3D-tubelet Conv3d) are hand-rolled in MLX — see [`docs/LESSONS.md`](docs/LESSONS.md).

## License

MIT, inherited from upstream V-JEPA2 (© Meta Platforms). Weights converted from the
official `facebook/vjepa2-vitl-*` checkpoints.
