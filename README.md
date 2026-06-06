# vjepa2-mlx

Apple **MLX** port of Meta's [V-JEPA2](https://github.com/facebookresearch/vjepa2) —
video **Joint-Embedding Predictive Architecture** (ViT-L encoder + predictor) for
Apple Silicon. **MIT.**

Video/image embedding extraction (classification, retrieval, VLM features) plus the
JEPA predictor / action-conditioned world-model, torch-free with HF auto-download.

> **Status: P0 (scaffold).** Config pinned (ViT-L/16, 24 layers, 3D-tubelet patch
> embed, 3D-RoPE). Op translation, parity, conversion, and the MLX-Swift port are
> in progress. See [`docs/PREFLIGHT.md`](docs/PREFLIGHT.md).

## Scope

Full: **encoder** (embeddings) + **attentive-pooler/classifier** + **predictor** +
**action-conditioned predictor** (world-model). Python-first (parity-verified +
published `mlx-community` weights), then a net-new **MLX-Swift** on-device port.

## Why a port

- The only prior MLX work (`gaarutyunov/vjepa2-mlx`) is experimental — no parity
  claim, no published weights, **no Swift**.
- The strategic target is **on-device Swift** video embedding (no impl exists);
  this Python port is its parity oracle and the published-weights artifact.

## License

MIT, inherited from upstream V-JEPA2 (© Meta Platforms).
