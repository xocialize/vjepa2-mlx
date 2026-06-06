# V-JEPA2-ViT-L-MLX — Benchmarks (M5 Max, fp16, GPU)

Mean of 3. Encoder = video → token embeddings; tokens = (T/2)·16·16.


| Task | Frames | Tokens | Latency | Peak mem |
|---|---|---|---|---|
| encoder embed | 8 | 1024 | 34 ms | 1181 MB |
| encoder embed | 16 | 2048 | 66 ms | 1485 MB |
| encoder embed | 32 | 4096 | 147 ms | 1817 MB |
| classify (SSv2) | 16 | 2048 | 71 ms | 2467 MB |
