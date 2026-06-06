"""P7: M5 Max benchmarks -> docs/REPORT.md. Encoder embedding latency at a few
frame counts + classifier, GPU fp16, mean of 3."""
from __future__ import annotations
import os, sys, time
import mlx.core as mx, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vjepa2_mlx.utils.weights import build_encoder, build_classifier

def _peak(): return mx.get_peak_memory()/1e6 if hasattr(mx,"get_peak_memory") else float("nan")

def main():
    rng = np.random.default_rng(0)
    enc = build_encoder(weights_dir="dist/V-JEPA2-vitl-fpc64-256")
    lines = ["# V-JEPA2-ViT-L-MLX — Benchmarks (M5 Max, fp16, GPU)\n",
             "Mean of 3. Encoder = video → token embeddings; tokens = (T/2)·16·16.\n",
             "\n| Task | Frames | Tokens | Latency | Peak mem |","|---|---|---|---|---|"]
    for T in (8, 16, 32):
        x = mx.array(rng.standard_normal((1, T, 3, 256, 256)).astype(np.float32))
        enc(x); mx.eval(enc(x))
        if hasattr(mx,"reset_peak_memory"): mx.reset_peak_memory()
        t0=time.perf_counter()
        for _ in range(3):
            h=enc(x); mx.eval(h)
        dt=(time.perf_counter()-t0)/3
        lines.append(f"| encoder embed | {T} | {(T//2)*256} | {dt*1000:.0f} ms | {_peak():.0f} MB |")
        print(lines[-1])
    clf = build_classifier(weights_dir="dist/V-JEPA2-vitl-fpc16-256-ssv2")
    x = mx.array(rng.standard_normal((1,16,3,256,256)).astype(np.float32))
    clf(x); mx.eval(clf(x))
    if hasattr(mx,"reset_peak_memory"): mx.reset_peak_memory()
    t0=time.perf_counter()
    for _ in range(3): mx.eval(clf(x))
    dt=(time.perf_counter()-t0)/3
    lines.append(f"| classify (SSv2) | 16 | 2048 | {dt*1000:.0f} ms | {_peak():.0f} MB |")
    print(lines[-1])
    os.makedirs("docs",exist_ok=True); open("docs/REPORT.md","w").write("\n".join(lines)+"\n")
    print("wrote docs/REPORT.md")

if __name__=="__main__": main()
