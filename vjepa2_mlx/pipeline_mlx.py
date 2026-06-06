"""V-JEPA2 embedding / classification pipeline + CLI. P5.

    vjepa2-mlx -i clip.mp4 --task embed   -o emb.npy
    vjepa2-mlx -i clip.mp4 --task classify

Preprocess (matches the HF video processor): sample N frames, resize to 256², /255,
ImageNet-normalize -> [1, T, C, H, W]. Encoder -> token embeddings (mean-pooled by
default). Classifier -> SSv2 logits/label.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 256


def _load_frames(path: str, num_frames: int) -> list[np.ndarray]:
    from PIL import Image
    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return [np.asarray(Image.open(path).convert("RGB"))]
    import av
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    c.close()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return [frames[i] for i in idx]


def preprocess(frames: list[np.ndarray]) -> mx.array:
    """frames: list of HxWx3 uint8 -> [1, T, C, H, W] normalized."""
    from PIL import Image
    proc = []
    for fr in frames:
        im = Image.fromarray(fr).resize((SIZE, SIZE), Image.BICUBIC)
        a = (np.asarray(im).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        proc.append(a.transpose(2, 0, 1))           # CHW
    if len(proc) % 2 == 1:                            # keep T even (tubelet 2)
        proc.append(proc[-1])
    video = np.stack(proc, axis=0)[None]              # [1, T, C, H, W]
    return mx.array(video)


def embed_video(path: str, num_frames: int = 16, weights_dir=None, pool: bool = True):
    from .utils.weights import build_encoder
    model = build_encoder(weights_dir)
    x = preprocess(_load_frames(path, num_frames))
    h = model(x)                                       # [1, N, 1024]
    mx.eval(h)
    h = np.array(h)[0]
    return h.mean(axis=0) if pool else h


def classify_video(path: str, num_frames: int = 16, weights_dir=None, topk: int = 5):
    from .utils.weights import build_classifier
    model = build_classifier(weights_dir)
    x = preprocess(_load_frames(path, num_frames))
    logits = np.array(model(x))[0]
    order = logits.argsort()[::-1][:topk]
    return [(int(i), float(logits[i])) for i in order]


def cli_main() -> None:
    p = argparse.ArgumentParser(description="V-JEPA2 MLX — video embeddings / classification")
    p.add_argument("-i", "--input", required=True, help="video or image")
    p.add_argument("--task", choices=["embed", "classify"], default="embed")
    p.add_argument("-o", "--output", default=None, help="embed: .npy path")
    p.add_argument("-f", "--num_frames", type=int, default=16)
    p.add_argument("--weights_dir", default=None)
    args = p.parse_args()

    if args.task == "embed":
        emb = embed_video(args.input, args.num_frames, args.weights_dir)
        out = args.output or "embedding.npy"
        np.save(out, emb)
        print(f"{args.input} -> {out}  embedding dim {emb.shape}")
    else:
        top = classify_video(args.input, args.num_frames, args.weights_dir)
        print(f"{args.input} top classes (SSv2 id, logit):")
        for i, s in top:
            print(f"  {i:4d}  {s:.3f}")


if __name__ == "__main__":
    cli_main()
