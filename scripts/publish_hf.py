"""P7: publish the fp16 dist/ artifacts to mlx-community.

Writes a model card per repo and uploads dist/<name>/. Requires HF auth
(xocialize @ mlx-community). Run after convert_to_mlx.py.

  python scripts/publish_hf.py [base|classifier|all]
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, create_repo, upload_folder

ORG = "mlx-community"
DIST = "dist"

CARDS = {
    "V-JEPA2-vitl-fpc64-256": """---
license: mit
library_name: mlx
pipeline_tag: video-classification
tags: [mlx, video, vjepa2, video-embeddings, world-model, apple-silicon]
---

# V-JEPA2 ViT-L (MLX) — encoder + JEPA predictor

Apple **MLX** fp16 port of Meta's [V-JEPA2](https://github.com/facebookresearch/vjepa2)
**ViT-L/16** (`facebook/vjepa2-vitl-fpc64-256`): video/image **embedding extraction**
plus the **JEPA latent-space predictor** (masked world-model). MIT.

```bash
pip install vjepa2-mlx   # https://github.com/xocialize/vjepa2-mlx
vjepa2-mlx -i clip.mp4 --task embed -o emb.npy
```

```python
from vjepa2_mlx.pipeline_mlx import embed_video
emb = embed_video("clip.mp4", num_frames=16)   # (1024,)
```

- **Arch**: ViT-L/16, 24 layers, hidden 1024, 16 heads, 3D-tubelet Conv3d patch embed,
  3D-RoPE; predictor 384/12/12.
- **Parity vs PyTorch (cpu fp32)**: encoder rel 2.66e-5 · predictor rel 1.67e-6.
- **Precision**: fp16 (~650 MB; encoder fp16 rel 4.7e-3).

MIT (© Meta Platforms). *Action-conditioned (robotics) predictor not included — separate ViT-g model.*
""",
    "V-JEPA2-vitl-fpc16-256-ssv2": """---
license: mit
library_name: mlx
pipeline_tag: video-classification
tags: [mlx, video, vjepa2, video-classification, something-something-v2, apple-silicon]
---

# V-JEPA2 ViT-L SSv2 classifier (MLX)

Apple **MLX** fp16 port of `facebook/vjepa2-vitl-fpc16-256-ssv2` — V-JEPA2 ViT-L
encoder + attentive-pooler + linear head for **Something-Something-v2** action
recognition (174 classes). MIT.

```bash
pip install vjepa2-mlx   # https://github.com/xocialize/vjepa2-mlx
vjepa2-mlx -i clip.mp4 --task classify
```

- **Parity vs PyTorch (cpu fp32)**: logits rel 2.43e-6, argmax matches (fp16 too).
- **Precision**: fp16 (~707 MB).

MIT (© Meta Platforms).
""",
    "V-JEPA2-AC-vitg": """---
license: mit
library_name: mlx
pipeline_tag: video-classification
tags: [mlx, video, vjepa2, world-model, action-conditioned, robotics, apple-silicon]
---

# V-JEPA2-AC ViT-g (MLX) — action-conditioned world-model

Apple **MLX** fp16 port of Meta's **V-JEPA2-AC** (`vjepa2-ac-vitg`): a **ViT-g**
video encoder + an **action-conditioned predictor** that, given encoder context
tokens + per-frame 7-DoF robot poses (action/state), predicts future latent
states — the world-model used for robot planning. MIT.

```python
from vjepa2_mlx.utils.weights import build_ac_encoder, build_ac_predictor
enc = build_ac_encoder()          # ViT-g encoder (hidden 1408, 40 layers)
pred = build_ac_predictor()       # AC predictor (frame-causal, 3D-RoPE)
# tokens = enc(video); future = pred(tokens, actions, states)
```

- **Arch**: ViT-g encoder (1408 / 40 / 22 heads, GELU, 3D-RoPE) + AC predictor
  (1024 / 24 / 16, fused-qkv ACRoPEAttention, frame-causal mask, 7-DoF action+state encoders).
- **Parity vs upstream torch (cpu fp32)**: encoder rel 3.5e-5 · AC predictor rel 8.8e-6
  (structural 8.9e-8). fp16: encoder rel 9.1e-3 · predictor 4.8e-4.
- **Precision**: fp16 (~2.6 GB).

MIT (© Meta Platforms). Converted from `vjepa2-ac-vitg.pt`.
""",
}


def publish(name: str) -> None:
    folder = os.path.join(DIST, name)
    if not os.path.isdir(folder):
        print(f"!! missing {folder}; run convert_to_mlx.py"); return
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(CARDS[name])
    repo_id = f"{ORG}/{name}"
    create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model",
                  commit_message=f"Add {name} (MLX fp16 V-JEPA2 port)")
    print(f"OK  https://huggingface.co/{repo_id}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    HfApi().whoami()
    table = {"base": "V-JEPA2-vitl-fpc64-256", "classifier": "V-JEPA2-vitl-fpc16-256-ssv2",
             "ac": "V-JEPA2-AC-vitg"}
    names = list(CARDS) if which == "all" else [table[which]]
    for n in names:
        publish(n)


if __name__ == "__main__":
    main()
