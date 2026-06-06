"""HF state_dict -> MLX weights. Operates on numpy (no torch import at use).

Only the Conv3d patch-embed weight changes layout:
  Conv3d (O,I,kT,kH,kW) -> (O,kT,kH,kW,I)   transpose(0,2,3,4,1)
Linear weights are (out,in) in both torch and MLX -> identity. LayerNorm / bias
pass through. Keys already match the MLX module tree 1:1 (verified).

`prefix` filters a component (e.g. "encoder.") so each can be loaded alone.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def convert_state_dict(sd: dict[str, np.ndarray], prefix: str | None = None) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for k, v in sd.items():
        if prefix is not None and not k.startswith(prefix):
            continue
        v = np.asarray(v)
        if v.ndim == 5:  # Conv3d patch embed
            v = np.transpose(v, (0, 2, 3, 4, 1))
        out[k] = mx.array(v)
    return out


def convert_fair_encoder(sd: dict[str, np.ndarray]) -> dict[str, mx.array]:
    """facebookresearch/vjepa2 ViT encoder state_dict -> my VJEPA2Model (HF-style).

    Remaps keys and SPLITS the fused qkv into separate query/key/value:
      patch_embed.proj          -> encoder.embeddings.patch_embeddings.proj  (Conv3d transpose)
      blocks.{i}.attn.qkv       -> encoder.layer.{i}.attention.{query,key,value}  (split 3*D)
      blocks.{i}.attn.proj      -> encoder.layer.{i}.attention.proj
      blocks.{i}.norm1/norm2/mlp-> encoder.layer.{i}.norm1/norm2/mlp
      norm                      -> encoder.layernorm
    """
    out: dict[str, mx.array] = {}
    for k, v in sd.items():
        v = np.asarray(v)
        if k.startswith("patch_embed.proj"):
            nk = k.replace("patch_embed.proj", "encoder.embeddings.patch_embeddings.proj")
            if v.ndim == 5:
                v = np.transpose(v, (0, 2, 3, 4, 1))
            out[nk] = mx.array(v)
        elif k.startswith("blocks."):
            i = k.split(".")[1]
            rest = k.split(".", 2)[2]
            base = f"encoder.layer.{i}."
            if rest.startswith("attn.qkv."):
                kind = rest.split(".")[-1]  # weight | bias
                D = v.shape[0] // 3
                q, kk, vv = v[:D], v[D:2 * D], v[2 * D:]
                out[base + f"attention.query.{kind}"] = mx.array(q)
                out[base + f"attention.key.{kind}"] = mx.array(kk)
                out[base + f"attention.value.{kind}"] = mx.array(vv)
            elif rest.startswith("attn.proj."):
                out[base + "attention." + rest[len("attn."):]] = mx.array(v)
            else:  # norm1/norm2/mlp.*
                out[base + rest] = mx.array(v)
        elif k.startswith("norm."):
            out["encoder.layernorm." + k[len("norm."):]] = mx.array(v)
    return out
