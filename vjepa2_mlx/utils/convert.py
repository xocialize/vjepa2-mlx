"""HF safetensors -> MLX weights. P5. Conv3d (O,I,kT,kH,kW)->(O,kT,kH,kW,I);
Linear identity; LayerNorm/bias passthrough; mx.eval before save."""
import mlx.core as mx, numpy as np
def convert_state_dict(sd: dict) -> dict:
    raise NotImplementedError("P5: after key set pinned in P1")
