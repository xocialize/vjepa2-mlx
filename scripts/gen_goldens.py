"""P1: HF transformers oracle. Install transformers, load VJEPA2Model (+ classifier,
+ AC), run seeded video clip on CPU fp32, capture per-component goldens
(patch-embed, 3D-rope q/k, per-layer, full encoder hidden states, predictor,
pooler) -> goldens/<name>.npz. Also pin 3D-RoPE params from modeling_vjepa2.py."""
raise NotImplementedError("P1")
