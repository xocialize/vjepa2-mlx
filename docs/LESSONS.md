# V-JEPA2 → MLX — skill-worthy lessons

Candidates to fold into the `mlx-porting` skill. Captured at the P3 pause
(encoder parity-locked). Each is verified parity-exact vs PyTorch on `mx.cpu`.

## 1. 3D-RoPE for video ViTs (NEW — strong candidate)

V-JEPA2 applies rotary embeddings over a **3D token grid** (depth, height, width),
splitting `head_dim` across the three axes. The reusable pattern:

- `head_dim` (64) → `d = h = w = 2*((head_dim // 3) // 2)` = 20 each, **remainder
  (4) passes through unrotated**. Don't assume the split is exact.
- Position ids decompose the *flattened* token index over the grid using the
  **config** `grid_size` (16), **not** the actual input spatial size:
  `frame = id // (g*g)`, `height = (id - g*g*frame) // g`, `width = remainder`.
- **The rotary itself has a quirk — match it, don't "fix" it.** `cos`/`sin` are
  **concat-tiled** (`emb = concat([e, e])`, so index `i` uses `freq[i mod D/2]`),
  but the rotated half is **interleaved** (`y[2j] = -x[2j+1]`, `y[2j+1] = x[2j]`).
  This is *not* the standard GPT-NeoX rope; replicating the exact op order is what
  passes parity (2.4e-6). See `ops/rope_3d.py`.

`mx.fast.rope` only covers the 1-D case; the 3-D composition (per-axis rope on
disjoint channel groups, then concat) is assembled by hand.

## 2. 3D-tubelet patch embed (Conv3d) in MLX (NEW)

Video patch embedding = `nn.Conv3d` with `kernel = stride = (tubelet, patch,
patch)`. MLX Conv3d is **NDHWC**, so:

- Feed `(B, T, H, W, C)` (permute from the upstream `(B, T, C, H, W)`).
- Weight transpose: PyTorch Conv3d `(O, I, kT, kH, kW)` → MLX `(O, kT, kH, kW, I)`
  = `transpose(0, 2, 3, 4, 1)`. (Extends the existing Conv2d `(0,2,3,1)` rule.)
- Token flatten order matches torch automatically: reshape `(B, D', H', W', C)`
  → `(B, D'·H'·W', C)` gives token index `d·H'·W' + h·W' + w`, identical to
  torch's `flatten(2).transpose(1,2)`.

## 3. Relative-error gating for large activations (REINFORCES the Zonos lesson)

V-JEPA2's encoder has **no final scaling** — `last_hidden` activations reach ~43.
Absolute `max_abs` over a 24-layer stack misfires (fp32 accumulation gives
abs ~1e-3 that *looks* large). Gate the full-pass on **relative** error
(`max|Δ| / max|ref|` = 2.66e-5). Already noted for Zonos; video ViTs are another
clear case — worth generalizing in the skill: *check the activation magnitude
before choosing an abs vs rel threshold.*

## 4. CONFIRM #2 reframe: "Python exists, but no Swift / no published weights"

Process lesson (now seen 3×: SeedVR2, V-JEPA2, and the inverse for Real-ESRGAN).
When CONFIRM gate #2 finds an existing MLX-**Python** port that is experimental
(no parity claim, **no published mlx-community weights, no Swift**), net-new Python
has low marginal value. The high-value reframe is **Python-first → Swift**: build a
parity-verified Python port + publish the missing weights artifact (which doubles
as the Swift port's oracle), then do the net-new MLX-Swift port that nobody has.
Don't let "a port exists" end the evaluation — check *which runtime* it serves.

---

### Also from this session (other ports, for the same skill pass)

- **RIFE** (`rife-mlx`): hand-rolled **`grid_sample`** (bilinear, `padding_mode='border'`,
  `align_corners=True`) and **bilinear `interpolate`** (align_corners variants) in MLX
  NHWC — both parity-exact; reusable for any warp/flow model. Plus: coarse-to-fine
  pyramids need **pad scaled with 1/scale** (caught by benchmarking), small flow nets
  are **fp16-sensitive** (ship fp32), and PyAV ≥10 dropped `add_stream(template=)` →
  use `add_stream_from_template` for audio passthrough.
