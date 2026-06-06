"""Model build + HF auto-download load path. P5.
Resolve: arg | $VJEPA2_MLX_WEIGHTS_DIR | dist/<name> | HF mlx-community/<name>."""
WEIGHTS_DIR_ENV = "VJEPA2_MLX_WEIGHTS_DIR"
HF_ORG = "mlx-community"
def build_model(checkpoint: str = "vitl-encoder", weights_dir=None):
    raise NotImplementedError("P5")
