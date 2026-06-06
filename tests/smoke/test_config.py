"""Smoke tests — config truth + scaffold imports (no torch, no weights)."""

from vjepa2_mlx.config import CHECKPOINTS, DEFAULT, VJEPA2Config


def test_vitl_config_pinned():
    c = VJEPA2Config()
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads) == (1024, 24, 16)
    assert c.intermediate_size == 4096
    assert c.head_dim == 64
    assert (c.patch_size, c.tubelet_size, c.crop_size, c.frames_per_clip) == (16, 2, 256, 64)
    assert c.layer_norm_eps == 1e-6 and c.qkv_bias is True


def test_grid_dims():
    c = VJEPA2Config()
    assert c.grid_depth == 32   # 64 / 2
    assert c.grid_hw == 16      # 256 / 16
    # total tokens = 32 * 16 * 16
    assert c.grid_depth * c.grid_hw * c.grid_hw == 8192


def test_predictor_config():
    c = VJEPA2Config()
    assert (c.pred_hidden_size, c.pred_num_hidden_layers, c.pred_num_attention_heads) == (384, 12, 12)


def test_checkpoint_registry():
    assert DEFAULT in CHECKPOINTS
    assert CHECKPOINTS["vitl-encoder"] == "facebook/vjepa2-vitl-fpc64-256"


def test_ops_and_models_import():
    # modules import even though forward/init raise NotImplementedError (skeletons)
    import vjepa2_mlx.ops.rope_3d  # noqa: F401
    import vjepa2_mlx.ops.patch_embed_3d  # noqa: F401
    import vjepa2_mlx.models.modeling_vjepa2  # noqa: F401
