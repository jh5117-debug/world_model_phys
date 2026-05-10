import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from physical_consistency.trainers.stage1_components import (
    CSGODataset,
    LoRALinear,
    LingBotStage1Helper,
    _patch_flash_attention_sdpa_fallback,
    _resolve_lingbot_import_root,
    apply_gradient_checkpointing,
    apply_lora_to_wan_model,
    apply_memory_efficient_wan_block_patch,
    configure_stage1_precision_env,
    export_pretrained_state_dict,
    extract_lora_state_dict,
    load_lora_state_dict,
    resolve_stage1_low_precision_dtype,
)


class _FloatNorm(nn.Module):
    def forward(self, x):
        return x.float()


class WanRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_seq_lens = []

    def forward(self, x):
        self.seen_seq_lens.append(x.shape[1])
        return x.float().type_as(x)


class _RecordingSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_dtype = None

    def forward(self, x, seq_lens, grid_sizes, freqs):
        self.last_dtype = x.dtype
        return x


class _RecordingCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_dtype = None

    def forward(self, x, context, context_lens):
        self.last_dtype = x.dtype
        return x


class _TypedFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False, dtype=torch.bfloat16)
        self.last_dtype = None
        self.seen_seq_lens = []
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim, dtype=torch.bfloat16))

    def forward(self, x):
        self.last_dtype = x.dtype
        self.seen_seq_lens.append(x.shape[1])
        return self.proj(x)


class _NoForwardSequential(nn.Sequential):
    def forward(self, x):
        raise AssertionError("patched code should run sequential children directly")


class _DummyWanBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim, dtype=torch.bfloat16))
        self.norm1 = _FloatNorm()
        self.self_attn = _RecordingSelfAttention()
        self.norm3 = _FloatNorm()
        self.cross_attn = _RecordingCrossAttention()
        self.norm2 = _FloatNorm()
        self.ffn = _TypedFFN(dim)
        self.cam_injector_layer1 = _TypedFFN(dim)
        self.cam_injector_layer2 = _TypedFFN(dim)
        self.cam_scale_layer = _TypedFFN(dim)
        self.cam_shift_layer = _TypedFFN(dim)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
        raise NotImplementedError("test should replace this forward via patching")


class _DummyWanModel(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyWanBlock(dim) for _ in range(num_blocks)])
        self.extra_norm = WanRMSNorm()


class _TinyLoRABlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )


class _TinyLoRAModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyLoRABlock(dim)])
        self.outside = nn.Linear(dim, dim, bias=False)


class _FakeVAE:
    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    def encode(self, _videos):
        return [self.latent.clone()]


def test_apply_memory_efficient_wan_block_patch_keeps_bfloat16_activations():
    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy")

    block = model.blocks[0]
    x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    e = torch.zeros(1, 3, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    out = block(
        x,
        e,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 3, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert out.dtype == torch.bfloat16
    assert block.self_attn.last_dtype == torch.bfloat16
    assert block.cross_attn.last_dtype == torch.bfloat16
    assert block.ffn.last_dtype == torch.bfloat16


def test_apply_memory_efficient_wan_block_patch_chunks_ffn_sequence():
    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)

    block = model.blocks[0]
    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert out.shape == x.shape
    assert block.ffn.seen_seq_lens == [2, 2, 1]
    out.float().sum().backward()
    assert x.grad is not None
    assert block.ffn.proj.weight.grad is not None


def test_apply_memory_efficient_wan_block_patch_avoids_outer_sequential_forward():
    model = _DummyWanModel(dim=4)
    model.blocks[0].ffn = _NoForwardSequential(_TypedFFN(dim=4))
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)

    block = model.blocks[0]
    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert out.shape == x.shape
    inner_ffn = block.ffn[0]
    assert inner_ffn.seen_seq_lens == [2, 2, 1]
    out.float().sum().backward()
    assert inner_ffn.proj.weight.grad is not None


def test_apply_memory_efficient_wan_block_patch_runs_camera_injection_full_sequence():
    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)

    block = model.blocks[0]
    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    c2ws = torch.ones(1, 5, 4, dtype=torch.bfloat16)

    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
        dit_cond_dict={"c2ws_plucker_emb": c2ws},
    )

    assert out.shape == x.shape
    assert block.cam_injector_layer1.seen_seq_lens == [5]
    assert block.cam_injector_layer2.seen_seq_lens == [5]
    assert block.cam_scale_layer.seen_seq_lens == [5]
    assert block.cam_shift_layer.seen_seq_lens == [5]
    out.float().sum().backward()
    assert x.grad is not None


def test_apply_memory_efficient_wan_block_patch_chunks_wan_sequence_norms():
    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", norm_chunk_size=2)

    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    out = model.extra_norm(x)

    assert out.dtype == torch.bfloat16
    assert model.extra_norm.seen_seq_lens == [2, 2, 1]
    out.float().sum().backward()
    assert x.grad is not None


def test_gradient_checkpointing_wraps_memory_efficient_wan_block(monkeypatch):
    checkpoint_calls = []
    early_stop_values = []

    def fake_checkpoint(fn, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return fn(*args)

    class _FakeEarlyStop:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            early_stop_values.append(self.value)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)
    monkeypatch.setattr("torch.utils.checkpoint.set_checkpoint_early_stop", _FakeEarlyStop)

    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)
    block = model.blocks[0]
    apply_gradient_checkpointing(model, "dummy", use_reentrant=False)

    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert out.shape == x.shape
    assert getattr(block, "_pc_gradient_checkpointing_patched") is True
    assert getattr(block, "_pc_inner_gradient_checkpointing") is False
    assert block.ffn.seen_seq_lens == [2, 2, 1]
    assert checkpoint_calls == [
        {"use_reentrant": False, "determinism_check": "none"},
    ]
    assert early_stop_values == [False]
    out.float().sum().backward()
    assert block.ffn.proj.weight.grad is not None


def test_gradient_checkpointing_can_use_inner_mode_for_memory_efficient_wan(monkeypatch):
    checkpoint_calls = []
    early_stop_values = []

    def fake_checkpoint(fn, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return fn(*args)

    class _FakeEarlyStop:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            early_stop_values.append(self.value)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)
    monkeypatch.setattr("torch.utils.checkpoint.set_checkpoint_early_stop", _FakeEarlyStop)

    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)
    block = model.blocks[0]
    apply_gradient_checkpointing(model, "dummy", use_reentrant=False, memory_efficient_mode="inner")

    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
    )

    assert out.shape == x.shape
    assert getattr(block, "_pc_gradient_checkpointing_patched") is True
    assert getattr(block, "_pc_inner_gradient_checkpointing") is True
    assert getattr(block, "_pc_checkpoint_mode") == "inner"
    assert block.ffn.seen_seq_lens == [2, 2, 1]
    assert checkpoint_calls == [
        {"use_reentrant": False, "determinism_check": "none"},
        {"use_reentrant": False, "determinism_check": "none"},
        {"use_reentrant": False, "determinism_check": "none"},
    ]
    assert early_stop_values == [False, False, False]
    out.float().sum().backward()
    assert block.ffn.proj.weight.grad is not None


def test_gradient_checkpointing_wraps_camera_injection_for_memory_efficient_wan(monkeypatch):
    checkpoint_calls = []
    early_stop_values = []

    def fake_checkpoint(fn, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return fn(*args)

    class _FakeEarlyStop:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            early_stop_values.append(self.value)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)
    monkeypatch.setattr("torch.utils.checkpoint.set_checkpoint_early_stop", _FakeEarlyStop)

    model = _DummyWanModel(dim=4)
    apply_memory_efficient_wan_block_patch(model, "dummy", ffn_chunk_size=2)
    block = model.blocks[0]
    apply_gradient_checkpointing(model, "dummy", use_reentrant=False)

    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    c2ws = torch.ones(1, 5, 4, dtype=torch.bfloat16)
    out = block(
        x,
        e,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        grid_sizes=torch.tensor([[1, 1, 1]], dtype=torch.int32),
        freqs=torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        context=context,
        context_lens=torch.tensor([2], dtype=torch.int32),
        dit_cond_dict={"c2ws_plucker_emb": c2ws},
    )

    assert out.shape == x.shape
    assert block.cam_injector_layer1.seen_seq_lens == [5]
    assert checkpoint_calls == [
        {"use_reentrant": False, "determinism_check": "none"},
    ]
    assert early_stop_values == [False]
    out.float().sum().backward()
    assert block.cam_injector_layer1.proj.weight.grad is not None


def test_gradient_checkpointing_can_skip_feature_hook_block(monkeypatch):
    checkpoint_calls = []

    def fake_checkpoint(fn, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return fn(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    model = _DummyWanModel(dim=4, num_blocks=2)
    apply_memory_efficient_wan_block_patch(model, "dummy")
    apply_gradient_checkpointing(model, "dummy", use_reentrant=False, skip_block_indices={1})

    assert getattr(model.blocks[0], "_pc_gradient_checkpointing_patched") is True
    assert not getattr(model.blocks[0], "_pc_gradient_checkpointing_skipped", False)
    assert not getattr(model.blocks[1], "_pc_gradient_checkpointing_patched", False)
    assert getattr(model.blocks[1], "_pc_gradient_checkpointing_skipped") is True

    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    e = torch.zeros(1, 5, 6, 4, dtype=torch.bfloat16)
    context = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    common_kwargs = {
        "seq_lens": torch.tensor([5], dtype=torch.int32),
        "grid_sizes": torch.tensor([[1, 1, 1]], dtype=torch.int32),
        "freqs": torch.zeros(1, 5, 4, dtype=torch.bfloat16),
        "context": context,
        "context_lens": torch.tensor([2], dtype=torch.int32),
    }

    _ = model.blocks[0](x, e, **common_kwargs)
    _ = model.blocks[1](x, e, **common_kwargs)

    assert checkpoint_calls == [{"use_reentrant": False, "determinism_check": "none"}]


def test_configure_stage1_precision_env_sets_mixed_safe_defaults(monkeypatch):
    for name in (
        "PC_STAGE1_FORCE_FP32",
        "PC_STAGE1_PRECISION_PROFILE",
        "PC_STAGE1_LOWP_DTYPE",
        "PC_VAE_FORCE_FP32",
        "PC_FORCE_LORA_FP32",
        "PC_LORA_DISABLE_AUTOCAST",
        "PC_FORCE_SDPA_FALLBACK",
        "PC_FORCE_SDPA_MATH",
        "PC_FORCE_ATTN_FP32",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = configure_stage1_precision_env("mixed_safe", "bf16")

    assert policy["profile"] == "mixed_safe"
    assert os.environ["PC_STAGE1_PRECISION_PROFILE"] == "mixed_safe"
    assert os.environ["PC_STAGE1_LOWP_DTYPE"] == "bf16"
    assert os.environ["PC_VAE_FORCE_FP32"] == "1"
    assert os.environ["PC_FORCE_LORA_FP32"] == "1"
    assert os.environ["PC_LORA_DISABLE_AUTOCAST"] == "1"
    assert os.environ["PC_FORCE_SDPA_FALLBACK"] == "1"
    assert "PC_FORCE_SDPA_MATH" not in os.environ
    assert "PC_FORCE_ATTN_FP32" not in os.environ
    assert resolve_stage1_low_precision_dtype() == torch.bfloat16


def test_configure_stage1_precision_env_supports_fp16(monkeypatch):
    for name in ("PC_STAGE1_FORCE_FP32", "PC_STAGE1_PRECISION_PROFILE", "PC_STAGE1_LOWP_DTYPE"):
        monkeypatch.delenv(name, raising=False)

    policy = configure_stage1_precision_env("native_lowp", "fp16")

    assert policy["profile"] == "native_lowp"
    assert os.environ["PC_STAGE1_LOWP_DTYPE"] == "fp16"
    assert resolve_stage1_low_precision_dtype() == torch.float16


def test_patch_flash_attention_sdpa_fallback_uses_low_precision_in_mixed_safe(monkeypatch):
    for name in (
        "PC_STAGE1_FORCE_FP32",
        "PC_STAGE1_PRECISION_PROFILE",
        "PC_FORCE_SDPA_FALLBACK",
        "PC_FORCE_SDPA_MATH",
        "PC_FORCE_ATTN_FP32",
    ):
        monkeypatch.delenv(name, raising=False)
    configure_stage1_precision_env("mixed_safe", "bf16")

    calls = []

    def _fake_sdpa_fallback(**kwargs):
        calls.append(kwargs["dtype"])
        return torch.zeros_like(kwargs["q"])

    wan_attention_module = SimpleNamespace(
        _sdpa_fallback=_fake_sdpa_fallback,
        flash_attention=lambda *args, **kwargs: None,
    )
    wan_model_module = SimpleNamespace(
        flash_attention=lambda *args, **kwargs: None,
    )

    _patch_flash_attention_sdpa_fallback(wan_attention_module, wan_model_module)

    q = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16)
    out = wan_attention_module.flash_attention(q=q, k=q, v=q, dtype=torch.bfloat16)

    assert torch.equal(out, torch.zeros_like(q))
    assert calls == [torch.bfloat16]
    assert wan_model_module.flash_attention is wan_attention_module.flash_attention


def test_apply_lora_to_wan_model_replaces_block_linears_and_freezes_base_params():
    model = _TinyLoRAModel(dim=4)
    apply_lora_to_wan_model(model, model_name="tiny", rank=2, alpha=2, dropout=0.0)

    assert isinstance(model.blocks[0].linear, LoRALinear)
    assert isinstance(model.blocks[0].ffn[0], LoRALinear)
    assert isinstance(model.blocks[0].ffn[2], LoRALinear)
    assert not isinstance(model.outside, LoRALinear)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(".lora_" in name for name in trainable_names)
    assert "outside.weight" not in trainable_names
    assert model._pc_lora_config["merge_mode"] == "inplace"


def test_lora_linear_chunked_forward_matches_unchunked_and_backprops():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 4, bias=False)
    plain = LoRALinear(base, rank=2, alpha=2, dropout=0.0)
    chunked = LoRALinear(torch.nn.Linear(4, 4, bias=False), rank=2, alpha=2, dropout=0.0, chunk_size=2)
    safe_chunked = LoRALinear(
        torch.nn.Linear(4, 4, bias=False),
        rank=2,
        alpha=2,
        dropout=0.0,
        chunk_size=2,
        merge_mode="out_of_place",
    )
    chunked.load_state_dict(plain.state_dict())
    safe_chunked.load_state_dict(plain.state_dict())

    x_plain = torch.randn(1, 5, 4, requires_grad=True)
    x_chunked = x_plain.detach().clone().requires_grad_(True)
    x_safe = x_plain.detach().clone().requires_grad_(True)

    out_plain = plain(x_plain)
    out_chunked = chunked(x_chunked)
    out_safe = safe_chunked(x_safe)

    assert torch.allclose(out_plain, out_chunked)
    assert torch.allclose(out_plain, out_safe)
    out_chunked.sum().backward()
    out_safe.sum().backward()
    assert x_chunked.grad is not None
    assert chunked.lora_A.weight.grad is not None
    assert chunked.lora_B.weight.grad is not None
    assert x_safe.grad is not None
    assert safe_chunked.lora_A.weight.grad is not None
    assert safe_chunked.lora_B.weight.grad is not None


def test_lora_numeric_audit_rejects_nonfinite_forward(monkeypatch):
    monkeypatch.setenv("PC_LORA_NUMERIC_AUDIT", "1")
    monkeypatch.setenv("PC_NUMERIC_AUDIT_LIMIT", "8")

    wrapped = LoRALinear(torch.nn.Linear(4, 4, bias=False), rank=2, alpha=2, dropout=0.0)
    x = torch.zeros(1, 2, 4)
    x[0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="lora_input"):
        wrapped(x)


def test_csgo_dataset_keep_aspect_resize_letterboxes():
    ds = object.__new__(CSGODataset)
    ds.height = 8
    ds.width = 8
    ds.keep_aspect = True

    frame = np.ones((4, 8, 3), dtype=np.uint8) * 127
    resized = CSGODataset._resize_frame(ds, frame)

    assert resized.shape == (8, 8, 3)
    assert np.all(resized[:2] == 0)
    assert np.all(resized[2:6] == 127)
    assert np.all(resized[6:] == 0)


def test_apply_lora_to_wan_model_accepts_out_of_place_merge_mode():
    model = _TinyLoRAModel(dim=4)
    apply_lora_to_wan_model(
        model,
        model_name="tiny",
        rank=2,
        alpha=2,
        dropout=0.0,
        merge_mode="out-of-place",
    )

    assert isinstance(model.blocks[0].linear, LoRALinear)
    assert model.blocks[0].linear.merge_mode == "out_of_place"
    assert model._pc_lora_config["merge_mode"] == "out_of_place"


def test_lora_state_dict_round_trip_and_merge_export():
    model = _TinyLoRAModel(dim=2)
    apply_lora_to_wan_model(model, model_name="tiny", rank=1, alpha=1, dropout=0.0)
    wrapped = model.blocks[0].linear

    with torch.no_grad():
        wrapped.base.weight.copy_(torch.eye(2))
        wrapped.lora_A.weight.fill_(2.0)
        wrapped.lora_B.weight.fill_(3.0)

    lora_state = extract_lora_state_dict(model)
    cloned = _TinyLoRAModel(dim=2)
    apply_lora_to_wan_model(cloned, model_name="tiny", rank=1, alpha=1, dropout=0.0)
    load_lora_state_dict(cloned, lora_state, model_name="tiny")

    for key, value in lora_state.items():
        assert torch.equal(cloned.state_dict()[key], value)

    prefixed_state = {f"branch.{key}": value.clone() for key, value in model.state_dict().items()}
    exported = export_pretrained_state_dict(model, prefixed_state, prefix="branch.")
    assert "blocks.0.linear.weight" in exported
    assert not any(".lora_" in key or ".base." in key for key in exported)

    expected = torch.eye(2) + torch.full((2, 2), 6.0)
    assert torch.allclose(exported["blocks.0.linear.weight"], expected)


def test_prepare_y_aligns_mask_with_latent_time_axis_for_non_1_plus_4k_frames():
    helper = LingBotStage1Helper.__new__(LingBotStage1Helper)
    helper.device = torch.device("cpu")
    helper.vae = _FakeVAE(torch.randn(16, 18, 2, 3, dtype=torch.bfloat16))

    video = torch.randn(3, 70, 8, 12)
    latent = torch.randn(16, 18, 2, 3, dtype=torch.bfloat16)

    y = helper.prepare_y(video, latent)

    assert y.shape == (20, 18, 2, 3)
    assert torch.all(y[:4, 0] == 1)
    assert torch.count_nonzero(y[:4, 1:]) == 0


def _make_fake_lingbot_checkout(root: Path) -> None:
    (root / "wan" / "modules").mkdir(parents=True, exist_ok=True)
    (root / "wan" / "__init__.py").write_text("", encoding="utf-8")
    (root / "wan" / "modules" / "__init__.py").write_text("", encoding="utf-8")
    (root / "wan" / "modules" / "model.py").write_text("class WanModel: ...\n", encoding="utf-8")


def test_resolve_lingbot_import_root_accepts_explicit_repo_root(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGBOT_CODE_DIR", raising=False)
    repo_root = tmp_path / "workspace" / "code" / "lingbot-world"
    _make_fake_lingbot_checkout(repo_root)

    resolved, attempted = _resolve_lingbot_import_root(str(repo_root), project_root=tmp_path / "project")

    assert resolved == repo_root.resolve()
    assert str(repo_root) in attempted


def test_resolve_lingbot_import_root_recovers_from_nvme_path_drift(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGBOT_CODE_DIR", raising=False)
    project_root = tmp_path / "home" / "nvme04" / "workspace" / "world_model_phys" / "PHYS" / "world_model_phys"
    configured = project_root.parents[1] / "code" / "lingbot-world"
    actual = tmp_path / "home" / "nvme03" / "workspace" / "world_model_phys" / "code" / "lingbot-world"
    _make_fake_lingbot_checkout(actual)

    resolved, attempted = _resolve_lingbot_import_root(str(configured), project_root=project_root)

    assert resolved == actual.resolve()
    assert any("nvme03" in candidate for candidate in attempted)


def test_resolve_lingbot_import_root_reports_checked_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGBOT_CODE_DIR", raising=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        _resolve_lingbot_import_root(str(tmp_path / "missing" / "lingbot-world"), project_root=tmp_path / "project")

    message = str(excinfo.value)
    assert "wan/modules/model.py" in message
    assert "Checked candidate import roots" in message
