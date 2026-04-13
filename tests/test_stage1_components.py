import torch
import torch.nn as nn

from physical_consistency.trainers.stage1_components import (
    LoRALinear,
    LingBotStage1Helper,
    apply_lora_to_wan_model,
    apply_memory_efficient_wan_block_patch,
    export_pretrained_state_dict,
    extract_lora_state_dict,
    load_lora_state_dict,
)


class _FloatNorm(nn.Module):
    def forward(self, x):
        return x.float()


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

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
        raise NotImplementedError("test should replace this forward via patching")


class _DummyWanModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyWanBlock(dim)])


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
