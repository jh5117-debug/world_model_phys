import torch
import torch.nn as nn

from physical_consistency.trainers.stage1_components import apply_memory_efficient_wan_block_patch


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
    x = torch.randn(1, 5, 4, dtype=torch.bfloat16)
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
