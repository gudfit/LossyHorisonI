import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch required for DistilBERT VQ smoke test"
)
import torch.nn as nn

from lossy_horizon.codecs.vqre import VQConfig
from lossy_horizon.models.distilbert_vq import apply_vq_to_distilbert


class DummyConfig:
    def __init__(self, dim: int, n_heads: int):
        self.hidden_size = dim
        self.num_attention_heads = n_heads


class TinyMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)
        self.dropout_attn = nn.Dropout(0.0)

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        bs, q_len, dim = query.size()
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)
        d_h = dim // self.n_heads
        q = q.view(bs, q_len, self.n_heads, d_h).permute(0, 2, 1, 3)
        k = k.view(bs, q_len, self.n_heads, d_h).permute(0, 2, 1, 3)
        v = v.view(bs, q_len, self.n_heads, d_h).permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-1, -2)) / (d_h**0.5)
        if mask is not None:
            mask_ = (mask == 0).view(bs, 1, 1, q_len)
            scores = scores.masked_fill(mask_, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.permute(0, 2, 1, 3).contiguous().view(bs, q_len, dim)
        context = self.out_lin(context)
        return (context,)


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention = TinyMultiHeadSelfAttention(dim, n_heads)


class TinyTransformer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [TinyTransformerBlock(dim, n_heads) for _ in range(n_layers)]
        )


class TinyDistilBert(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.transformer = TinyTransformer(dim, n_heads, n_layers)


class TinyDistilModel(nn.Module):
    def __init__(self, vocab: int, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.config = DummyConfig(dim, n_heads)
        self.distilbert = TinyDistilBert(dim, n_heads, n_layers)
        self.emb = nn.Embedding(vocab, dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        x = self.emb(input_ids)
        mask = attention_mask
        for blk in self.distilbert.transformer.layer:
            x = blk.attention(x, x, x, mask)[0]

        class Out:
            def __init__(self, h):
                self.last_hidden_state = h

        return Out(x)


def test_apply_vq_to_distilbert_smoke():
    torch.manual_seed(0)
    vocab = 101
    dim = 16
    heads = 4
    layers = 2
    seq = 8
    batch = 2

    base = TinyDistilModel(vocab, dim, heads, layers)
    vq_cfg = VQConfig(codebook_k_k=32, codebook_k_v=32)
    apply_vq_to_distilbert(base, vq_cfg, step_provider=lambda: 100)

    input_ids = torch.randint(0, vocab, (batch, seq))
    attn_mask = torch.ones(batch, seq, dtype=torch.long)

    for blk in base.distilbert.transformer.layer:
        assert hasattr(blk.attention, "_vq_state")

    out = base(input_ids=input_ids, attention_mask=attn_mask)
    assert hasattr(out, "last_hidden_state")
    assert out.last_hidden_state.shape == (batch, seq, dim)

    for blk in base.distilbert.transformer.layer:
        attn = blk.attention
        assert isinstance(attn._vq_state, dict)
        assert any(
            k.startswith("key_h0_") or k.startswith("value_h0_")
            for k in attn._vq_state["losses"].keys()
        )
        for v in attn._vq_state["losses"].values():
            assert torch.is_tensor(v) and v.dim() == 0
