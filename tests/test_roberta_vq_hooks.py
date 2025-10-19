import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch required for RoBERTa VQ smoke test"
)
import torch.nn as nn

from lossy_horizon.codecs.vqre import VQConfig
from lossy_horizon.models.roberta_vq import apply_vq_to_roberta


class DummyConfig:
    def __init__(self, hidden_size: int, num_attention_heads: int):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class TinyRobertaSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        B, T, Hdim = hidden_states.shape

        def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (
                self.num_attention_heads,
                self.attention_head_size,
            )
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        query_layer = transpose_for_scores(self.query(hidden_states))
        key_layer = transpose_for_scores(self.key(hidden_states))
        value_layer = transpose_for_scores(self.value(hidden_states))
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / (self.attention_head_size**0.5)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(B, T, self.all_head_size)
        return (context_layer,)


class TinyRobertaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self = TinyRobertaSelfAttention(hidden_size, num_heads)


class TinyRobertaLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = TinyRobertaAttention(hidden_size, num_heads)


class TinyRobertaEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [TinyRobertaLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )


class TinyRoberta(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.encoder = TinyRobertaEncoder(hidden_size, num_heads, num_layers)


class TinyRobertaModel(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_heads: int, num_layers: int
    ):
        super().__init__()
        self.config = DummyConfig(
            hidden_size=hidden_size, num_attention_heads=num_heads
        )
        self.roberta = TinyRoberta(hidden_size, num_heads, num_layers)
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        x = self.emb(input_ids)
        mask_exp = None
        if attention_mask is not None:
            mask_exp = (1.0 - attention_mask.float()) * -10000.0
            mask_exp = mask_exp[:, None, None, :]
        for lyr in self.roberta.encoder.layer:
            x = lyr.attention.self(x, attention_mask=mask_exp)[0]

        class Out:
            def __init__(self, h):
                self.last_hidden_state = h

        return Out(x)


def test_apply_vq_to_roberta_smoke():
    torch.manual_seed(0)
    vocab = 97
    hidden = 16
    heads = 4
    layers = 2
    seq = 8
    batch = 2

    base = TinyRobertaModel(vocab, hidden, heads, layers)
    vq_cfg = VQConfig(codebook_k_k=32, codebook_k_v=32)
    apply_vq_to_roberta(base, vq_cfg, step_provider=lambda: 100)

    input_ids = torch.randint(0, vocab, (batch, seq))
    attn_mask = torch.ones(batch, seq, dtype=torch.long)

    for lyr in base.roberta.encoder.layer:
        assert hasattr(lyr.attention.self, "_vq_state")

    out = base(input_ids=input_ids, attention_mask=attn_mask)
    assert hasattr(out, "last_hidden_state")
    assert out.last_hidden_state.shape == (batch, seq, hidden)

    for lyr in base.roberta.encoder.layer:
        attn = lyr.attention.self
        assert hasattr(attn, "_vq_state")
        assert isinstance(attn._vq_state, dict)
        assert any(
            k.startswith("key_h0_") or k.startswith("value_h0_")
            for k in attn._vq_state["losses"].keys()
        )
        for v in attn._vq_state["losses"].values():
            assert torch.is_tensor(v) and v.dim() == 0
