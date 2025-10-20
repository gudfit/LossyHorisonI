import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for refine smoke test")
import torch.nn as nn

from lossy_horizon.codecs.vqre import refine_decode


class DummyConfig:
    def __init__(self, max_position_embeddings: int):
        self.max_position_embeddings = max_position_embeddings


class DummyLM(nn.Module):
    def __init__(self, vocab: int, hidden: int, max_pos: int):
        super().__init__()
        self.config = DummyConfig(max_pos)
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids=None, attention_mask=None):
        x = self.emb(input_ids)
        logits = self.proj(x)  # [B, T, V]

        class Out:
            def __init__(self, logits):
                self.logits = logits

        return Out(logits)


class DummyTok:
    # Provide optional special token ids; left as None so all positions are eligible
    cls_token_id = None
    sep_token_id = None
    pad_token_id = None
    mask_token_id = None


def _run_refine(device: torch.device, use_half: bool):
    vocab = 101
    hidden = 16
    max_pos = 64
    B = 1
    T = 200  # force sliding-window path (> max_pos)
    model = DummyLM(vocab, hidden, max_pos).to(device)
    if use_half:
        model = model.half()
    ids = torch.randint(0, vocab, (B, T), device=device, dtype=torch.long)
    out = refine_decode(DummyTok(), model, ids, steps=2, topm=8)
    assert out.shape == ids.shape
    # Ensure outputs are token ids in range
    assert out.dtype == torch.long
    assert int(out.max()) < vocab and int(out.min()) >= 0


def test_refine_decode_sliding_cpu():
    _run_refine(torch.device("cpu"), use_half=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_refine_decode_sliding_fp16_cuda():
    _run_refine(torch.device("cuda"), use_half=True)

