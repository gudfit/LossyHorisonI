from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..bits.bit_cost import positional_bits_cost_from_indices, token_entropy_bits


@dataclass
class PMPayload:
    keep_positions: List[int]
    kept_token_ids: List[int]
    pos_method: str
    bits_pos_enum: float
    bits_pos_rle: float
    bits_pos: float
    bits_tok: float
    total_bits: float


class PMEncoder:
    def __init__(
        self, aux_token_coder=None, vocab_size: int = 30522, pos_coding: str = "min"
    ):
        self.aux_token_coder = aux_token_coder
        self.vocab_size = vocab_size
        assert pos_coding in ("min", "enumerative", "rle")
        self.pos_coding = pos_coding

    def encode(self, token_ids: List[int], keep_indices: List[int]) -> PMPayload:
        n = len(token_ids)
        kept_ids = [token_ids[i] for i in keep_indices]
        bits_pos_min, pos_method, bits_enum, bits_rle = (
            positional_bits_cost_from_indices(n, keep_indices)
        )
        if self.pos_coding == "enumerative":
            bits_pos, pos_method = bits_enum, "enumerative"
        elif self.pos_coding == "rle":
            bits_pos, pos_method = bits_rle, "rle"
        else:
            bits_pos = bits_pos_min

        if self.aux_token_coder is None:
            import math

            bits_tok = len(kept_ids) * math.log2(max(2, self.vocab_size))
        else:

            if hasattr(self.aux_token_coder, "neglog2_stream"):
                neglog2 = self.aux_token_coder.neglog2_stream(kept_ids)
            elif hasattr(self.aux_token_coder, "neglog2"):
                neglog2 = self.aux_token_coder.neglog2(kept_ids)
            else:
                neglog2 = self.aux_token_coder(kept_ids)
            bits_tok = token_entropy_bits(neglog2)

        total_bits = bits_pos + bits_tok
        return PMPayload(
            keep_positions=keep_indices,
            kept_token_ids=kept_ids,
            pos_method=pos_method,
            bits_pos_enum=bits_enum,
            bits_pos_rle=bits_rle,
            bits_pos=bits_pos,
            bits_tok=bits_tok,
            total_bits=total_bits,
        )


class PMDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(
        self,
        payload: PMPayload,
        mask_indices: List[int],
        predicted_tokens: Dict[int, int],
    ) -> List[int]:
        max_idx = -1
        if payload.keep_positions:
            max_idx = max(max_idx, max(payload.keep_positions))
        if mask_indices:
            max_idx = max(max_idx, max(mask_indices))
        n = max_idx + 1
        full = [0] * n
        for pos, tid in zip(payload.keep_positions, payload.kept_token_ids):
            full[pos] = tid
        for i in mask_indices:
            full[i] = predicted_tokens[i]
        covered = set(payload.keep_positions) | set(mask_indices)
        assert len(covered) == n, "keep+mask must cover 0..n-1 exactly"
        return full
