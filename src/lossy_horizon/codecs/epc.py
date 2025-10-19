from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..bits.bit_cost import positional_bits_cost_from_indices, binary_entropy_bits
from ..bits.rank_model import RankPMF, RankModel
from ..bits.rans import (
    rans_bits_bernoulli,
    rans_bits_from_pmf,
    rans_bits_adaptive_unigram,
)


@dataclass
class EPCCorrection:
    index: int
    flag: int
    rank: Optional[int] = None
    token_id: Optional[int] = None


@dataclass
class EPCPayload:
    keep_positions: List[int]
    kept_token_ids: List[int]
    corrections: List[EPCCorrection]
    pos_method: str
    bits_pos_enum: float
    bits_pos_rle: float
    bits_pos: float
    bits_tok_kept: float
    bits_flag: float
    bits_rank: float
    bits_corr_tok: float
    bits_flag_ac: float
    bits_rank_ac: float
    bits_corr_tok_ac: float
    total_bits: float


class EPCEncoder:
    def __init__(
        self,
        rank_threshold: int = 16,
        fallback: bool = True,
        aux_token_coder_kept=None,
        aux_token_coder_corr=None,
        vocab_size: int = 30522,
        rank_pmf: Optional[RankPMF] = None,
        rank_model: Optional[RankModel] = None,
        aux_token_coder=None,
    ):
        self.K = rank_threshold
        self.fallback = fallback
        self.aux_token_coder = (
            aux_token_coder_kept
            if aux_token_coder_kept is not None
            else aux_token_coder
        )
        self.aux_token_coder_corr = aux_token_coder_corr
        self.vocab_size = vocab_size
        self.rank_pmf = rank_pmf
        self.rank_model = rank_model

    def encode(
        self,
        token_ids: List[int],
        keep_indices: List[int],
        mask_indices: List[int],
        rank_by_index: Dict[int, int],
    ) -> EPCPayload:
        n = len(token_ids)
        bits_pos_min, pos_method, bits_enum, bits_rle = (
            positional_bits_cost_from_indices(n, keep_indices)
        )
        pos_coding = getattr(self, "pos_coding", "min")
        if pos_coding == "enumerative":
            bits_pos, pos_method = bits_enum, "enumerative"
        elif pos_coding == "rle":
            bits_pos, pos_method = bits_rle, "rle"
        else:
            bits_pos = bits_pos_min

        corrections: List[EPCCorrection] = []
        flags = []
        bits_rank = 0.0
        bits_corr_tok = 0.0
        rank_symbols: List[int] = []
        flag_symbols: List[int] = []
        corr_tokens: List[int] = []

        for i in mask_indices:
            r = int(rank_by_index.get(i, 1))
            if r <= 1:
                corrections.append(EPCCorrection(index=i, flag=0))
                flags.append(0)
                flag_symbols.append(0)
                continue
            flags.append(1)
            flag_symbols.append(1)
            if 2 <= r <= self.K:
                corrections.append(EPCCorrection(index=i, flag=1, rank=r))
                if self.rank_model is not None:
                    c_bits = self.rank_model.bits(r)
                elif self.rank_pmf is not None:
                    c_bits = self.rank_pmf.bits(r)
                else:
                    if self.K <= 2:
                        c_bits = 0.0
                    else:
                        c_bits = math_log2_ceil(self.K - 1)
                bits_rank += c_bits
                rank_symbols.append(r - 2)
            else:
                if self.fallback:
                    corrections.append(
                        EPCCorrection(index=i, flag=1, token_id=token_ids[i])
                    )
                    if self.aux_token_coder_corr is None:
                        import math

                        bits_corr_tok += math.log2(max(2, self.vocab_size))
                    else:
                        if hasattr(self.aux_token_coder_corr, "neglog2_stream"):
                            neglog2 = self.aux_token_coder_corr.neglog2_stream(
                                [token_ids[i]]
                            )
                        elif hasattr(self.aux_token_coder_corr, "neglog2"):
                            neglog2 = self.aux_token_coder_corr.neglog2([token_ids[i]])
                        else:
                            neglog2 = self.aux_token_coder_corr([token_ids[i]])
                        bits_corr_tok += float(neglog2[0])
                    corr_tokens.append(token_ids[i])
                else:
                    corrections.append(EPCCorrection(index=i, flag=1))

        p1 = (sum(flags) / max(1, len(flags))) if flags else 0.0
        bits_flag = len(flags) * binary_entropy_bits(p1)
        bits_flag_ac = rans_bits_bernoulli(flag_symbols, p1) if flag_symbols else 0.0

        kept_token_ids = [token_ids[i] for i in keep_indices]
        if self.aux_token_coder is None:
            import math

            bits_tok_kept = len(kept_token_ids) * math.log2(max(2, self.vocab_size))
        else:
            if hasattr(self.aux_token_coder, "neglog2_stream"):
                neglog2 = self.aux_token_coder.neglog2_stream(kept_token_ids)
            elif hasattr(self.aux_token_coder, "neglog2"):
                neglog2 = self.aux_token_coder.neglog2(kept_token_ids)
            else:
                neglog2 = self.aux_token_coder(kept_token_ids)
            bits_tok_kept = float(sum(neglog2))

        if rank_symbols:
            if self.rank_model is not None:
                pmf = self.rank_model.pmf_list()
            elif self.rank_pmf is not None:
                pmf = [self.rank_pmf.prob(r) for r in range(2, self.K + 1)]
            else:
                counts = [0] * (self.K - 1)
                for sym in rank_symbols:
                    if 0 <= sym < len(counts):
                        counts[sym] += 1
                total = sum(counts)
                pmf = [c / total if total > 0 else 1.0 / (self.K - 1) for c in counts]
            bits_rank_ac = rans_bits_from_pmf(rank_symbols, pmf)
        else:
            bits_rank_ac = 0.0
        bits_corr_tok_ac = (
            rans_bits_adaptive_unigram(corr_tokens, self.vocab_size)
            if corr_tokens
            else 0.0
        )

        total_bits = bits_pos + bits_tok_kept + bits_flag + bits_rank + bits_corr_tok
        return EPCPayload(
            keep_positions=keep_indices,
            kept_token_ids=kept_token_ids,
            corrections=corrections,
            pos_method=pos_method,
            bits_pos_enum=bits_enum,
            bits_pos_rle=bits_rle,
            bits_pos=bits_pos,
            bits_tok_kept=bits_tok_kept,
            bits_flag=bits_flag,
            bits_rank=bits_rank,
            bits_corr_tok=bits_corr_tok,
            bits_flag_ac=bits_flag_ac,
            bits_rank_ac=bits_rank_ac,
            bits_corr_tok_ac=bits_corr_tok_ac,
            total_bits=total_bits,
        )


class EPCDecoder:
    def __init__(self, rank_threshold: int = 16):
        self.K = rank_threshold

    def decode(
        self,
        payload: EPCPayload,
        mask_indices: List[int],
        topk_by_index: Dict[int, List[int]],
        predicted_top1: Dict[int, int],
    ) -> List[int]:
        for i in mask_indices:
            tk = topk_by_index.get(i, [])
            assert len(tk) >= self.K or (
                self.K <= 0
            ), "Top-K list shorter than rank_threshold; ensure ranks_and_topk k==rank_threshold"
        max_idx = -1
        if payload.keep_positions:
            max_idx = max(max_idx, max(payload.keep_positions))
        if mask_indices:
            max_idx = max(max_idx, max(mask_indices))
        n = max_idx + 1
        full = [0] * n
        for pos, tid in zip(payload.keep_positions, payload.kept_token_ids):
            full[pos] = tid

        corr_map = {c.index: c for c in payload.corrections}
        for i in mask_indices:
            c = corr_map.get(i)
            if c is None or c.flag == 0:
                full[i] = predicted_top1[i]
            elif c.rank is not None and 2 <= c.rank <= self.K:
                cand = topk_by_index[i]
                if c.rank - 1 < len(cand):
                    full[i] = cand[c.rank - 1]
                else:
                    full[i] = predicted_top1[i]
            elif c.token_id is not None:
                full[i] = c.token_id
            else:
                full[i] = predicted_top1[i]
        covered = set(payload.keep_positions) | set(mask_indices)
        assert len(covered) == n, "keep+mask must cover 0..n-1 exactly"
        return full


def math_log2_ceil(x: int) -> float:
    import math

    return math.ceil(math.log2(max(2, x)))
