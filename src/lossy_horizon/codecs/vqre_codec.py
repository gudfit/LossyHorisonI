from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..bits.bit_cost import positional_bits_cost_from_indices
from ..bits.rank_model import RankPMF
from ..bits.rans import (
    rans_bits_bernoulli,
    rans_bits_from_pmf,
    rans_bits_adaptive_unigram,
    ac_bits_adaptive_unigram_ideal,
)
from ..bits.ar_coder import NGramARCoder


@dataclass
class VQREPayload:
    anchor_positions: List[int]
    anchor_token_ids: List[int]
    pos_method: str
    bits_pos_enum: float
    bits_pos_rle: float
    bits_anchors_tok: float
    flags: List[int]
    corr_syms: List[int]
    fallback_token_ids: List[int]
    bits_flag: float
    bits_sym: float
    bits_fallback_tok: float
    bits_flag_ac: float
    bits_sym_ac: float
    bits_fallback_tok_ac: float
    bits_fallback_tok_ideal: float
    total_bits: float


class VQREEncoder:
    def __init__(
        self,
        rank_threshold: int = 16,
        anchors_every: int = 0,
        rank_pmf: Optional[RankPMF] = None,
        vocab_size: int = 30522,
        ar_coder_anchors: Optional[NGramARCoder] = None,
        ar_coder_corr: Optional[NGramARCoder] = None,
    ):
        self.K = int(rank_threshold)
        assert self.K >= 2
        self.anchors_every = int(anchors_every)
        self.rank_pmf = rank_pmf
        self.vocab_size = int(vocab_size)
        self.ar_coder_anchors = ar_coder_anchors
        self.ar_coder_corr = ar_coder_corr

    def encode(
        self,
        token_ids: List[int],
        mask_indices: List[int],
        ranks_by_index: Dict[int, int],
        topk_by_index: Dict[int, List[int]],
    ) -> VQREPayload:
        n = len(token_ids)
        anchors = (
            list(range(0, n, self.anchors_every)) if self.anchors_every > 0 else []
        )
        anchor_set = set(anchors)
        bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(
            n, anchors
        )

        anchor_token_ids = [token_ids[i] for i in anchors]
        if self.ar_coder_anchors is None:
            import math

            bits_anchors_tok = len(anchor_token_ids) * math.log2(
                max(2, self.vocab_size)
            )
        else:
            bits_anchors_tok = sum(
                self.ar_coder_anchors.neglog2_stream(anchor_token_ids)
            )

        flags: List[int] = []
        corr_syms: List[int] = []
        fallback_token_ids: List[int] = []
        flag_symbols: List[int] = []
        sym_symbols: List[int] = []

        for i in mask_indices:
            if i in anchor_set:
                continue
            r = int(ranks_by_index.get(i, 1))
            if r <= 1:
                flags.append(0)
                flag_symbols.append(0)
            else:
                flags.append(1)
                flag_symbols.append(1)
                if 2 <= r <= self.K:
                    sym = r - 2  # 0..K-2
                    corr_syms.append(sym)
                    sym_symbols.append(sym)
                else:
                    sym = self.K - 1
                    corr_syms.append(sym)
                    sym_symbols.append(sym)
                    fallback_token_ids.append(token_ids[i])

        if flags:
            p1 = sum(flags) / len(flags)
            import math

            def H2(p: float) -> float:
                if p <= 0.0 or p >= 1.0:
                    return 0.0
                q = 1.0 - p
                return -p * math.log2(p) - q * math.log2(q)

            bits_flag = len(flags) * H2(p1)
            bits_flag_ac = rans_bits_bernoulli(flag_symbols, p1)
        else:
            bits_flag = 0.0
            bits_flag_ac = 0.0

        if sym_symbols:
            if self.rank_pmf is not None:
                pmf_ranks = [self.rank_pmf.prob(r) for r in range(2, self.K + 1)]
                eps = 1.0 / max(100.0, float(self.K))
                p_fallback = max(eps, sum(pmf_ranks) * eps)
                Z = sum(pmf_ranks) + p_fallback
                pmf_syms = [p / Z for p in pmf_ranks] + [p_fallback / Z]
            else:
                counts = [0] * self.K
                for s in sym_symbols:
                    counts[int(s)] += 1
                total = sum(counts)
                pmf_syms = [c / total if total > 0 else 1.0 / self.K for c in counts]
            bits_sym_ac = rans_bits_from_pmf(sym_symbols, pmf_syms)
            import math

            bits_sym = sum(-math.log2(max(pmf_syms[s], 1e-12)) for s in sym_symbols)
        else:
            bits_sym_ac = 0.0
            bits_sym = 0.0

        if fallback_token_ids:
            if self.ar_coder_corr is None:
                import math

                bits_fallback_tok = len(fallback_token_ids) * math.log2(
                    max(2, self.vocab_size)
                )
            else:
                bits_fallback_tok = sum(
                    self.ar_coder_corr.neglog2_stream(fallback_token_ids)
                )
            bits_fallback_tok_ac = rans_bits_adaptive_unigram(
                fallback_token_ids, self.vocab_size
            )
            bits_fallback_tok_ideal = ac_bits_adaptive_unigram_ideal(
                fallback_token_ids, self.vocab_size
            )
        else:
            bits_fallback_tok = 0.0
            bits_fallback_tok_ac = 0.0
            bits_fallback_tok_ideal = 0.0

        total_bits = (
            bits_pos + bits_anchors_tok + bits_flag + bits_sym + bits_fallback_tok
        )

        return VQREPayload(
            anchor_positions=anchors,
            anchor_token_ids=anchor_token_ids,
            pos_method=pos_method,
            bits_pos_enum=bits_enum,
            bits_pos_rle=bits_rle,
            bits_anchors_tok=bits_anchors_tok,
            flags=flags,
            corr_syms=corr_syms,
            fallback_token_ids=fallback_token_ids,
            bits_flag=bits_flag,
            bits_sym=bits_sym,
            bits_fallback_tok=bits_fallback_tok,
            bits_flag_ac=bits_flag_ac,
            bits_sym_ac=bits_sym_ac,
            bits_fallback_tok_ac=bits_fallback_tok_ac,
            bits_fallback_tok_ideal=bits_fallback_tok_ideal,
            total_bits=total_bits,
        )


class VQREDecoder:
    def __init__(self, rank_threshold: int = 16):
        self.K = int(rank_threshold)
        assert self.K >= 2

    def decode(
        self,
        payload: VQREPayload,
        mask_indices: List[int],
        topk_by_index: Dict[int, List[int]],
        predicted_top1: Dict[int, int],
        original_token_ids: Optional[List[int]] = None,
    ) -> List[int]:
        if original_token_ids is not None:
            n = len(original_token_ids)
            full = list(original_token_ids)
        else:
            max_idx = -1
            all_positions = set(mask_indices) | set(payload.anchor_positions)
            if all_positions:
                max_idx = max(all_positions)
            n = max_idx + 1
            full = [0] * n

        for pos, tid in zip(payload.anchor_positions, payload.anchor_token_ids):
            if 0 <= pos < n:
                full[pos] = tid

        f_iter = iter(payload.flags)
        s_iter = iter(payload.corr_syms)
        t_iter = iter(payload.fallback_token_ids)
        for i in mask_indices:
            if i in payload.anchor_positions:
                continue
            flag = next(f_iter, 0)
            if flag == 0:
                full[i] = predicted_top1[i]
                continue
            sym = next(s_iter)
            if sym <= self.K - 2:
                r = int(sym) + 2
                cand = topk_by_index.get(i, [])
                full[i] = cand[r - 1] if (r - 1) < len(cand) else predicted_top1[i]
            else:
                full[i] = next(t_iter, predicted_top1[i])
        return full
