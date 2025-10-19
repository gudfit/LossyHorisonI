from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Callable

import torch
import torch.nn.functional as F

from .bert_vq import apply_vq_to_bert, VQConfig


@dataclass
class Tokenised:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: List[str]


class MLMScorer:
    def __init__(
        self, model_name: str = "bert-base-cased", device: Optional[str] = None
    ):
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=False
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, local_files_only=False
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, local_files_only=True
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.mask_token_id = self.tokenizer.mask_token_id

    def apply_vq(self, cfg: VQConfig, step_provider: Callable[[], int]) -> None:
        mt = getattr(self.model.config, "model_type", "")
        mt = (mt or "").lower()
        if mt == "bert":
            from .bert_vq import apply_vq_to_bert

            apply_vq_to_bert(self.model, cfg, step_provider)
        elif mt == "roberta":
            from .roberta_vq import apply_vq_to_roberta

            apply_vq_to_roberta(self.model, cfg, step_provider)
        elif mt in ("distilbert",):
            from .distilbert_vq import apply_vq_to_distilbert

            apply_vq_to_distilbert(self.model, cfg, step_provider)
        else:
            raise ValueError(f"VQ hooks not implemented for model_type={mt}")

    def tokenize(self, text: str) -> Tokenised:
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return Tokenised(
            input_ids=input_ids, attention_mask=attention_mask, tokens=tokens
        )

    @torch.no_grad()
    def token_surprisal(
        self,
        text: str,
        window_mask: Optional[List[int]] = None,
        batch_size: int = 64,
    ) -> Tuple[List[float], Tokenised, List[int]]:
        tok = self.tokenize(text)
        ids = tok.input_ids.clone()
        B, T = ids.shape
        assert B == 1
        surprisals = [0.0] * T
        max_len = int(getattr(self.model.config, "max_position_embeddings", 512))
        max_len = max(2, max_len)

        all_idx = list(range(T)) if window_mask is None else window_mask
        specials = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        eligible = []
        for i in all_idx:
            orig_id = int(ids[0, i])
            if orig_id in specials:
                surprisals[i] = float("inf")
            else:
                eligible.append(i)

        if T <= max_len:
            for start in range(0, len(eligible), max(1, batch_size)):
                chunk = eligible[start : start + batch_size]
                if not chunk:
                    continue
                batch = ids.repeat(len(chunk), 1)
                for bi, pos in enumerate(chunk):
                    batch[bi, pos] = self.mask_token_id
                attn = tok.attention_mask.repeat(len(chunk), 1)
                logits = self.model(input_ids=batch, attention_mask=attn).logits
                probs_pos = torch.softmax(
                    logits[torch.arange(len(chunk)), chunk], dim=-1
                )
                true_ids = ids[0, chunk]
                p_true = probs_pos[torch.arange(len(chunk)), true_ids]
                p_true = torch.clamp(p_true, min=1e-12)
                s_bits = (-torch.log2(p_true)).tolist()
                for pos, s in zip(chunk, s_bits):
                    surprisals[pos] = float(s)
        else:
            overlap = max(16, max_len // 4)
            stride = max(1, max_len - overlap)
            computed = [False] * T
            win_start = 0
            while win_start < T:
                win_end = min(T, win_start + max_len)
                win_chunk_abs = [
                    i
                    for i in eligible
                    if (not computed[i]) and (win_start <= i < win_end)
                ]
                if win_chunk_abs:
                    rel_all = [i - win_start for i in win_chunk_abs]
                    for inner in range(0, len(win_chunk_abs), max(1, batch_size)):
                        sub_abs = win_chunk_abs[inner : inner + max(1, batch_size)]
                        sub_rel = rel_all[inner : inner + max(1, batch_size)]
                        batch = ids[:, win_start:win_end].repeat(len(sub_abs), 1)
                        for bi, pos_rel in enumerate(sub_rel):
                            batch[bi, pos_rel] = self.mask_token_id
                        attn = tok.attention_mask[:, win_start:win_end].repeat(
                            len(sub_abs), 1
                        )
                        logits = self.model(input_ids=batch, attention_mask=attn).logits
                        probs_pos = torch.softmax(
                            logits[torch.arange(len(sub_abs)), sub_rel], dim=-1
                        )
                        true_ids = ids[0, sub_abs]
                        p_true = probs_pos[torch.arange(len(sub_abs)), true_ids]
                        p_true = torch.clamp(p_true, min=1e-12)
                        s_bits = (-torch.log2(p_true)).tolist()
                        for pos_abs, s in zip(sub_abs, s_bits):
                            surprisals[pos_abs] = float(s)
                            computed[pos_abs] = True
                if win_end == T:
                    break
                win_start += stride
        return surprisals, tok, eligible

    @torch.no_grad()
    def token_ranks(
        self, text: str, indices: List[int], batch_size: int = 64
    ) -> Tuple[Dict[int, int], Tokenised]:
        tok = self.tokenize(text)
        ids = tok.input_ids.clone()
        B, T = ids.shape
        max_len = int(getattr(self.model.config, "max_position_embeddings", 512))
        max_len = max(2, max_len)
        specials = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        rank_by_index: Dict[int, int] = {}
        eligible = [i for i in indices if int(ids[0, i]) not in specials]
        for i in indices:
            if int(ids[0, i]) in specials:
                rank_by_index[i] = 1

        if T <= max_len:
            for start in range(0, len(eligible), max(1, batch_size)):
                chunk = eligible[start : start + batch_size]
                if not chunk:
                    continue
                batch = ids.repeat(len(chunk), 1)
                for bi, pos in enumerate(chunk):
                    batch[bi, pos] = self.mask_token_id
                attn = tok.attention_mask.repeat(len(chunk), 1)
                logits = self.model(input_ids=batch, attention_mask=attn).logits
                scores_pos = logits[torch.arange(len(chunk)), chunk]
                true_ids = ids[0, chunk]
                s_true = scores_pos[torch.arange(len(chunk)), true_ids]
                ranks = (scores_pos > s_true.unsqueeze(-1)).sum(dim=1) + 1
                for pos, r in zip(chunk, ranks.tolist()):
                    rank_by_index[pos] = int(r)
        else:
            overlap = max(16, max_len // 4)
            stride = max(1, max_len - overlap)
            computed = {i: False for i in eligible}
            win_start = 0
            while win_start < T:
                win_end = min(T, win_start + max_len)
                win_chunk_abs = [
                    i
                    for i in eligible
                    if (not computed[i]) and (win_start <= i < win_end)
                ]
                if win_chunk_abs:
                    rel_all = [i - win_start for i in win_chunk_abs]
                    for inner in range(0, len(win_chunk_abs), max(1, batch_size)):
                        sub_abs = win_chunk_abs[inner : inner + max(1, batch_size)]
                        sub_rel = rel_all[inner : inner + max(1, batch_size)]
                        batch = ids[:, win_start:win_end].repeat(len(sub_abs), 1)
                        for bi, pos_rel in enumerate(sub_rel):
                            batch[bi, pos_rel] = self.mask_token_id
                        attn = tok.attention_mask[:, win_start:win_end].repeat(
                            len(sub_abs), 1
                        )
                        logits = self.model(input_ids=batch, attention_mask=attn).logits
                        scores_pos = logits[torch.arange(len(sub_abs)), sub_rel]
                        true_ids = ids[0, sub_abs]
                        s_true = scores_pos[torch.arange(len(sub_abs)), true_ids]
                        ranks = (scores_pos > s_true.unsqueeze(-1)).sum(dim=1) + 1
                        for pos_abs, r in zip(sub_abs, ranks.tolist()):
                            rank_by_index[pos_abs] = int(r)
                            computed[pos_abs] = True
                if win_end == T:
                    break
                win_start += stride
        return rank_by_index, tok

    @torch.no_grad()
    def topk_indices(self, text: str, index: int, k: int) -> List[int]:
        out = self.topk_indices_batch(text, [index], k)
        return out.get(index, [])

    @torch.no_grad()
    def topk_indices_batch(
        self, text: str, indices: List[int], k: int, batch_size: int = 64
    ) -> Dict[int, List[int]]:
        tok = self.tokenize(text)
        ids = tok.input_ids.clone()
        B, T = ids.shape
        max_len = int(getattr(self.model.config, "max_position_embeddings", 512))
        max_len = max(2, max_len)
        specials = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        eligible = [i for i in indices if int(ids[0, i]) not in specials]
        result: Dict[int, List[int]] = {i: [] for i in indices}
        for i in indices:
            if int(ids[0, i]) in specials:
                result[i] = [int(ids[0, i])]

        if T <= max_len:
            for start in range(0, len(eligible), max(1, batch_size)):
                chunk = eligible[start : start + batch_size]
                if not chunk:
                    continue
                batch = ids.repeat(len(chunk), 1)
                for bi, pos in enumerate(chunk):
                    batch[bi, pos] = self.mask_token_id
                attn = tok.attention_mask.repeat(len(chunk), 1)
                logits = self.model(input_ids=batch, attention_mask=attn).logits
                for bi, pos in enumerate(chunk):
                    topk = torch.topk(logits[bi, pos], k=k, dim=-1).indices.tolist()
                    result[pos] = [int(t) for t in topk]
        else:
            overlap = max(16, max_len // 4)
            stride = max(1, max_len - overlap)
            computed = {i: False for i in eligible}
            win_start = 0
            while win_start < T:
                win_end = min(T, win_start + max_len)
                win_chunk_abs = [
                    i
                    for i in eligible
                    if (not computed[i]) and (win_start <= i < win_end)
                ]
                if win_chunk_abs:
                    rel_all = [i - win_start for i in win_chunk_abs]
                    for inner in range(0, len(win_chunk_abs), max(1, batch_size)):
                        sub_abs = win_chunk_abs[inner : inner + max(1, batch_size)]
                        sub_rel = rel_all[inner : inner + max(1, batch_size)]
                        batch = ids[:, win_start:win_end].repeat(len(sub_abs), 1)
                        for bi, pos_rel in enumerate(sub_rel):
                            batch[bi, pos_rel] = self.mask_token_id
                        attn = tok.attention_mask[:, win_start:win_end].repeat(
                            len(sub_abs), 1
                        )
                        logits = self.model(input_ids=batch, attention_mask=attn).logits
                        for bi, pos_abs in enumerate(sub_abs):
                            pos_rel = sub_rel[bi]
                            topk = torch.topk(
                                logits[bi, pos_rel], k=k, dim=-1
                            ).indices.tolist()
                            result[pos_abs] = [int(t) for t in topk]
                            computed[pos_abs] = True
                if win_end == T:
                    break
                win_start += stride
        return result

    @torch.no_grad()
    def ranks_and_topk(
        self, text: str, indices: List[int], k: int, batch_size: int = 64
    ) -> Tuple[Dict[int, int], Dict[int, List[int]], Tokenised]:
        tok = self.tokenize(text)
        ids = tok.input_ids.clone()
        B, T = ids.shape
        max_len = int(getattr(self.model.config, "max_position_embeddings", 512))
        max_len = max(2, max_len)
        specials = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        rank_by_index: Dict[int, int] = {}
        topk_by_index: Dict[int, List[int]] = {i: [] for i in indices}
        eligible = [i for i in indices if int(ids[0, i]) not in specials]
        for i in indices:
            if int(ids[0, i]) in specials:
                rank_by_index[i] = 1
                topk_by_index[i] = [int(ids[0, i])]

        if T <= max_len:
            for start in range(0, len(eligible), max(1, batch_size)):
                chunk = eligible[start : start + batch_size]
                if not chunk:
                    continue
                batch = ids.repeat(len(chunk), 1)
                for bi, pos in enumerate(chunk):
                    batch[bi, pos] = self.mask_token_id
                attn = tok.attention_mask.repeat(len(chunk), 1)
                logits = self.model(input_ids=batch, attention_mask=attn).logits
                scores_pos = logits[torch.arange(len(chunk)), chunk]
                topk = torch.topk(scores_pos, k=k, dim=-1).indices
                for bi, pos in enumerate(chunk):
                    topk_by_index[pos] = [int(t) for t in topk[bi].tolist()]
                true_ids = ids[0, chunk]
                s_true = scores_pos[torch.arange(len(chunk)), true_ids]
                ranks = (scores_pos > s_true.unsqueeze(-1)).sum(dim=1) + 1
                for pos, r in zip(chunk, ranks.tolist()):
                    rank_by_index[pos] = int(r)
        else:
            overlap = max(16, max_len // 4)
            stride = max(1, max_len - overlap)
            computed = {i: False for i in eligible}
            win_start = 0
            while win_start < T:
                win_end = min(T, win_start + max_len)
                win_chunk_abs = [
                    i
                    for i in eligible
                    if (not computed[i]) and (win_start <= i < win_end)
                ]
                if win_chunk_abs:
                    rel_all = [i - win_start for i in win_chunk_abs]
                    for inner in range(0, len(win_chunk_abs), max(1, batch_size)):
                        sub_abs = win_chunk_abs[inner : inner + max(1, batch_size)]
                        sub_rel = rel_all[inner : inner + max(1, batch_size)]
                        batch = ids[:, win_start:win_end].repeat(len(sub_abs), 1)
                        for bi, pos_rel in enumerate(sub_rel):
                            batch[bi, pos_rel] = self.mask_token_id
                        attn = tok.attention_mask[:, win_start:win_end].repeat(
                            len(sub_abs), 1
                        )
                        logits = self.model(input_ids=batch, attention_mask=attn).logits
                        scores_pos = logits[torch.arange(len(sub_abs)), sub_rel]
                        topk = torch.topk(scores_pos, k=k, dim=-1).indices
                        for bi, pos_abs in enumerate(sub_abs):
                            topk_by_index[pos_abs] = [int(t) for t in topk[bi].tolist()]
                        true_ids = ids[0, sub_abs]
                        s_true = scores_pos[torch.arange(len(sub_abs)), true_ids]
                        ranks = (scores_pos > s_true.unsqueeze(-1)).sum(dim=1) + 1
                        for pos_abs, r in zip(sub_abs, ranks.tolist()):
                            rank_by_index[pos_abs] = int(r)
                            computed[pos_abs] = True
                if win_end == T:
                    break
                win_start += stride
        return rank_by_index, topk_by_index, tok

    @torch.no_grad()
    def ranks_and_topk_batched(
        self, text: str, mask_indices: List[int], k: int, batch_size: int = 64
    ) -> Tuple[Dict[int, int], Dict[int, List[int]], Tokenised]:
        return self.ranks_and_topk(text, mask_indices, k=k, batch_size=batch_size)

    @torch.no_grad()
    def surprisal_batched(
        self,
        text: str,
        eligible_indices: Optional[List[int]] = None,
        batch_size: int = 64,
    ) -> Tuple[List[float], Tokenised, List[int]]:
        return self.token_surprisal(
            text, window_mask=eligible_indices, batch_size=batch_size
        )
