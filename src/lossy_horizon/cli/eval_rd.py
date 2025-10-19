from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Dict
import numpy as np
import csv
from datetime import datetime

from ..models import MLMScorer
from ..policies import select_mask_set
from ..codecs import PMEncoder, EPCEncoder, PMDecoder, EPCDecoder
from ..bits import RankPMF, RankModel
from ..bits.ar_coder import NGramARCoder
from ..utils.repro import set_global_seed

try:
    import sacrebleu
except Exception:
    sacrebleu = None
try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def char_fidelity(ref: str, hyp: str) -> float:
    ed = levenshtein(ref, hyp)
    denom = max(1, max(len(ref), len(hyp)))
    return 1.0 - (ed / denom)


def msd(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def read_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Sweep RD curves (PM/EPC) with metrics and seeds"
    )
    ap.add_argument("--model", type=str, default="bert-base-cased")
    ap.add_argument("--codec", type=str, choices=["pm", "epc"], default="epc")
    ap.add_argument("--texts-file", type=str, required=True)
    ap.add_argument(
        "--p-mask-list", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8]
    )
    ap.add_argument("--rank-list", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--rank-calib-file", type=str, default=None)
    ap.add_argument(
        "--rank-calib-out",
        type=str,
        default=None,
        help="Optional dir to save calibrated rank PMFs as JSON (counts + alpha)",
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument(
        "--pos-coding", type=str, choices=["min", "enumerative", "rle"], default="min"
    )
    ap.add_argument("--half", action="store_true", help="Run inference in fp16 on CUDA")
    args = ap.parse_args()

    texts = read_texts(args.texts_file)
    calib_texts = []
    if args.rank_calib_file:
        calib_texts = read_texts(args.rank_calib_file)
        overlap = set(calib_texts) & set(texts)
        if overlap:
            print(
                f"[WARN] Calibration file shares {len(overlap)} texts with evaluation set; consider using a disjoint calibration set."
            )
    scorer = MLMScorer(args.model)

    try:
        import torch

        if getattr(args, "half", False) and torch.cuda.is_available():
            scorer.model.half()
            print("[info] Enabled fp16 inference on CUDA")
    except Exception:
        pass

    if args.codec == "pm":
        results: Dict[float, Dict[str, List[float]]] = {
            p: {"bpc": [], "charfid": [], "chrf": [], "berts": []}
            for p in args.p_mask_list
        }
        for s in range(args.seeds):
            set_global_seed(args.seed_base + s, deterministic=True)
            for p_mask in args.p_mask_list:
                ar_coder = NGramARCoder(order=3, vocab_size=scorer.tokenizer.vocab_size)
                enc = PMEncoder(
                    aux_token_coder=ar_coder,
                    vocab_size=scorer.tokenizer.vocab_size,
                    pos_coding=args.pos_coding,
                )
                dec = PMDecoder(scorer.tokenizer)
                total_bits = 0.0
                total_chars = 0
                refs: List[str] = []
                hyps: List[str] = []
                for text in texts:
                    surprisals, tok, eligible = scorer.token_surprisal(text)
                    specials = {
                        scorer.tokenizer.cls_token_id,
                        scorer.tokenizer.sep_token_id,
                        scorer.tokenizer.pad_token_id,
                    }
                    ids_list = tok.input_ids[0].tolist()
                    forbidden = {i for i, tid in enumerate(ids_list) if tid in specials}
                    mask_idx, keep_idx = select_mask_set(
                        surprisals,
                        p_mask=p_mask,
                        forbidden=forbidden,
                        eligible=eligible,
                    )
                    token_ids = tok.input_ids[0].tolist()
                    payload = enc.encode(token_ids, keep_idx)
                    _, topk_by_index, _ = scorer.ranks_and_topk(text, mask_idx, k=1)
                    top1 = {
                        i: tk[0] if tk else token_ids[i]
                        for i, tk in topk_by_index.items()
                    }
                    full_ids = dec.decode(payload, mask_idx, top1)
                    recon = scorer.tokenizer.decode(full_ids, skip_special_tokens=True)
                    refs.append(text)
                    hyps.append(recon)
                    total_bits += payload.total_bits
                    total_chars += len(text)
                bpc = total_bits / max(1, total_chars)
                cfid = sum(char_fidelity(r, h) for r, h in zip(refs, hyps)) / max(
                    1, len(refs)
                )
                results[p_mask]["bpc"].append(bpc)
                results[p_mask]["charfid"].append(cfid)
                if sacrebleu is not None:
                    results[p_mask]["chrf"].append(
                        sacrebleu.corpus_chrf(hyps, [refs]).score / 100.0
                    )
                if bert_score is not None:
                    _, _, F1 = bert_score(hyps, refs, lang="en")
                    results[p_mask]["berts"].append(float(F1.mean().item()))
        for p_mask in args.p_mask_list:
            mb, sb = msd(results[p_mask]["bpc"])
            mf, sf = msd(results[p_mask]["charfid"])
            chrf_vals = results[p_mask]["chrf"]
            berts_vals = results[p_mask]["berts"]
            chrf_str = (
                "NA"
                if not chrf_vals
                else f"{msd(chrf_vals)[0]:.4f} ± {msd(chrf_vals)[1]:.4f}"
            )
            bert_str = (
                "NA"
                if not berts_vals
                else f"{msd(berts_vals)[0]:.4f} ± {msd(berts_vals)[1]:.4f}"
            )
            print(
                f"PM p_mask={p_mask:.2f} -> BPC={mb:.4f} ± {sb:.4f}, CharFid={mf:.4f} ± {sf:.4f}, ChrF={chrf_str}, BERTScore={bert_str}"
            )
    else:
        results: Dict[Tuple[float, int], Dict[str, List[float]]] = {
            (p, K): {"bpc": [], "bpc_ac": [], "charfid": [], "chrf": [], "berts": []}
            for p in args.p_mask_list
            for K in args.rank_list
        }
        for s in range(args.seeds):
            set_global_seed(args.seed_base + s, deterministic=True)
            for p_mask in args.p_mask_list:
                for K in args.rank_list:
                    ar_kept = NGramARCoder(
                        order=3, vocab_size=scorer.tokenizer.vocab_size
                    )
                    ar_corr = NGramARCoder(
                        order=3, vocab_size=scorer.tokenizer.vocab_size
                    )
                    rank_model = None
                    if args.rank_calib_file:
                        rank_model = RankModel(K=K, alpha=1.0)
                        rank_model.train_on_texts(scorer, calib_texts, p_mask=p_mask)
                        if args.rank_calib_out:
                            os.makedirs(args.rank_calib_out, exist_ok=True)
                            out_path = os.path.join(
                                args.rank_calib_out,
                                f"pmf_p{p_mask:.2f}_K{K}.json",
                            )
                            meta = rank_model.to_json_dict()
                            meta.update(
                                {
                                    "p_mask": float(p_mask),
                                    "model": args.model,
                                    "source": os.path.abspath(args.rank_calib_file),
                                }
                            )
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(meta, f, indent=2, sort_keys=True)
                    enc = EPCEncoder(
                        rank_threshold=K,
                        fallback=True,
                        aux_token_coder_kept=ar_kept,
                        aux_token_coder_corr=ar_corr,
                        vocab_size=scorer.tokenizer.vocab_size,
                        rank_model=rank_model,
                    )
                    enc.pos_coding = args.pos_coding
                    dec = EPCDecoder(rank_threshold=K)
                    total_bits = 0.0
                    total_bits_ac = 0.0
                    total_chars = 0
                    refs: List[str] = []
                    hyps: List[str] = []
                    for text in texts:
                        surprisals, tok, eligible = scorer.token_surprisal(text)
                        specials = {
                            scorer.tokenizer.cls_token_id,
                            scorer.tokenizer.sep_token_id,
                            scorer.tokenizer.pad_token_id,
                        }
                        ids_list = tok.input_ids[0].tolist()
                        forbidden = {
                            i for i, tid in enumerate(ids_list) if tid in specials
                        }
                        mask_idx, keep_idx = select_mask_set(
                            surprisals,
                            p_mask=p_mask,
                            forbidden=forbidden,
                            eligible=eligible,
                        )
                        token_ids = tok.input_ids[0].tolist()
                        ranks, topk_by_index, _ = scorer.ranks_and_topk(
                            text, mask_idx, k=K
                        )
                        payload = enc.encode(token_ids, keep_idx, mask_idx, ranks)
                        top1 = {
                            i: tk[0] if tk else token_ids[i]
                            for i, tk in topk_by_index.items()
                        }
                        full_ids = dec.decode(payload, mask_idx, topk_by_index, top1)
                        recon = scorer.tokenizer.decode(
                            full_ids, skip_special_tokens=True
                        )
                        refs.append(text)
                        hyps.append(recon)
                        total_bits += payload.total_bits
                        total_bits_ac += (
                            payload.bits_pos
                            + payload.bits_tok_kept
                            + payload.bits_flag_ac
                            + payload.bits_rank_ac
                            + payload.bits_corr_tok_ac
                        )
                        total_chars += len(text)
                    bpc = total_bits / max(1, total_chars)
                    bpc_ac = total_bits_ac / max(1, total_chars)
                    cf = sum(char_fidelity(r, h) for r, h in zip(refs, hyps)) / max(
                        1, len(refs)
                    )
                    results[(p_mask, K)]["bpc"].append(bpc)
                    results[(p_mask, K)]["bpc_ac"].append(bpc_ac)
                    results[(p_mask, K)]["charfid"].append(cf)
                    if sacrebleu is not None:
                        results[(p_mask, K)]["chrf"].append(
                            sacrebleu.corpus_chrf(hyps, [refs]).score / 100.0
                        )
                    if bert_score is not None:
                        _, _, F1 = bert_score(hyps, refs, lang="en")
                        results[(p_mask, K)]["berts"].append(float(F1.mean().item()))
        for p_mask in args.p_mask_list:
            for K in args.rank_list:
                xs = results[(p_mask, K)]
                mb, sb = msd(xs["bpc"])
                mba, sba = (
                    msd(xs["bpc_ac"]) if xs["bpc_ac"] else (float("nan"), float("nan"))
                )
                mf, sf = msd(xs["charfid"])
                chrf_vals = xs["chrf"]
                berts_vals = xs["berts"]
                chrf_str = (
                    "NA"
                    if not chrf_vals
                    else f"{msd(chrf_vals)[0]:.4f} ± {msd(chrf_vals)[1]:.4f}"
                )
                bert_str = (
                    "NA"
                    if not berts_vals
                    else f"{msd(berts_vals)[0]:.4f} ± {msd(berts_vals)[1]:.4f}"
                )
                print(
                    f"EPC p_mask={p_mask:.2f}, K={K} -> BPC={mb:.4f} ± {sb:.4f}, CharFid={mf:.4f} ± {sf:.4f}, ChrF={chrf_str}, BERTScore={bert_str}"
                )

    if args.out_csv and args.codec == "epc":
        rows = []
        for p_mask in args.p_mask_list:
            for K in args.rank_list:
                xs = results[(p_mask, K)]
                mb, sb = msd(xs["bpc"])
                mba, sba = (
                    msd(xs["bpc_ac"]) if xs["bpc_ac"] else (float("nan"), float("nan"))
                )
                mf, sf = msd(xs["charfid"])
                mchr, schr = (
                    msd(xs["chrf"]) if xs["chrf"] else (float("nan"), float("nan"))
                )
                mbert, sbert = (
                    msd(xs["berts"]) if xs["berts"] else (float("nan"), float("nan"))
                )
                rows.append(
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "codec": "epc",
                        "model": args.model,
                        "p_mask": p_mask,
                        "K": K,
                        "seeds": args.seeds,
                        "BPC_entropy_mean": mb,
                        "BPC_entropy_sd": sb,
                        "BPC_ac_mean": mba,
                        "BPC_ac_sd": sba,
                        "CharFid_mean": mf,
                        "CharFid_sd": sf,
                        "ChrF_mean": mchr,
                        "ChrF_sd": schr,
                        "BERTScore_mean": mbert,
                        "BERTScore_sd": sbert,
                    }
                )
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
