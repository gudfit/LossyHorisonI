from __future__ import annotations

import argparse
import os
import csv
from datetime import datetime
from statistics import mean, pstdev
from typing import List, Dict, Tuple

from ..models import MLMScorer
from ..models.bert_vq import VQConfig
from ..policies import select_mask_set
from ..codecs.vqre_codec import VQREEncoder, VQREDecoder
from ..codecs.vqre import refine_decode
from ..bits.ar_coder import NGramARCoder
from ..utils.repro import set_global_seed
import torch
from ..metrics.char_fidelity import char_fidelity


def read_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description="Evaluate VQ+RE RD curves")
    ap.add_argument("--model", type=str, default="bert-base-cased")
    ap.add_argument("--texts-file", type=str, required=True)
    ap.add_argument("--p-mask-list", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--rank-list", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--anchors-every", type=int, nargs="+", default=[0, 16, 32])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument(
        "--pos-coding", type=str, choices=["min", "enumerative", "rle"], default="min"
    )
    ap.add_argument("--refine-steps", type=int, default=2)
    ap.add_argument("--refine-topm", type=int, default=8)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--half", action="store_true", help="Run inference in fp16 on CUDA")
    args = ap.parse_args()

    texts = read_texts(args.texts_file)
    scorer = MLMScorer(args.model)

    cfg = VQConfig()
    scorer.apply_vq(cfg, step_provider=lambda: 10_000)

    try:
        import torch

        if args.half and torch.cuda.is_available():
            scorer.model.half()
            print("[info] Enabled fp16 inference on CUDA")
    except Exception:
        pass

    results: Dict[Tuple[float, int, int], Dict[str, List[float]]] = {}

    for p in args.p_mask_list:
        for K in args.rank_list:
            for ae in args.anchors_every:
                key = (p, K, ae)
                results[key] = {"bpc": [], "charfid": []}

    for s in range(args.seeds):
        set_global_seed(args.seed_base + s, deterministic=True)
        for p_mask in args.p_mask_list:
            for K in args.rank_list:
                for ae in args.anchors_every:
                    enc = VQREEncoder(
                        rank_threshold=K,
                        anchors_every=ae,
                        vocab_size=scorer.tokenizer.vocab_size,
                        ar_coder_anchors=NGramARCoder(
                            order=3, vocab_size=scorer.tokenizer.vocab_size
                        ),
                        ar_coder_corr=NGramARCoder(
                            order=3, vocab_size=scorer.tokenizer.vocab_size
                        ),
                    )
                    dec = VQREDecoder(rank_threshold=K)
                    total_bits = 0.0
                    total_chars = 0
                    cf_vals: List[float] = []

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

                        ranks, topk_by_index, _ = scorer.ranks_and_topk(
                            text, mask_idx, k=K
                        )
                        token_ids = ids_list

                        payload = enc.encode(token_ids, mask_idx, ranks, topk_by_index)

                        top1 = {
                            i: (tk[0] if tk else token_ids[i])
                            for i, tk in topk_by_index.items()
                        }

                        full_ids = dec.decode(
                            payload,
                            mask_idx,
                            topk_by_index,
                            top1,
                            original_token_ids=token_ids,
                        )
                        if args.refine_steps and args.refine_steps > 0:
                            ids_t = torch.tensor(
                                [full_ids], dtype=torch.long, device=scorer.device
                            )
                            ids_t = refine_decode(
                                scorer.tokenizer,
                                scorer.model,
                                ids_t,
                                steps=args.refine_steps,
                                topm=args.refine_topm,
                            )
                            full_ids = ids_t[0].tolist()
                        recon = scorer.tokenizer.decode(
                            full_ids, skip_special_tokens=True
                        )

                        total_bits += payload.total_bits
                        total_chars += len(text)
                        cf_vals.append(char_fidelity(text, recon))

                    bpc = total_bits / max(1, total_chars)
                    cf = sum(cf_vals) / max(1, len(cf_vals))
                    results[(p_mask, K, ae)]["bpc"].append(bpc)
                    results[(p_mask, K, ae)]["charfid"].append(cf)
                    print(
                        f"VQRE seed={s} p={p_mask:.2f} K={K} A={ae} -> BPC={bpc:.4f}, CharFid={cf:.4f}"
                    )

    if args.out_csv:
        rows = []
        for (p, K, ae), vals in results.items():
            bpc_m = mean(vals["bpc"])
            bpc_sd = pstdev(vals["bpc"]) if len(vals["bpc"]) > 1 else 0.0
            cf_m = mean(vals["charfid"])
            cf_sd = pstdev(vals["charfid"]) if len(vals["charfid"]) > 1 else 0.0
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "codec": "vqre",
                    "model": args.model,
                    "p_mask": p,
                    "K": K,
                    "anchors_every": ae,
                    "seeds": len(vals["bpc"]),
                    "BPC_mean": bpc_m,
                    "BPC_sd": bpc_sd,
                    "CharFid_mean": cf_m,
                    "CharFid_sd": cf_sd,
                }
            )
        os.makedirs(
            os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True
        )
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
