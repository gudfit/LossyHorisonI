from __future__ import annotations

import argparse
from typing import Dict, List

from ..models import MLMScorer
from ..policies import select_mask_set
from ..codecs import PMEncoder, PMDecoder, EPCEncoder, EPCDecoder
from ..bits.ar_coder import NGramARCoder


def main():
    ap = argparse.ArgumentParser(description="Compress a single text with PM/EPC")
    ap.add_argument("--model", type=str, default="bert-base-cased")
    ap.add_argument("--codec", type=str, choices=["pm", "epc"], default="pm")
    ap.add_argument("--p-mask", type=float, default=0.6)
    ap.add_argument("--rank-threshold", type=int, default=16)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument(
        "--pos-coding", type=str, choices=["min", "enumerative", "rle"], default="min"
    )
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    scorer = MLMScorer(args.model)
    surprisals, tok, eligible = scorer.token_surprisal(args.text)
    specials = {
        scorer.tokenizer.cls_token_id,
        scorer.tokenizer.sep_token_id,
        scorer.tokenizer.pad_token_id,
    }
    ids_list = tok.input_ids[0].tolist()
    forbidden = {i for i, tid in enumerate(ids_list) if tid in specials}
    mask_idx, keep_idx = select_mask_set(
        surprisals, p_mask=args.p_mask, forbidden=forbidden, eligible=eligible
    )
    token_ids = tok.input_ids[0].tolist()
    ar_coder = NGramARCoder(order=3, vocab_size=scorer.tokenizer.vocab_size)

    if args.codec == "pm":
        enc = PMEncoder(
            aux_token_coder=ar_coder,
            vocab_size=scorer.tokenizer.vocab_size,
            pos_coding=args.pos_coding,
        )
        payload = enc.encode(token_ids, keep_idx)

        _, topk_by_index, _ = scorer.ranks_and_topk(
            args.text, mask_idx, k=args.rank_threshold
        )
        top1 = {i: tk[0] if tk else token_ids[i] for i, tk in topk_by_index.items()}
        dec = PMDecoder(scorer.tokenizer)
        full_ids = dec.decode(payload, mask_idx, top1)
        recon = scorer.tokenizer.decode(full_ids, skip_special_tokens=True)
        print(
            f"PM bits: pos={payload.bits_pos:.2f} [{payload.pos_method}], tok={payload.bits_tok:.2f}, total={payload.total_bits:.2f}"
        )
        print("Reconstruction:\n", recon)
    else:
        ranks, topk_by_index, _ = scorer.ranks_and_topk(
            args.text, mask_idx, args.rank_threshold
        )
        enc = EPCEncoder(
            rank_threshold=args.rank_threshold,
            fallback=(not args.no_fallback),
            aux_token_coder_kept=ar_coder,
            aux_token_coder_corr=NGramARCoder(
                order=3, vocab_size=scorer.tokenizer.vocab_size
            ),
            vocab_size=scorer.tokenizer.vocab_size,
        )

        enc.pos_coding = args.pos_coding
        payload = enc.encode(token_ids, keep_idx, mask_idx, ranks)
        top1 = {i: tk[0] if tk else token_ids[i] for i, tk in topk_by_index.items()}
        dec = EPCDecoder(rank_threshold=args.rank_threshold)
        full_ids = dec.decode(payload, mask_idx, topk_by_index, top1)
        recon = scorer.tokenizer.decode(full_ids, skip_special_tokens=True)
        print(
            f"EPC bits: pos={payload.bits_pos:.2f} [{payload.pos_method}], kept={payload.bits_tok_kept:.2f}, flag={payload.bits_flag:.2f} (ac={payload.bits_flag_ac:.2f}), rank={payload.bits_rank:.2f} (ac={payload.bits_rank_ac:.2f}), corr_tok={payload.bits_corr_tok:.2f} (ac={payload.bits_corr_tok_ac:.2f}), total={payload.total_bits:.2f}"
        )
        print("Reconstruction:\n", recon)


if __name__ == "__main__":
    main()
