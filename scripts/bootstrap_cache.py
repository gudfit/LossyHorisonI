from __future__ import annotations

import argparse
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="bert-base-cased", help="HF model id or local path"
    )
    ap.add_argument(
        "--hf_home", default="cache/hf", help="Base cache directory for HF artifacts"
    )
    ap.add_argument(
        "--dataset",
        default="wikitext",
        help="HF dataset name (e.g., wikitext) or 'none' to skip",
    )
    ap.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="HF dataset config name (if required)",
    )
    ap.add_argument(
        "--dataset-split",
        default="test",
        help="Dataset split to download (e.g., train/validation/test)",
    )
    ap.add_argument(
        "--out-texts",
        default="examples/wikitext2_test.txt",
        help="Where to write concatenated text lines",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on number of examples (0 = all)",
    )
    args = ap.parse_args()

    os.makedirs(args.hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", os.path.join(args.hf_home, "transformers")
    )
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.hf_home, "datasets"))

    print(f"Downloading {args.model} into {args.hf_home} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForMaskedLM.from_pretrained(args.model)
    _ = tok("hello")
    print("Model ready.")

    ds_name = (args.dataset or "").strip().lower()
    if ds_name and ds_name != "none":
        try:
            from datasets import load_dataset
        except Exception as e:
            print(
                "[WARN] HuggingFace 'datasets' not installed; skipping dataset download."
            )
        else:
            print(
                f"Downloading dataset {args.dataset}/{args.dataset_config} [{args.dataset_split}] ..."
            )
            try:
                dataset = load_dataset(
                    args.dataset, args.dataset_config, split=args.dataset_split
                )
            except TypeError:
                dataset = load_dataset(args.dataset, split=args.dataset_split)
            os.makedirs(
                os.path.dirname(os.path.abspath(args.out_texts)) or ".", exist_ok=True
            )
            num = 0
            with open(args.out_texts, "w", encoding="utf-8") as f:
                for ex in dataset:
                    t = ex["text"] if "text" in ex else None
                    if t is None:
                        for v in ex.values():
                            if isinstance(v, str):
                                t = v
                                break
                    if not t:
                        continue
                    t = t.strip()
                    if not t:
                        continue
                    f.write(t + "\n")
                    num += 1
                    if args.max_examples and num >= args.max_examples:
                        break
            print(f"Wrote {num} lines to {args.out_texts}")

    print("Done.")


if __name__ == "__main__":
    main()
