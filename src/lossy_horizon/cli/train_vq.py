from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from ..models import MLMScorer
from ..models.bert_vq import VQConfig, apply_vq_to_bert, collect_vq_losses_from_bert
from ..utils.repro import set_global_seed


class TextLineDataset(Dataset):
    def __init__(self, files: List[str]) -> None:
        self.lines: List[str] = []
        for p in files:
            with open(p, "r", encoding="utf-8") as f:
                self.lines.extend([ln.strip() for ln in f if ln.strip()])

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        return self.lines[idx]


def _find_text_files(root: str) -> List[str]:
    paths: List[str] = []
    if os.path.isdir(root):
        for dn, _sub, files in os.walk(root):
            for fn in files:
                if fn.endswith(".txt") or fn.endswith(".text") or fn.endswith(".dat"):
                    paths.append(os.path.join(dn, fn))
    elif os.path.isfile(root):
        paths = [root]
    return paths


def main():
    ap = argparse.ArgumentParser(
        description="Light VQ fine-tuning for masked LM with KV quantisation"
    )
    ap.add_argument("--model", type=str, default="bert-base-cased")
    ap.add_argument("--data", type=str, default="data/calib")
    ap.add_argument("--save-dir", type=str, default="runs/vq_ckpt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed, deterministic=True)

    files = _find_text_files(args.data)
    if not files:
        raise FileNotFoundError(f"No text files found under {args.data}")
    ds = TextLineDataset(files)

    scorer = MLMScorer(args.model)
    tokenizer = scorer.tokenizer
    model = scorer.model
    device = scorer.device

    global_step = 0

    def step_provider() -> int:
        return global_step

    cfg = VQConfig()
    apply_vq_to_bert(model, cfg, step_provider)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    try:
        from transformers import DataCollatorForLanguageModeling
    except Exception as e:
        raise RuntimeError("transformers is required for masking collator") from e

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    def collate_fn(batch_texts: List[str]):
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        batch = collator(enc["input_ids"])
        batch["attention_mask"] = enc["attention_mask"]
        return batch

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_vq = 0.0
        count = 0
        for batch in dl:
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optim.zero_grad(set_to_none=True)
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            ce_loss = out.loss
            vq_losses = collect_vq_losses_from_bert(model)
            vq_term = (
                sum(vq_losses.values())
                if vq_losses
                else torch.tensor(0.0, device=device)
            )
            loss = ce_loss + vq_term
            loss.backward()
            optim.step()

            total_loss += float(loss.detach().cpu().item())
            total_ce += float(ce_loss.detach().cpu().item())
            total_vq += float(vq_term.detach().cpu().item())
            count += 1

        avg_loss = total_loss / max(1, count)
        avg_ce = total_ce / max(1, count)
        avg_vq = total_vq / max(1, count)
        print(
            f"epoch {epoch+1}/{args.epochs} step={global_step} loss={avg_loss:.4f} ce={avg_ce:.4f} vq={avg_vq:.4f}"
        )

    print(f"Saving to {args.save_dir}")
    model.eval()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
