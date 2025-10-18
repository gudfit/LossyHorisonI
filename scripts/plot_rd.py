#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def _get_float(row, *keys, default=float("nan")):
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(row[k])
            except Exception:
                pass
    return default


def main():
    ap = argparse.ArgumentParser(description="Plot RD curves with error bars from CSV")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--codec", type=str, choices=["pm", "epc", "vqre"], default="epc")
    ap.add_argument("--out", type=str, default="rd_plot.png")
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("codec") != args.codec:
            continue
        key = f"K={row.get('K','')}" if args.codec != "pm" else "PM"
        groups[key].append(row)

    plt.figure(figsize=(6, 4))
    for label, items in groups.items():
        data = []
        for it in items:
            bpc = _get_float(it, "BPC_mean", "BPC_entropy_mean", "BPC_ac_mean")
            bpc_sd = _get_float(
                it, "BPC_sd", "BPC_entropy_sd", "BPC_ac_sd", default=0.0
            )
            cf = _get_float(it, "CharFid_mean")
            cf_sd = _get_float(it, "CharFid_sd", default=0.0)
            if not (bpc == bpc and cf == cf):
                continue
            data.append((bpc, 1.0 - cf, bpc_sd, cf_sd))
        data.sort(key=lambda x: x[0])
        if not data:
            continue
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        xerr = [d[2] for d in data]
        yerr = [d[3] for d in data]
        plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, label=label, marker="o", capsize=3)

    plt.xlabel("BPC (lower is better)")
    plt.ylabel("1 - CharFidelity (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
