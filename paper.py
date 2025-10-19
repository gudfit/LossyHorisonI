from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], env=None):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="One-button paper runner")
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of HF model ids; if set overrides --model.",
    )
    ap.add_argument("--texts", default="examples/texts.txt")
    ap.add_argument(
        "--calib",
        default=None,
        help="Optional separate calibration file for EPC rank PMF",
    )
    ap.add_argument("--hf_home", default="cache/hf")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=0)
    ap.add_argument(
        "--p-mask", dest="p_mask", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8]
    )
    ap.add_argument("--rank-list", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--anchors-every", type=int, nargs="+", default=[0, 16, 32])
    ap.add_argument(
        "--pos-coding", choices=["min", "enumerative", "rle"], default="min"
    )
    ap.add_argument(
        "--half", action="store_true", help="Use fp16 inference where supported"
    )
    args = ap.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"runs/paper-{ts}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(Path(args.hf_home).resolve()))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(Path(args.hf_home, "transformers").resolve())
    )
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(Path(args.hf_home, "datasets").resolve())
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    models = args.models or [args.model]
    for model_id in models:
        outdir_m = (outdir / model_id.replace("/", "_")).resolve()
        outdir_m.mkdir(parents=True, exist_ok=True)
        csvdir_m = outdir_m / "csv"
        plotsdir_m = outdir_m / "plots"
        csvdir_m.mkdir(parents=True, exist_ok=True)
        plotsdir_m.mkdir(parents=True, exist_ok=True)

        run(
            [
                sys.executable,
                "scripts/bootstrap_cache.py",
                "--model",
                model_id,
                "--hf_home",
                args.hf_home,
            ]
        )

        pm_csv = csvdir_m / "pm.csv"
        pm_cmd = [
            sys.executable,
            "-m",
            "lossy_horizon.cli.eval_rd",
            "--codec",
            "pm",
            "--model",
            model_id,
            "--texts-file",
            args.texts,
            "--p-mask-list",
            *[str(p) for p in args.p_mask],
            "--seeds",
            str(args.seeds),
            "--seed-base",
            str(args.seed_base),
            "--pos-coding",
            args.pos_coding,
            "--out-csv",
            str(pm_csv),
        ]
        if args.half:
            pm_cmd.append("--half")
        run(pm_cmd)

        epc_csv = csvdir_m / "epc.csv"
        epc_cmd = [
            sys.executable,
            "-m",
            "lossy_horizon.cli.eval_rd",
            "--codec",
            "epc",
            "--model",
            model_id,
            "--texts-file",
            args.texts,
            "--p-mask-list",
            *[str(p) for p in args.p_mask],
            "--rank-list",
            *[str(k) for k in args.rank_list],
            "--seeds",
            str(args.seeds),
            "--seed-base",
            str(args.seed_base),
            "--pos-coding",
            args.pos_coding,
            "--out-csv",
            str(epc_csv),
        ]
        if args.calib:
            calib_dir = outdir_m / "rank_calib"
            calib_dir.mkdir(exist_ok=True)
            epc_cmd += [
                "--rank-calib-file",
                args.calib,
                "--rank-calib-out",
                str(calib_dir),
            ]
        if args.half:
            epc_cmd.append("--half")
        run(epc_cmd)

        vqre_csv = csvdir_m / "vqre.csv"
        vq_cmd = [
            sys.executable,
            "-m",
            "lossy_horizon.cli.eval_vqre",
            "--model",
            model_id,
            "--texts-file",
            args.texts,
            "--p-mask-list",
            *[str(p) for p in args.p_mask],
            "--rank-list",
            *[str(k) for k in args.rank_list],
            "--anchors-every",
            *[str(a) for a in args.anchors_every],
            "--seeds",
            str(args.seeds),
            "--seed-base",
            str(args.seed_base),
            "--pos-coding",
            args.pos_coding,
            "--out-csv",
            str(vqre_csv),
        ]
        if args.half:
            vq_cmd.append("--half")
        run(vq_cmd)

        run(
            [
                sys.executable,
                "scripts/plot_rd.py",
                "--csv",
                str(pm_csv),
                "--codec",
                "pm",
                "--out",
                str(plotsdir_m / "pm.png"),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/plot_rd.py",
                "--csv",
                str(epc_csv),
                "--codec",
                "epc",
                "--out",
                str(plotsdir_m / "epc.png"),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/plot_rd.py",
                "--csv",
                str(vqre_csv),
                "--codec",
                "vqre",
                "--out",
                str(plotsdir_m / "vqre.png"),
            ]
        )

        manifest = {
            "timestamp": ts,
            "model": model_id,
            "texts": str(args.texts),
            "calib": str(args.calib) if args.calib else None,
            "p_mask": args.p_mask,
            "rank_list": args.rank_list,
            "anchors_every": args.anchors_every,
            "pos_coding": args.pos_coding,
            "seeds": args.seeds,
            "half": args.half,
            "env": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
                "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
            "outputs": {
                "pm_csv": str(pm_csv),
                "epc_csv": str(epc_csv),
                "vqre_csv": str(vqre_csv),
                "pm_plot": str(plotsdir_m / "pm.png"),
                "epc_plot": str(plotsdir_m / "epc.png"),
                "vqre_plot": str(plotsdir_m / "vqre.png"),
            },
        }
        with open(outdir_m / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        readme = outdir_m / "README.md"
        lines = []
        lines.append("# Paper Run\n\n")
        lines.append("## Environment\n")
        for k in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "HF_DATASETS_CACHE",
            "TOKENIZERS_PARALLELISM",
            "CUBLAS_WORKSPACE_CONFIG",
        ]:
            v = os.environ.get(k)
            if v:
                lines.append(f"export {k}={v}\n")
        lines.append("\n## Commands\n")

        def fmt(cmd):
            return " ".join(str(c) for c in cmd)

        lines.append(
            f"python scripts/bootstrap_cache.py --model {model_id} --hf_home {args.hf_home}\n"
        )
        lines.append(fmt(pm_cmd) + "\n")
        lines.append(fmt(epc_cmd) + "\n")
        lines.append(fmt(vq_cmd) + "\n")
        lines.append("\n## Plots\n")
        lines.append(
            f"python scripts/plot_rd.py --csv {pm_csv} --codec pm --out {plotsdir_m / 'pm.png'}\n"
        )
        lines.append(
            f"python scripts/plot_rd.py --csv {epc_csv} --codec epc --out {plotsdir_m / 'epc.png'}\n"
        )
        lines.append(
            f"python scripts/plot_rd.py --csv {vqre_csv} --codec vqre --out {plotsdir_m / 'vqre.png'}\n"
        )
        readme.write_text("".join(lines), encoding="utf-8")

    print(f"\nAll done. Results in: {outdir}")


if __name__ == "__main__":
    main()
