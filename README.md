# The Lossy Horizon: Error‑Bounded Predictive Coding for Lossy Text Compression (Episode I)

This repo contains the reference implementation used for the paper “The Lossy Horizon: Error‑Bounded Predictive Coding for Lossy Text Compression (Episode I)”. It provides three lossy codecs built around a masked language model (MLM) decompressor and tooling to reproduce rate–distortion (RD) curves:

- Predictive Masking (PM)
- Error‑Bounded Predictive Coding (EPC)
- Vector Quantisation with Residual Patching (VQ+RE)

A PDF of the paper is in `paper/Root.pdf`.

## What’s Inside

- `src/lossy_horizon/`
  - `cli/` – runnable commands:
    - `compress.py` – compress a single text with PM/EPC
    - `eval_rd.py` – sweep RD curves for PM/EPC, export CSV
    - `eval_vqre.py` – sweep RD for VQ+RE
    - `train_vq.py` – optional light VQ fine‑tuning (KV quantisation)
  - `codecs/` – the three codecs (`pm.py`, `epc.py`, `vqre.py`, `vqre_codec.py`)
  - `bits/` – entropy coding utils: rANS approximations, n‑gram AR coder, positional bit cost, rank PMF/model
  - `models/` – `MLMScorer` wrapper and VQ hooks for BERT/RoBERTa/DistilBERT
  - `policies/` – surprisal‑driven masking with windowed equalisation and run caps
  - `metrics/` – CharFidelity, ChrF, BERTScore
  - `utils/` – reproducibility and lightweight logging helpers
- `scripts/`
  - `bootstrap_cache.py` – download HF models/datasets, optionally emit a text file
  - `plot_rd.py` – small RD plotter for the CSV outputs
- `examples/` – tiny text sample for quick smoke tests
- `paper.py` – one‑button script to run the full PM/EPC/VQ+RE sweep and produce plots
- `tests/` – unit tests for bits, policies, EPC, VQ hooks, and refinement

## Installation

Requirements: Python 3.10+, PyTorch, Hugging Face `transformers`.

Option A (uv):

- `./scripts/uv_cli_setup.sh`

Option B (pip):

- `python -m venv .venv && source .venv/bin/activate`
- `pip install -e '.[cli]'`

Optional metrics for ChrF/BERTScore (already included in extras): `sacrebleu`, `bert-score`.

Tip: Set HF caches to avoid re‑downloading models repeatedly:

- `export HF_HOME=$(pwd)/cache/hf`
- `export TRANSFORMERS_CACHE=$HF_HOME/transformers`
- `export HF_DATASETS_CACHE=$HF_HOME/datasets`

## Quick Start

Compress a single text with PM or EPC:

- `python -m lossy_horizon.cli.compress --model bert-base-cased --codec pm --p-mask 0.6 --text "The quick brown fox ..."`
- `python -m lossy_horizon.cli.compress --model bert-base-cased --codec epc --p-mask 0.6 --rank-threshold 16 --text "The quick brown fox ..."`

This prints a bit breakdown and the reconstruction. For EPC it reports both simple entropy estimates and practical rANS approximations for flags/ranks/fallback tokens.

## Reproducing RD Curves (PM/EPC)

- PM sweep: `python -m lossy_horizon.cli.eval_rd --codec pm --model roberta-base --texts-file examples/texts.txt --p-mask-list 0.2 0.4 0.6 0.8 --seeds 5 --out-csv runs/pm.csv`
- EPC sweep: `python -m lossy_horizon.cli.eval_rd --codec epc --model roberta-base --texts-file examples/texts.txt --p-mask-list 0.2 0.4 0.6 0.8 --rank-list 4 16 64 --seeds 5 --out-csv runs/epc.csv`

Options:

- `--pos-coding {min,enumerative,rle}` controls the positional stream; `min` chooses the cheaper of enumerative or RLE per sequence.
- `--half` enables fp16 on CUDA GPUs.
- `--mask-batch-size` tunes per‑position MLM batch size.
- `--rank-calib-file/--rank-calib-out` calibrates a rank PMF on disjoint text for EPC.

Plotting: `python scripts/plot_rd.py --csv runs/epc.csv --codec epc --out runs/epc.png`

## VQ+RE Baseline

Evaluate transform‑coding with residual patching:

- `python -m lossy_horizon.cli.eval_vqre --model bert-base-cased --texts-file examples/texts.txt --p-mask-list 0.4 0.6 0.8 --rank-list 4 16 64 --anchors-every 0 16 32 --seeds 3 --out-csv runs/vqre.csv`

Notes:

- VQ hooks quantise attention K/V during inference; optional light fine‑tuning is available via `python -m lossy_horizon.cli.train_vq` if desired.
- `--refine-steps, --refine-topm` enable iterative refinement to denoise the decode without extra side‑info.

## One‑Button Paper Runner

`paper.py` orchestrates everything: cache bootstrap, PM/EPC/VQ+RE sweeps, and plots.

- Example: `python paper.py --models bert-base-cased roberta-base distilroberta-base --texts examples/texts.txt --p-mask 0.2 0.4 0.6 0.8 --rank-list 4 16 64 --anchors-every 0 16 32 --seeds 5`
- Outputs land under `runs/paper-YYYYMMDD-HHMMSS/<model>/` with CSVs, plots, and a manifest/README detailing commands and env.

## Key Ideas Mapped to Code

- Predictive Masking (PM): `codecs/pm.py` encodes kept‑token positions and their ids; decode infills masks with MLM `argmax`.
- Error‑Bounded Predictive Coding (EPC): `codecs/epc.py` adds a residual channel of rank‑indexed overrides and fallbacks; flags and ranks are entropy‑coded (see `bits/rans.py`, `bits/rank_model.py`).
- VQ+RE: `codecs/vqre.py` provides EMA codebooks for K/V quantisation with scheduled self‑feeding; `codecs/vqre_codec.py` encodes anchors plus a rank‑based residual stream.
- Masking policy: `policies/masking.py` selects low‑surprisal tokens per window with a run‑length cap to avoid long masked runs.
- Metrics: `metrics/char_fidelity.py` (character‑level fidelity), `metrics/chrf.py`, `metrics/bertscore.py`.

## Datasets and Text Inputs

- Pass your own newline‑separated texts via `--texts-file`.
- To fetch a quick dataset sample: `python scripts/bootstrap_cache.py --model roberta-base --dataset wikitext --dataset-config wikitext-2-raw-v1 --dataset-split test --out-texts examples/wikitext2_test.txt --max-examples 5000` and then point `--texts-file` to the generated path.

## Reproducibility

- Set seeds via `--seeds` and `--seed-base` (default deterministic settings are applied where possible). On CUDA, we set `CUBLAS_WORKSPACE_CONFIG` in `paper.py` for determinism.

## Running Tests

- `./scripts/uv_test.sh` (uv) or `pip install -e '.[test]' && pytest -q`

## Notes and Limitations

- Long sequences are handled with sliding windows under each model’s max position; batch sizes and fp16 can be adjusted to fit memory.
- ChrF/BERTScore are optional; when packages are missing, they are reported as NA.
- Entropy coding for the vocabulary fallback stream reports both an ideal adaptive AC lower bound and a practical rANS approximation; this gap does not affect relative RD trends.

## Citation

If you use this code, please cite the paper:

```
N. Aghanya, J. Li, K. Wang. "The Lossy Horizon: Error‑Bounded Predictive Coding for Lossy Text Compression (Episode I)", 2025.
```
