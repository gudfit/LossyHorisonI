# Paper Run

## Environment
export HF_HOME=/home/ubuntu/LossyHorisonI/cache/hf
export TRANSFORMERS_CACHE=/home/ubuntu/LossyHorisonI/cache/hf/transformers
export HF_DATASETS_CACHE=/home/ubuntu/LossyHorisonI/cache/hf/datasets
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

## Commands
python scripts/bootstrap_cache.py --model roberta-base --hf_home cache/hf
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_rd --codec pm --model roberta-base --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/pm.csv --half --mask-batch-size 1024 --limit-texts 5
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_rd --codec epc --model roberta-base --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --rank-list 4 16 64 128 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/epc.csv --half --mask-batch-size 1024 --limit-texts 5
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_vqre --model roberta-base --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --rank-list 4 16 64 128 --anchors-every 0 16 32 64 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/vqre.csv --half --mask-batch-size 1024 --limit-texts 5

## Plots
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/pm.csv --codec pm --out /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/plots/pm.png
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/epc.csv --codec epc --out /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/plots/epc.png
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/csv/vqre.csv --codec vqre --out /home/ubuntu/LossyHorisonI/runs/paper-20251021-055524/roberta-base/plots/vqre.png
