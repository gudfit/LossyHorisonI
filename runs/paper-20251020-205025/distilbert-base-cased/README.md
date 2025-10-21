# Paper Run

## Environment
export HF_HOME=/home/ubuntu/LossyHorisonI/cache/hf
export TRANSFORMERS_CACHE=/home/ubuntu/LossyHorisonI/cache/hf/transformers
export HF_DATASETS_CACHE=/home/ubuntu/LossyHorisonI/cache/hf/datasets
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

## Commands
python scripts/bootstrap_cache.py --model distilbert-base-cased --hf_home cache/hf
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_rd --codec pm --model distilbert-base-cased --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/pm.csv --half --mask-batch-size 1024 --limit-texts 10
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_rd --codec epc --model distilbert-base-cased --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --rank-list 4 16 64 128 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/epc.csv --half --mask-batch-size 1024 --limit-texts 10
/home/ubuntu/LossyHorisonI/.venv/bin/python3 -m lossy_horizon.cli.eval_vqre --model distilbert-base-cased --texts-file examples/wikitext103_test.txt --p-mask-list 0.2 0.4 0.6 0.8 --rank-list 4 16 64 128 --anchors-every 0 16 32 --seeds 5 --seed-base 0 --pos-coding min --out-csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/vqre.csv --half --mask-batch-size 1024 --limit-texts 10

## Plots
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/pm.csv --codec pm --out /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/plots/pm.png
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/epc.csv --codec epc --out /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/plots/epc.png
python scripts/plot_rd.py --csv /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/csv/vqre.csv --codec vqre --out /home/ubuntu/LossyHorisonI/runs/paper-20251020-205025/distilbert-base-cased/plots/vqre.png
