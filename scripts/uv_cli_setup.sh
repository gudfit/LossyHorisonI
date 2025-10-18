#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
	uv venv .venv
fi
source .venv/bin/activate
uv pip install -e '.[cli]'
echo "CLI ready. Examples:"
echo "  python -m lossy_horizon.cli.compress --help"
echo "  python -m lossy_horizon.cli.eval_rd --help"
