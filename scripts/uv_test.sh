#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
	uv venv .venv
fi
source .venv/bin/activate
uv pip install -e '.[test]'
pytest -q "$@"
