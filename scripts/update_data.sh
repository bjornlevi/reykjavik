#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REYKJAVIK_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${REYKJAVIK_PYTHON:-$REPO_DIR/.venv/bin/python}"

cd "$REPO_DIR"

"$PYTHON_BIN" scripts/download_arsuppgjor.py
"$PYTHON_BIN" scripts/prepare_arsuppgjor.py
"$PYTHON_BIN" scripts/lookup_vm_entities.py
