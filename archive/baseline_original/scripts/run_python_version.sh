#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_python_version.sh [example]
# Example names: gouldian, potoroo, examples2states2actions, examples2states4actions, examples2states6actions, examples2states10actions

EXAMPLE="${1:-gouldian}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export EXAMPLE_MODULE="$EXAMPLE"
export UAMS_SEED="${UAMS_SEED:-123}"
export UAMS_PRECISION="${UAMS_PRECISION:-0.1}"
export UAMS_TIMEOUT="${UAMS_TIMEOUT:-3600}"

python "python_port/src/building hmMDP/main.py"
python "python_port/src/solving hmMDP/main.py"

OUT_DIR="comparison/python/${EXAMPLE}"
mkdir -p "$OUT_DIR"

python - <<'PY'
import importlib
import os
import shutil
import sys
from pathlib import Path

root = Path.cwd()
example = os.environ["EXAMPLE_MODULE"]
data_dir = root / "python_port" / "data"
sys.path.insert(0, str(data_dir))
cfg = importlib.import_module(example).CONFIG

out_dir = root / "comparison" / "python" / example
out_dir.mkdir(parents=True, exist_ok=True)
for key in ("file_mean_params", "file_pomdpx", "file_outpolicy"):
    p = root / cfg[key]
    if p.exists():
        shutil.copy2(p, out_dir / p.name)

print(f"Python artifacts saved in: {out_dir}")
PY
