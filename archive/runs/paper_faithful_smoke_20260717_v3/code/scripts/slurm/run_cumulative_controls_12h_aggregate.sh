#!/usr/bin/env bash
#SBATCH --job-name=cumctrl-agg
#SBATCH --partition=comp
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/cumctrl_agg_%j.out
#SBATCH --error=logs/cumctrl_agg_%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/cumulative_controls_12h}"

mkdir -p "$ROOT/logs" "$OUTPUT_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m real_ecology_benchmark.cli aggregate \
  --root "$OUTPUT_ROOT/evaluation" \
  --output "$OUTPUT_ROOT/aggregate.json"

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/cumulative_controls_12h")
summaries = list((root / "evaluation").rglob("summary.json"))
gates = list((root / "gates").glob("*.json"))
payload = {
    "summary_count": len(summaries),
    "gate_count": len(gates),
    "aggregate_exists": (root / "aggregate.json").exists(),
}
with (root / "completion_snapshot.json").open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
