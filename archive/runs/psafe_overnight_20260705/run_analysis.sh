#!/usr/bin/env bash
#SBATCH --job-name=psafe-analyze
#SBATCH --partition=comp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

RUN=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705
ROOT="$RUN/code/real_ecology_cont_obser"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Per-penalty built-in aggregate + summary (defensive: never abort the batch).
for p in 2 5 10 20; do
  OUTP="$RUN/outputs/p$p"
  [[ -d "$OUTP/evaluation" ]] || { echo "skip p$p: no evaluation dir"; continue; }
  python -m real_ecology_benchmark.cli aggregate \
    --root "$OUTP/evaluation" --output "$OUTP/aggregate.json" || echo "aggregate p$p failed"
  python "$ROOT/scripts/summarize_real_outputs.py" "$OUTP" \
    --aggregate "$OUTP/aggregate.json" --output "$OUTP/summary.txt" || echo "summarize p$p failed"
done

# Cross-penalty decision table + convergence plots.
python "$RUN/analysis/analyze_psafe.py" "$RUN" || echo "analyze_psafe failed"

echo "=== analysis complete: $(date) ==="
ls -la "$RUN/analysis"
