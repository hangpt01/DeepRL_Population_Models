#!/bin/bash
# Diagnostic-replay task. Usage: sbatch replay_task.sbatch <CELL> <SIDE>
# Mirrors the accepted P=10 arrays 58493916/58493918: partition m3h, account ce25,
# QOS m3h, 1 CPU/task. Pinned to the accepted run's CPU (xenon-8452Y / Sapphire Rapids)
# so BLAS kernel selection matches the hardware that produced the accepted returns.
#SBATCH --job-name=diag_replay
#SBATCH --partition=m3h
#SBATCH --account=ce25
#SBATCH --qos=m3h
#SBATCH --constraint=xenon-8452Y
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1/parity/slurm-%x-%A-%a-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CELL="$1"; SIDE="$2"
ROOT=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_diagnostic_replay_20260726_v1
cd "$ROOT/scripts"
echo "=== node provenance ==="
echo "host=$(hostname)"
lscpu | grep -E "Model name|Architecture|CPU family|^Model:|Flags" | grep -iE "model|arch|family" || true
python3 -c "import numpy;print('numpy',numpy.__version__)"
python3 -c "import numpy;numpy.show_config()" 2>/dev/null | grep -iE "openblas configuration|version" | head -3 || true
echo "=== run $CELL $SIDE ==="
srun python3 run_diagnostic_replay.py "$CELL" "$SIDE"
