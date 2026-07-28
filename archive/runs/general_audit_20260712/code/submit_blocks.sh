#!/usr/bin/env bash
# General-method search-budget audit -- release the sweep (rev-4 plan).
#
# CONCURRENCY CAPS ARE PROPORTIONAL TO COST, NOT EQUAL.  A per-array `%N` cap is a
# reservation, not a share: when the cheap arrays finish they CANNOT hand their CPUs to
# BA-MCTS.  Equal %80 caps would have made BA-MCTS (706 CPU-h) run ~10 h alone while
# RefPlan and OGSRL idled their cores.  Sized to cost, all three land within ~0.1 h:
#
#   array    rows   BS   tasks   CPU-h   cap    wall     --time
#   refplan  1584   11    144     194   %44    4.4 h    04:00:00
#   bamcts   1584    3    528     706   %158   ~5.4 h   12:00:00   <- worst task 5.43 h
#   ogsrl     288    2    144     169   %38    4.5 h    04:00:00
#   analysis                 1                          01:00:00
#   total    3456         817 tasks (MaxSubmit=1000; MaxArraySize=1001, max idx 527 OK)
#
# --time is sized against the WORST task, not the mean: a whole-cell BA-MCTS task on the
# slowest cell is ~9.3 h, which is why BS=3 (worst 5.43 h) with a 12 h wall, i.e. 2.2x margin.
# A mean-sized walltime silently kills the tail -- and the tail is the expensive corner we
# most need in order to claim "more search budget did not close the gap".
#
# Datasets are READ from the completed motivation run and never regenerated (G2 verified
# all 144 hashes match), so both sides provably saw identical offline data.
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$RUN/code"
CONFIG="$SNAP/configs/motivation_native.yaml"
MOTIV="$(cd "$RUN/../motivation_native_20260711/outputs/main" && pwd)"

[[ -d "$SNAP" ]] || { echo "no snapshot" >&2; exit 1; }
[[ -f "$RUN/gates/G1_no_op_proof.json" ]] || { echo "G1 gate artifact missing" >&2; exit 1; }
[[ -f "$RUN/gates/G2_dataset_identity.json" ]] || { echo "G2 gate artifact missing" >&2; exit 1; }
python3 - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
for gate in ("G1_no_op_proof", "G2_dataset_identity"):
    data = json.loads((run / "gates" / f"{gate}.json").read_text())
    if not data.get("pass"):
        raise SystemExit(f"{gate} did NOT pass -- refusing to submit")
print("gates G1 + G2 verified PASS")
PY

cd "$RUN/logs"
JOBIDS=()
submit () {  # $1=method $2=BS $3=cap $4=time
  local rows nb jid
  rows=$(($(wc -l < "$RUN/manifests/manifest_$1.csv") - 1))
  nb=$(( (rows + $2 - 1) / $2 ))
  jid=$(sbatch --parsable --job-name="gaudit-$1" --time="$4" \
    --export=ALL,ROOT="$SNAP",CONFIG="$CONFIG",OUTPUT_ROOT="$RUN/outputs",DATASET_ROOT="$MOTIV",BS="$2" \
    --array="0-$((nb - 1))%$3" "$RUN/run_block.sh" "$RUN/manifests/manifest_$1.csv")
  echo "submitted $1 -> job $jid  ($rows rows, BS=$2, $nb tasks, cap %$3, time $4)"
  JOBIDS+=("$jid")
}

# Caps/BS/walltimes RECALIBRATED from the canary's measured per-config costs.
# The rev-4 model (cost ~ depth x sims / horizon x sequences) was wrong in BOTH
# directions: refplan 1.64x and bamcts 1.60x UNDER, ogsrl 4x OVER (its runtime is
# dominated by training, not rollout depth).  Measured total 1478 CPU-h, not 1069.
#   method  BS  tasks  cap   wall   worst task   --time
#   refplan 11   144   %51  6.13h     7.78h     10:00:00
#   bamcts   2   792  %182  6.18h     6.02h     08:00:00
#   ogsrl    8    36    %7  5.77h     1.90h     04:00:00
#   total       973 tasks (MaxSubmit 1000); max array idx 791 (MaxArraySize 1001)
submit refplan 11  51 10:00:00
submit bamcts   2 182 08:00:00
submit ogsrl    8   7 04:00:00

DEP=$(IFS=:; echo "${JOBIDS[*]}")
ANALYSIS=$(sbatch --parsable --dependency="afterany:$DEP" --job-name=gaudit-analysis \
  --export=ALL,ROOT="$SNAP",RUN="$RUN" "$RUN/run_analysis.sh")
echo "analysis (afterany) -> $ANALYSIS"
{ printf '%s\n' "${JOBIDS[@]}"; echo "analysis=$ANALYSIS"; } > "$RUN/job_ids.txt"
echo "SUBMITTED -> $RUN/job_ids.txt"
