#!/usr/bin/env bash
# Submit the motivation sweep: main grid + native resolution replicate + aggregation.
#
# Task accounting against MaxSubmitJobs=1000:
#   main grid      1440 rows / BS=5  = 288 tasks
#   resolution b31  576 rows / BS=2  = 288 tasks
#   resolution b91  576 rows / BS=2  = 288 tasks
#   analysis                          =   1 task
#                                     -----------
#                                       865 tasks   (under the 1000 cap)
#
# Concurrency is capped at 240 (of the 250-CPU quota), leaving headroom for the
# analysis job so it is not starved behind the sweep.
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$RUN/code"
M="$RUN/manifests"
CONC=240

[[ -d "$SNAP" ]] || { echo "no snapshot: run prepare_run.sh first" >&2; exit 1; }
cd "$RUN/logs"

JOBIDS=()
submit () {  # $1=manifest $2=config $3=output_root $4=block_size $5=tag $6=array_spec
  local rows nb jid
  rows=$(($(wc -l < "$1") - 1))
  nb=$(( (rows + $4 - 1) / $4 ))
  local spec="${6:-0-$((nb - 1))}"
  jid=$(sbatch --parsable --job-name="motiv-$5" \
    --export=ALL,ROOT="$SNAP",CONFIG="$2",OUTPUT_ROOT="$3",BS="$4" \
    --array="$spec%$CONC" "$RUN/run_block.sh" "$1")
  echo "submitted $5 -> job $jid  ($nb blocks of $4 rows, array=$spec)"
  JOBIDS+=("$jid")
}

# ---- main grid: 5 methods x 288 cells.  BS=5 keeps each cell's 5 rows in one task,
# ---- so its dataset is generated exactly once with no cross-task lock contention.
submit "$M/manifest.csv" "$SNAP/configs/motivation_native.yaml" \
       "$RUN/outputs/main" 5 main

# ---- resolution replicate (acceptance 7.6): natives only, coarse and fine grids.
# ---- Separate OUTPUT_ROOTs so they cannot clobber the main grid's native rows.
# ---- The general methods do not depend on native_state_bins, so they are not re-run.
for bins in 31 91; do
  submit "$M/manifest_natives.csv" "$SNAP/configs/motivation_native_b${bins}.yaml" \
         "$RUN/outputs/res_b${bins}" 2 "res${bins}"
done

DEP=$(IFS=:; echo "${JOBIDS[*]}")
ANALYSIS=$(sbatch --parsable --dependency="afterany:$DEP" \
  --export=ALL,ROOT="$SNAP",RUN="$RUN" "$RUN/run_analysis.sh")
echo "analysis (afterany) -> $ANALYSIS"

{ printf '%s\n' "${JOBIDS[@]}"; echo "analysis=$ANALYSIS"; } > "$RUN/job_ids.txt"
echo "ALL SUBMITTED -> $RUN/job_ids.txt"
