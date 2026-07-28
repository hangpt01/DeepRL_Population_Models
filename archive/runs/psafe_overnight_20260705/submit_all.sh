#!/usr/bin/env bash
# Submit the full P_safe overnight grid: 4 collapse_penalty variants on CPU,
# against the frozen snapshot, then a dependency-chained analysis job.
#   p10 -> full learned grid (safe+yield), 3 shards
#   p2/p5/p20 -> safe-mode rows only (penalty is inert in yield), 2 shards each
set -euo pipefail

RUN=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705
SNAP=$RUN/code/real_ecology_cont_obser
M=$RUN/manifests
ROW=$SNAP/scripts/slurm/run_real_row.sh
CONC=32
cd "$RUN/logs"

JOBIDS=()
submit () {  # $1=config $2=output_root $3=manifest $4=arrayspec $5=tag
  local jid
  jid=$(sbatch --parsable \
    --job-name="psafe-$5" \
    --export=ALL,ROOT="$SNAP",CONFIG="$1",OUTPUT_ROOT="$2" \
    --array="$4%$CONC" --time=18:00:00 \
    "$ROW" "$3")
  echo "submitted $5 -> job $jid ($3)"
  JOBIDS+=("$jid")
}

# p10: full grid (both reward modes)
submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part000.csv" "0-999" p10a
submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part001.csv" "0-999" p10b
submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part002.csv" "0-591" p10c

# p2 / p5 / p20: safe-only rows
for p in 2 5 20; do
  submit "$SNAP/configs/real_experiment_p$p.yaml" "$RUN/outputs/p$p" "$M/manifest_safe_only_part000.csv" "0-999" "p${p}a"
  submit "$SNAP/configs/real_experiment_p$p.yaml" "$RUN/outputs/p$p" "$M/manifest_safe_only_part001.csv" "0-295" "p${p}b"
done

DEP=$(IFS=:; echo "${JOBIDS[*]}")
ANALYSIS=$(sbatch --parsable --dependency="afterany:$DEP" "$RUN/run_analysis.sh")
echo "analysis job (afterany:$DEP) -> $ANALYSIS"

{ printf '%s\n' "${JOBIDS[@]}"; echo "analysis=$ANALYSIS"; } > "$RUN/job_ids.txt"
echo "ALL SUBMITTED. Job ids in $RUN/job_ids.txt"
