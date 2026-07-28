#!/usr/bin/env bash
# Block-packed submission (stays under MaxSubmitJobs=1000). Each array task runs
# BS manifest rows. ~409 tasks total for BS=16.
set -euo pipefail

RUN=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705
SNAP=$RUN/code/real_ecology_cont_obser
M=$RUN/manifests
BLK=$RUN/run_block.sh
BS=16
CONC=48
cd "$RUN/logs"

nblocks () { local rows; rows=$(($(wc -l < "$1") - 1)); echo $(( (rows + BS - 1) / BS )); }
JOBIDS=()
submit () {  # $1=config $2=output_root $3=manifest $4=tag
  local nb spec jid
  nb=$(nblocks "$3"); spec="0-$((nb - 1))"
  jid=$(sbatch --parsable --job-name="psafe-$4" \
    --export=ALL,ROOT="$SNAP",CONFIG="$1",OUTPUT_ROOT="$2",BS="$BS" \
    --array="$spec%$CONC" "$BLK" "$3")
  echo "submitted $4 -> job $jid  ($nb blocks, $3)"
  JOBIDS+=("$jid")
}

submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part000.csv" p10a
submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part001.csv" p10b
submit "$SNAP/configs/real_experiment_p10.yaml" "$RUN/outputs/p10" "$M/manifest_full_learned_part002.csv" p10c
for p in 2 5 20; do
  submit "$SNAP/configs/real_experiment_p$p.yaml" "$RUN/outputs/p$p" "$M/manifest_safe_only_part000.csv" "p${p}a"
  submit "$SNAP/configs/real_experiment_p$p.yaml" "$RUN/outputs/p$p" "$M/manifest_safe_only_part001.csv" "p${p}b"
done

DEP=$(IFS=:; echo "${JOBIDS[*]}")
ANALYSIS=$(sbatch --parsable --dependency="afterany:$DEP" "$RUN/run_analysis.sh")
echo "analysis job (afterany) -> $ANALYSIS"
{ printf '%s\n' "${JOBIDS[@]}"; echo "analysis=$ANALYSIS"; } > "$RUN/job_ids.txt"
echo "ALL SUBMITTED -> $RUN/job_ids.txt"
