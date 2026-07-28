#!/usr/bin/env bash
# Overnight watcher: polls the P_safe grid, logs status, exits when the analysis
# job is terminal (or on a failure spike / hard cap) so Claude is re-notified.
RUN=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705
ARRAYS=$(grep -v '^analysis=' "$RUN/job_ids.txt" | tr '\n' ',' | sed 's/,$//')
ANALYSIS=$(grep '^analysis=' "$RUN/job_ids.txt" | cut -d= -f2)
LOG="$RUN/monitor_status.txt"
POLL=300
MAX=$((4 * 3600))
start=$(date +%s)

term_state () { sacct -n -X -j "$1" --format=State 2>/dev/null | head -1 | tr -d ' '; }

while true; do
  now=$(date +%s); el=$(( (now - start) / 60 ))
  q=$(squeue -u "$USER" -h -r -o "%t" 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  fails=$(sacct -n -X -j "$ARRAYS" --format=State 2>/dev/null | grep -cE 'FAILED|TIMEOUT|OUT_OF')
  done_ct=$(sacct -n -X -j "$ARRAYS" --format=State 2>/dev/null | grep -c COMPLETED)
  astate=$(term_state "$ANALYSIS")
  {
    echo "[$(date '+%F %T')] elapsed=${el}m queue={ $q} array_completed=$done_ct array_failed=$fails analysis=$astate"
  } >> "$LOG"
  # exit conditions
  case "$astate" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
      echo "ANALYSIS TERMINAL: $astate (elapsed ${el}m, array_failed=$fails)"; break;;
  esac
  if (( fails > 60 )); then echo "FAILURE SPIKE: $fails failed tasks (elapsed ${el}m)"; break; fi
  if (( now - start > MAX )); then echo "HARD CAP reached (elapsed ${el}m, analysis=$astate)"; break; fi
  sleep "$POLL"
done

echo "=== watcher exit $(date '+%F %T') ==="
tail -3 "$LOG"
ls -la "$RUN/analysis" 2>/dev/null
