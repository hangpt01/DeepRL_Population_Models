#!/usr/bin/env bash
# Freeze a code snapshot and generate the manifests for the motivation run.
#
# Jobs run from the SNAPSHOT, never from the live repo: a spooled Slurm array can
# start hours after submission, and a live tree that is being edited underneath it
# produces results nobody can reproduce.  (P_safe run lesson.)
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models
SNAP="$RUN/code"

if [[ -e "$SNAP" ]]; then
  echo "refusing to overwrite an existing snapshot: $SNAP" >&2
  echo "(a frozen snapshot is the run's provenance -- delete it deliberately if you mean to re-freeze)" >&2
  exit 1
fi

echo "== freezing snapshot -> $SNAP"
mkdir -p "$SNAP"
# Everything a row needs to run and everything that defines what it computed.
for path in src configs scripts real_ecology_data pyproject.toml Makefile; do
  cp -r "$REPO/$path" "$SNAP/"
done
mkdir -p "$SNAP/docs/benchmark"
cp "$REPO"/docs/benchmark/SERVER_{HANDOFF,CONTEXT,AUDIT,PLAN}_*.md "$SNAP/docs/benchmark/" 2>/dev/null || true
find "$SNAP" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "== recording provenance"
{
  echo "run:        motivation_native_20260711"
  echo "frozen_at:  $(date -Is)"
  echo "repo:       $REPO"
  echo "git_commit: $(cd "$REPO" && git rev-parse HEAD)"
  echo "git_status:"
  (cd "$REPO" && git status --short)
} > "$RUN/PROVENANCE.txt"

echo "== generating manifests from the SNAPSHOT (not the live tree)"
export PYTHONPATH="$SNAP/src"
python "$SNAP/scripts/make_motivation_native_manifest.py" \
  --output "$RUN/manifests/manifest.csv"
python "$SNAP/scripts/make_motivation_native_manifest.py" --natives-only \
  --output "$RUN/manifests/manifest_natives.csv"

main_rows=$(($(wc -l < "$RUN/manifests/manifest.csv") - 1))
native_rows=$(($(wc -l < "$RUN/manifests/manifest_natives.csv") - 1))
echo "   main manifest:    $main_rows rows  (expect 1440)"
echo "   natives manifest: $native_rows rows  (expect 576)"
[[ "$main_rows" -eq 1440 ]] || { echo "UNEXPECTED main manifest size" >&2; exit 1; }
[[ "$native_rows" -eq 576 ]] || { echo "UNEXPECTED natives manifest size" >&2; exit 1; }

# The 5 rows of a cell must be contiguous, or BS=5 packing would split a cell
# across tasks and two tasks would race on the same dataset lock.
python - "$RUN/manifests/manifest.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
key = lambda r: (r["reward_mode"], r["population"], r["environment"], r["sigma_obs"])
bad = [i for i in range(0, len(rows), 5)
       if len({key(r) for r in rows[i:i + 5]}) != 1]
if bad:
    raise SystemExit(f"cell blocks are not 5-row aligned at offsets {bad[:5]}")
print(f"   cell alignment: OK ({len(rows)//5} contiguous 5-row cells)")
PY

echo
echo "snapshot ready: $SNAP"
echo "next: sbatch the canary (array task 32 = Amur tiger/ricker/sigma=0/safe), then submit_blocks.sh"
