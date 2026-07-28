#!/usr/bin/env bash
#SBATCH --job-name=gaudit-analysis
#SBATCH --partition=comp
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

ROOT="${ROOT:?set ROOT}"
RUN="${RUN:?set RUN}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - "$RUN" <<'PY'
"""Does MORE SEARCH BUDGET close the general-vs-native gap?

Pre-registered reading (plan §7), decided before the numbers existed:
  * if the BEST-over-configs general method closes the gap in a material fraction of
    cells -> the motivation run's negative result is about SEARCH BUDGET, and the
    headline is retractable;
  * if it does not -> the general methods are MODEL-limited, the negative result stands,
    and the environment hands the natives an exact model.
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
motiv = run.parent / "motivation_native_20260711" / "outputs" / "main"
analysis = run / "analysis"
analysis.mkdir(parents=True, exist_ok=True)

def load(pattern):
    out = []
    for f in glob.glob(pattern, recursive=True):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out

tuned = load(str(run / "outputs" / "evaluation" / "**" / "summary.json"))
base = load(str(motiv / "evaluation" / "**" / "summary.json"))

print(f"tuned rows : {len(tuned)}  (expect 3456)")
if len(tuned) != 3456:
    # The overwrite detector: a short count means two configs collided on one path.
    print("*** WARNING: row count != 3456 -- possible config_tag collision/overwrite ***")

CELL = lambda s: (s["reward_mode"], s["population"], s["environment"], s["sigma_obs"])
GENERAL, NATIVE = {"refplan", "bamcts", "ogsrl"}, {"plus_native", "moor_native"}

# as-run general + native baselines from the completed motivation run
as_run, native = defaultdict(dict), defaultdict(dict)
for s in base:
    if s["model"] in GENERAL:
        as_run[CELL(s)][s["model"]] = s["operational_return_mean"]
    elif s["model"] in NATIVE:
        native[CELL(s)][s["model"]] = s["operational_return_mean"]

# best-over-configs tuned general, per cell
best_tuned = defaultdict(dict)
for s in tuned:
    cell, m = CELL(s), s["model"]
    v = s["operational_return_mean"]
    if m not in best_tuned[cell] or v > best_tuned[cell][m]:
        best_tuned[cell][m] = v

rows, closed, total = [], 0, 0
for cell, nat in native.items():
    if cell not in best_tuned or cell not in as_run:
        continue
    nat_best = max(nat.values())
    run_best = max(as_run[cell].values())
    # the tuned general gets the best of (as-run, every tuned config)
    tuned_best = max([run_best] + list(best_tuned[cell].values()))
    total += 1
    beat = tuned_best > nat_best
    closed += beat
    rows.append({
        "cell": list(cell), "native_best": nat_best, "as_run_best": run_best,
        "tuned_best": tuned_best, "gap_as_run": run_best - nat_best,
        "gap_tuned": tuned_best - nat_best, "tuned_beats_native": bool(beat),
        "budget_gain": tuned_best - run_best,
    })

verdict = {
    "cells": total,
    "tuned_general_beats_native": closed,
    "rate": closed / total if total else 0.0,
    "as_run_rate": sum(r["gap_as_run"] > 0 for r in rows) / total if total else 0.0,
    "mean_gap_as_run": float(np.mean([r["gap_as_run"] for r in rows])) if rows else 0.0,
    "mean_gap_tuned": float(np.mean([r["gap_tuned"] for r in rows])) if rows else 0.0,
    "mean_budget_gain": float(np.mean([r["budget_gain"] for r in rows])) if rows else 0.0,
    "reading": None,
}
verdict["reading"] = (
    "SEARCH-BUDGET explanation SUPPORTED -- more budget closes the gap; the negative "
    "result is retractable"
    if verdict["rate"] >= 0.5 else
    "MODEL-LIMITED explanation SUPPORTED -- more search budget does NOT close the gap; "
    "the motivation run's negative result stands"
)
with (analysis / "budget_verdict.json").open("w") as h:
    json.dump({"verdict": verdict, "cells": rows}, h, indent=2, sort_keys=True)

print()
print(f"  cells compared              : {total}")
print(f"  as-run general beats native : {verdict['as_run_rate']:.1%}")
print(f"  TUNED  general beats native : {verdict['rate']:.1%}")
print(f"  mean gap  (as-run)          : {verdict['mean_gap_as_run']:+.3f}")
print(f"  mean gap  (tuned best)      : {verdict['mean_gap_tuned']:+.3f}")
print(f"  mean gain from extra budget : {verdict['mean_budget_gain']:+.3f}")
print()
print(f"  READING: {verdict['reading']}")
print(f"\nwritten -> {analysis / 'budget_verdict.json'}")
PY
