#!/usr/bin/env bash
#SBATCH --job-name=motiv-analysis
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
import json
from pathlib import Path
import sys

import numpy as np

from real_ecology_benchmark.manifest import aggregate_summaries

run = Path(sys.argv[1])
analysis = run / "analysis"
analysis.mkdir(parents=True, exist_ok=True)

# THE headline table.  The default (same-filter) path returns an EMPTY comparison
# here: the general methods run at filter=learned and the native baselines at
# filter=native_discrete, and the legacy scenario key includes `filter`, so no
# challenger would ever be paired with a baseline.  Pass the roles explicitly.
BASELINES = (("plus_native", "native_discrete"), ("moor_native", "native_discrete"))

result = aggregate_summaries(
    run / "outputs" / "main",
    analysis / "aggregate_main.json",
    challenger_filter="learned",
    baselines=BASELINES,
)
cells = result["beats_both_cells"]
print(f"main grid: {result['runs']} summaries, {len(cells)} paired cells")
if not cells:
    raise SystemExit("FATAL: headline comparison is empty -- cross-filter pairing failed")

for bins in (31, 91):
    root = run / "outputs" / f"res_b{bins}"
    if not root.exists():
        continue
    res = aggregate_summaries(root, analysis / f"aggregate_res_b{bins}.json")
    print(f"resolution b{bins}: {res['runs']} summaries")

# Acceptance 7.6: a coarser or finer native grid must not flip the headline.  The
# headline is the SIGN of (general - best native) per cell, so recompute that sign
# with the native returns taken from each grid and check it is stable.
def returns_by_cell(path, models=None):
    if not Path(path).exists():
        return {}
    with Path(path).open() as handle:
        data = json.load(handle)
    out = {}
    for row in data["summaries"]:
        model = row.get("model")
        if models and model not in models:
            continue
        key = (row.get("reward_mode"), row.get("population"),
               row.get("environment"), row.get("sigma_obs"))
        out.setdefault(key, {})[model] = row.get("operational_return_mean")
    return out


NATIVE_MODELS = {"plus_native", "moor_native"}
GENERAL_MODELS = {"refplan", "bamcts", "ogsrl"}

main_cells = returns_by_cell(analysis / "aggregate_main.json")
grids = {
    bins: returns_by_cell(analysis / f"aggregate_res_b{bins}.json", NATIVE_MODELS)
    for bins in (31, 91)
}

# A "flip" is a REVERSAL of the headline -- a cell where the general method wins at
# one native grid and LOSES at another.  An exact tie (diff == 0) is not a reversal:
# at sigma=0 the environment is deterministic, so a general method and a native that
# happen to find the same policy produce bit-identical returns.  Counting a tie as a
# third "sign" (np.sign -> 0) reports phantom flips.
flips, ties, compared = [], 0, 0
for key, models in main_cells.items():
    general = {m: v for m, v in models.items() if m in GENERAL_MODELS and v is not None}
    if not general:
        continue
    best_general = max(general.values())
    diffs = {}
    for label, natives in (
        ("main_b61", {m: v for m, v in models.items() if m in NATIVE_MODELS}),
        ("coarse_b31", grids[31].get(key, {})),
        ("fine_b91", grids[91].get(key, {})),
    ):
        values = [v for v in natives.values() if v is not None]
        if values:
            # Beats-both: the general method must beat the BEST native baseline.
            diffs[label] = best_general - max(values)
    if len(diffs) < 2:
        continue
    compared += 1
    values = list(diffs.values())
    if max(values) > 0.0 > min(values):          # wins somewhere, loses somewhere
        flips.append({"cell": list(key), "diffs": {k: round(v, 6) for k, v in diffs.items()}})
    elif any(v == 0.0 for v in values):          # exact tie at some grid; not a reversal
        ties += 1

verdict = {
    "cells_compared": compared,
    "cells_where_headline_flips": len(flips),
    "cells_with_an_exact_tie": ties,
    "pass": not flips,
    "flips": flips[:25],
}
with (analysis / "resolution_check.json").open("w") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)
status = "PASS" if verdict["pass"] else "FAIL"
print(f"resolution check [{status}]: {len(flips)}/{compared} cells flip the headline")

print(f"analysis written -> {analysis}")
PY
