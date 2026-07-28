#!/usr/bin/env bash
# Aggregate both matched arms and fail loudly if either sweep is incomplete.
#SBATCH --job-name=hidden-rk-analysis
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

ROOT="${ROOT:?set ROOT to frozen snapshot}"
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
summaries = []
for path in (run / "outputs").rglob("summary.json"):
    row = json.loads(path.read_text())
    row["path"] = str(path)
    summaries.append(row)

counts = {
    regime: sum(row.get("expose_rk") == regime for row in summaries)
    for regime in ("full", "hidden")
}
completion = {
    "expected_per_arm": 1440,
    "summary_counts": counts,
    "complete": counts == {"full": 1440, "hidden": 1440},
}
(analysis / "completion.json").write_text(
    json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not completion["complete"]:
    raise SystemExit(f"incomplete sweep: {counts}")

headline = aggregate_summaries(
    run / "outputs",
    analysis / "aggregate_headline_vs_moor.json",
    challenger_filter="learned",
    baselines=(("moor_native", "native_discrete"),),
)
closed_world = aggregate_summaries(
    run / "outputs",
    analysis / "aggregate_context_vs_both_natives.json",
    challenger_filter="learned",
    baselines=(
        ("moor_native", "native_discrete"),
        ("plus_native", "native_discrete"),
    ),
)

def key(row):
    return (
        row.get("reward_mode"), row.get("population"), row.get("environment"),
        float(row.get("sigma_obs")), row.get("model"), row.get("filter"),
    )

by_regime = {
    regime: {key(row): row for row in summaries if row.get("expose_rk") == regime}
    for regime in ("full", "hidden")
}
if set(by_regime["full"]) != set(by_regime["hidden"]):
    raise SystemExit("full and hidden completed summaries are not cell/method matched")

paired = []
for item in sorted(by_regime["full"]):
    full = by_regime["full"][item]
    hidden = by_regime["hidden"][item]
    paired.append({
        "reward_mode": item[0], "population": item[1], "environment": item[2],
        "sigma_obs": item[3], "model": item[4], "filter": item[5],
        "full_return": full["operational_return_mean"],
        "hidden_return": hidden["operational_return_mean"],
        "hidden_minus_full": hidden["operational_return_mean"] - full["operational_return_mean"],
    })

groups = {}
for row in paired:
    group = (row["reward_mode"], row["model"])
    groups.setdefault(group, []).append(row["hidden_minus_full"])
paired_summary = [
    {
        "reward_mode": reward_mode,
        "model": model,
        "cells": len(values),
        "hidden_minus_full_mean": float(np.mean(values)),
        "hidden_minus_full_median": float(np.median(values)),
    }
    for (reward_mode, model), values in sorted(groups.items())
]

report = {
    "completion": completion,
    "headline_cell_rows": len(headline["beats_both_cells"]),
    "closed_world_context_cell_rows": len(closed_world["beats_both_cells"]),
    "paired_hidden_vs_full": paired_summary,
    "safe_mode_interpretation": (
        "Information-limited under the private safety objective when public safety channels "
        "are weak; this is not evidence that OGSRL intrinsically fails at safety."
    ),
    "historical_07_11_status": (
        "Historical motivation result only; this run is the current-semantics matched comparison."
    ),
}
(analysis / "comparison_summary.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(analysis / "paired_hidden_vs_full.json").write_text(
    json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

