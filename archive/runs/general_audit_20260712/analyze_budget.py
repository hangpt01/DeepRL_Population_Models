#!/usr/bin/env python3
"""Budget verdict, computed ONLY on cells with complete config coverage.

Why this exists rather than the inline analysis in run_analysis.sh: that version pooled
every cell that had *any* tuned row, which swept in 69 `yield` cells left over from the
first sweep (cancelled when the window shrank).  Those cells have PARTIAL config coverage,
so their "best tuned general" is a max over a truncated config set -- which understates the
tuned general and therefore biases the verdict TOWARD the "tuning doesn't help" conclusion
this run exists to test.  A cell is used only if all 24 configs are present.

Pre-registered reading (fixed before the numbers existed, plan section 7):
  tuned general closes the gap in >= 50% of cells -> SEARCH-BUDGET explanation, headline retractable
  otherwise                                       -> MODEL-LIMITED explanation, negative result stands
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
MOTIV = RUN.parent / "motivation_native_20260711" / "outputs" / "main"

GENERAL = ("refplan", "bamcts", "ogsrl")
NATIVE = ("plus_native", "moor_native")
NEED = {"refplan": 11, "bamcts": 11, "ogsrl": 2}   # new configs per method


def load(pattern):
    out = []
    for f in glob.glob(pattern, recursive=True):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass                                     # a truncated summary is not evidence
    return out


tuned = [r for r in load(str(RUN / "outputs/evaluation/**/summary.json")) if "config_tag" in r]
base = load(str(MOTIV / "evaluation/**/summary.json"))

cell_of = lambda r: (r["reward_mode"], r["population"], r["environment"], r["sigma_obs"])

# --- coverage gate: keep only cells with ALL 24 configs present -----------------
configs = defaultdict(set)
for r in tuned:
    configs[(cell_of(r), r["model"])].add(r["config_tag"])
touched = sorted({k[0] for k in configs})
complete = [c for c in touched
            if all(len(configs.get((c, m), set())) >= NEED[m] for m in NEED)]
dropped = [c for c in touched if c not in complete]

print(f"tuned rows            : {len(tuned)}")
print(f"cells touched         : {len(touched)}")
print(f"cells with FULL 24 cfg: {len(complete)}   <- used")
print(f"cells DROPPED (partial): {len(dropped)}   <- cancelled-sweep leftovers, excluded")
if dropped:
    modes = defaultdict(int)
    for c in dropped:
        modes[c[0]] += 1
    print(f"    dropped by reward_mode: {dict(modes)}")
print()

keep = set(complete)

as_run, native, best_tuned = defaultdict(dict), defaultdict(dict), defaultdict(dict)
for r in base:
    c = cell_of(r)
    if c not in keep:
        continue
    if r["model"] in GENERAL:
        as_run[c][r["model"]] = r["operational_return_mean"]
    elif r["model"] in NATIVE:
        native[c][r["model"]] = r["operational_return_mean"]
for r in tuned:
    c, m, v = cell_of(r), r["model"], r["operational_return_mean"]
    if c in keep and (m not in best_tuned[c] or v > best_tuned[c][m]):
        best_tuned[c][m] = v

rows = []
for c in sorted(keep):
    if not native[c] or not as_run[c]:
        continue
    nat = max(native[c].values())
    run_ = max(as_run[c].values())
    # the tuned general gets the best of (as-run, every one of its 24 tuned configs)
    tun = max([run_] + list(best_tuned[c].values()))
    rows.append({
        "cell": list(c), "native_best": nat, "as_run_best": run_, "tuned_best": tun,
        "gap_as_run": run_ - nat, "gap_tuned": tun - nat,
        "budget_gain": tun - run_, "tuned_beats_native": tun > nat,
    })

n = len(rows)
closed = sum(r["tuned_beats_native"] for r in rows)
as_run_win = sum(r["gap_as_run"] > 0 for r in rows)
verdict = {
    "cells_used": n,
    "cells_dropped_partial_coverage": len(dropped),
    "as_run_general_beats_native": as_run_win,
    "as_run_rate": as_run_win / n if n else 0.0,
    "tuned_general_beats_native": closed,
    "tuned_rate": closed / n if n else 0.0,
    "mean_gap_as_run": float(np.mean([r["gap_as_run"] for r in rows])) if rows else 0.0,
    "mean_gap_tuned": float(np.mean([r["gap_tuned"] for r in rows])) if rows else 0.0,
    "mean_budget_gain": float(np.mean([r["budget_gain"] for r in rows])) if rows else 0.0,
    "max_budget_gain": float(np.max([r["budget_gain"] for r in rows])) if rows else 0.0,
    "reward_modes": sorted({r["cell"][0] for r in rows}),
}
verdict["reading"] = (
    "SEARCH-BUDGET explanation SUPPORTED: more budget closes the gap; the motivation "
    "run's negative result is about tuning and is retractable."
    if verdict["tuned_rate"] >= 0.5 else
    "MODEL-LIMITED explanation SUPPORTED: more search budget does NOT close the gap. "
    "The motivation run's negative result stands."
)

out = RUN / "analysis" / "budget_verdict_clean.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"verdict": verdict, "cells": rows}, indent=2, sort_keys=True))

print(f"  cells used                  : {n}  (reward modes: {verdict['reward_modes']})")
print(f"  as-run  general beats native: {as_run_win}/{n} = {verdict['as_run_rate']:.1%}")
print(f"  TUNED   general beats native: {closed}/{n} = {verdict['tuned_rate']:.1%}")
print(f"  mean gap  as-run            : {verdict['mean_gap_as_run']:+.3f}")
print(f"  mean gap  tuned (best of 24): {verdict['mean_gap_tuned']:+.3f}")
print(f"  mean gain from extra budget : {verdict['mean_budget_gain']:+.3f}")
print(f"  max  gain from extra budget : {verdict['max_budget_gain']:+.3f}")
print()
print(f"  READING: {verdict['reading']}")
print(f"\nwritten -> {out}")
