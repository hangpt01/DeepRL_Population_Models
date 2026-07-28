#!/usr/bin/env python3
"""Reproduce every table/number used in the empirical results report from the
committed run artifacts (outputs/aggregate.json, outputs/evaluation/*/summary.json,
outputs/calibration/*, outputs/gates/*).  Run from the repo root:
    PYTHONPATH=src python scripts/extract_report_tables.py
Convention: the report uses the LEARNED filter as primary; per-method x sigma
values are means over the six (family) cells at that sigma.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

METHODS = [
    "mopo", "refplan", "bamcts", "moor", "plus",
    "ensemble_value_disagreement_pessimism", "ogsrl",
]
SIGMAS = [0.0, 0.1, 0.2, 0.4]


def load_summaries():
    rows = []
    for p in glob.glob("outputs/evaluation/*/*/*/summary.json"):
        d = json.load(open(p))
        parts = Path(p).parts
        cell = parts[parts.index("evaluation") + 1]
        env, a, sig = cell.split("_")
        d.update(_cell=cell, _env=env, _act=int(a.rstrip("a")),
                 _sig=float(sig.replace("sigma", "")),
                 _method=parts[-3], _filter=parts[-2])
        rows.append(d)
    return rows


def mean_over_families(S, filt, field, method, sig):
    v = [s[field] for s in S if s["_filter"] == filt and s["_method"] == method and s["_sig"] == sig]
    return float(np.mean(v)) if v else float("nan")


def main():
    S = load_summaries()
    assert len(S) == 384, len(S)

    def table(field, filt="learned", fmt="{:6.2f}"):
        print(f"\n[{field}] filter={filt} (rows=method, cols=sigma {SIGMAS}, +all)")
        for m in METHODS:
            per = [mean_over_families(S, filt, field, m, s) for s in SIGMAS]
            allv = float(np.mean([s[field] for s in S if s["_filter"] == filt and s["_method"] == m]))
            print(f"  {m:8} " + " ".join(fmt.format(x) for x in per) + " | " + fmt.format(allv))

    table("operational_return_mean")
    table("true_return_mean")
    table("collapse_entry_mean", fmt="{:6.3f}")
    table("unsafe_fraction_mean", fmt="{:6.3f}")
    table("filter_rmse_mean", fmt="{:6.1f}")
    table("fallback_count_mean", fmt="{:7.3f}")

    print("\n[LEARNED-ONLY overall mean operational return — USE THIS FOR RANKING]")
    ranking = sorted(((m, float(np.mean([s["operational_return_mean"]
                     for s in S if s["_filter"] == "learned" and s["_method"] == m])))
                     for m in METHODS), key=lambda kv: -kv[1])
    for m, v in ranking:
        print(f"  {m:8} {v:.2f}")

    print("\n[filter-MIXED model_return_mean from aggregate.json — DO NOT rank with this]")
    A = json.load(open("outputs/aggregate.json"))
    print("  ", {k: round(v, 2) for k, v in A["model_return_mean"].items()})

    print("\n[filter ablation: learned vs raw (mean op_return over all cells)]")
    for m in METHODS:
        L = np.mean([s["operational_return_mean"] for s in S if s["_filter"] == "learned" and s["_method"] == m])
        R = np.mean([s["operational_return_mean"] for s in S if s["_filter"] == "raw" and s["_method"] == m])
        print(f"  {m:8} learned={L:6.2f} raw={R:6.2f} delta={L-R:+.2f}")

    print("\n[PLUS/MOOR fidelity: learned/raw/ricker]")
    for m in ("plus", "moor"):
        vals = {f: np.mean([s["operational_return_mean"] for s in S if s["_filter"] == f and s["_method"] == m])
                for f in ("learned", "raw", "ricker")}
        print(f"  {m}: " + " ".join(f"{f}={vals[f]:.2f}" for f in vals)
              + f"  (learned-ricker={vals['learned']-vals['ricker']:+.2f})")

    print("\n[beats-both rate, learned & raw]")
    for filt in ("learned", "raw"):
        rr = {(r["method"], r["sigma_obs"]): r for r in A["beats_both_rate"] if r["filter"] == filt}
        print(f"  filter={filt}")
        for m in [x for x in METHODS if x not in ("plus", "moor")]:
            print(f"    {m:8} " + " ".join(f"{rr.get((m, s), {}).get('rate', float('nan')):4.2f}" for s in SIGMAS))

    print("\n[operational vs true return: row-level max |diff| and post-averaging max]")
    diffs = []
    for p in glob.glob("outputs/evaluation/*/*/learned/episodes.csv"):
        for row in csv.DictReader(open(p)):
            diffs.append(abs(float(row["operational_return"]) - float(row["true_return"])))
    avg = [abs(mean_over_families(S, "learned", "operational_return_mean", m, s)
               - mean_over_families(S, "learned", "true_return_mean", m, s))
           for m in METHODS for s in SIGMAS]
    print(f"  row-level max |op-true| = {max(diffs):.4f}; method×sigma-averaged max = {max(avg):.4f}")

    print("\n[which methods improve monotonically-ish with sigma? (op_return sig0 -> sig0.4)]")
    for m in METHODS:
        v = [mean_over_families(S, "learned", "operational_return_mean", m, s) for s in SIGMAS]
        print(f"  {m:8} {v[0]:.2f} -> {v[-1]:.2f}  improves={v[-1] > v[0]}")

    print("\n[calibration rates (final, per family)]")
    for f in sorted(glob.glob("outputs/calibration/*.json")):
        if any(x in f for x in ("backup", "regenerated", "tally")):
            continue
        d = json.load(open(f)); cell = Path(f).stem
        if cell.endswith("sigma0.0"):
            print(f"  {cell[:-9]:12} rate={d['incident_collapse_rate_healthy_starts']:.4f} pass={d['collapse_band_pass']}")

    print("\n[gate-gap ranges by gate type (reward_gap / collapse_gap; all pass)]")
    groups = {"hard (allee/regime)": [], "diagnostic (theta)": []}
    for f in sorted(glob.glob("outputs/gates/*.json")):
        g = json.load(open(f))
        key = "hard (allee/regime)" if g.get("hard_gate") else "diagnostic (theta)"
        groups[key].append((g["reward_gap"], g["collapse_gap"], g["passed"]))
    for key, vals in groups.items():
        rg = [v[0] for v in vals]; cg = [v[1] for v in vals]
        print(f"  {key:22} reward_gap [{min(rg):.2f},{max(rg):.2f}] "
              f"collapse_gap [{min(cg):.2f},{max(cg):.2f}] all_pass={all(v[2] for v in vals)} (n={len(vals)})")

    print("\n[beats-both bootstrap intervals (learned), from aggregate.json]")
    rr = {(r["method"], r["sigma_obs"]): r for r in A["beats_both_rate"] if r["filter"] == "learned"}
    for m in [x for x in METHODS if x not in ("plus", "moor")]:
        cis = " ".join(f"{rr[(m, s)]['rate']:.2f}[{rr[(m, s)]['bootstrap_ci_low']:.2f},"
                       f"{rr[(m, s)]['bootstrap_ci_high']:.2f}]" for s in SIGMAS if (m, s) in rr)
        print(f"  {m:8} {cis}")


if __name__ == "__main__":
    main()
