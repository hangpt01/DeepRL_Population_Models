#!/usr/bin/env python3
"""Driver: load a replay root, validate schema, compute M1-M15, write report.

    python3 run_analysis.py --root /path/to/replay_root --out ./out
    python3 run_analysis.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

import metrics as M
from figures import (fig_headroom, fig_posterior_vs_switching,
                     fig_raw_vs_centred, fig_return_decomposition)
from loader import load_run, schema_report


def species_of(cell_id: str) -> str | None:
    for s in ("amur_tiger", "crab_eating_fox", "egyptian_vulture"):
        if s in cell_id:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="replay root containing logs/")
    ap.add_argument("--out", default="./analysis_out")
    ap.add_argument("--accepted-csv",
                    help="MATCHED_P10_144_METHOD_CELLS.csv for parity checks")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "test_metrics.py"]))

    os.makedirs(args.out, exist_ok=True)
    logs = load_run(args.root)
    if not logs:
        raise SystemExit(f"no logs found under {args.root}/logs/")

    rep = schema_report(logs)
    rep.to_csv(os.path.join(args.out, "schema_report.csv"), index=False)
    print(rep.to_string(index=False))

    accepted = {}
    if args.accepted_csv and os.path.exists(args.accepted_csv):
        acc = pd.read_csv(args.accepted_csv)
        for _, r in acc.iterrows():
            key = (str(r.get("cell_id", "")), str(r.get("method", "")).lower())
            accepted[key] = r.get("return_mean")

    results, rows = {}, []
    for (cell, method), log in sorted(logs.items()):
        res = M.compute_all(log, accepted_return_mean=accepted.get((cell, method)),
                            species=species_of(cell))
        results[f"{cell}__{method}"] = res
        flat = {"cell_id": cell, "method": method}
        for mname, mval in res.items():
            if not mval.get("available"):
                continue
            for k, v in mval.items():
                if k not in ("available", "reason") and not isinstance(v, str):
                    flat[f"{mname.split('_')[0]}.{k}"] = v
        rows.append(flat)

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "metrics_table.csv"), index=False)

    # figures where the inputs exist
    m5 = {c: r["M5_raw_vs_centred"] for c, r in results.items()
          if r.get("M5_raw_vs_centred", {}).get("available")}
    if m5:
        fig_raw_vs_centred(m5, os.path.join(args.out, "fig_raw_vs_centred.png"))
    m6 = {c.split("__")[-1]: r["M6_reward_decomposition"]
          for c, r in results.items()
          if r.get("M6_reward_decomposition", {}).get("available")}
    if m6:
        fig_return_decomposition(m6, os.path.join(
            args.out, "fig_return_decomposition.png"))
    for (cell, method), log in sorted(logs.items()):
        r = results[f"{cell}__{method}"]
        if r.get("M1_posterior_movement", {}).get("available"):
            fig_posterior_vs_switching(
                log, r["M1_posterior_movement"], r["M2_switch_fractions"],
                os.path.join(args.out, f"fig_posterior_{cell}__{method}.png"))

    print(f"\nwrote {args.out}/metrics.json, metrics_table.csv, figures")
    parity = [(c, r["M6_reward_decomposition"].get("parity_pass"))
              for c, r in results.items()
              if "parity_pass" in r.get("M6_reward_decomposition", {})]
    if parity:
        bad = [c for c, ok in parity if not ok]
        print("PARITY:", "all pass" if not bad else f"FAIL {bad}")


if __name__ == "__main__":
    main()
