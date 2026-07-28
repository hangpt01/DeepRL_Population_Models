#!/usr/bin/env python3
"""Diagnostic audit: are the general methods under-tuned, or genuinely out-modelled?

The motivation run found native ecological baselines beating general offline MBRL in
99.5% of cells.  Two very different explanations survive that result:

  (A) the general methods are UNDER-TUNED  -> the headline is about planner budget;
  (B) the natives are NEAR-ORACLE (they read exact per-action r and K from public
      tables while the generals must learn dynamics) -> the headline is about the
      environment not instantiating the uncertainty the paper claims.

These lead to different papers, so they must be separated on evidence.  This script
runs one row per (cell, method, horizon, pessimism) and, separately, measures the
learned model's one-step error against the native mechanistic model on held-out data.

Run one row:   audit_general_methods.py row  <index> --output-root DIR
Model quality: audit_general_methods.py model --output-root DIR
Report:        audit_general_methods.py report --output-root DIR
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.beliefs import LearnedLinearProposal
from real_ecology_benchmark.config import load_config, real_environment_like
from real_ecology_benchmark.native_solver import predict_ricker_next
from real_ecology_benchmark.pipeline import ensure_dataset, run_method

# Three cells spanning the axes that matter: the reference Ricker cell, the cell where
# the mechanistic forms disagree most (native strongest), and a noisy regime cell.
CELLS = (
    ("Amur tiger", "ricker", 0.2, "safe"),
    ("Iberian lynx", "allee", 0.2, "safe"),
    ("Puerto Rican parrot", "regime", 0.4, "safe"),
)
# As-run: horizon=5, pessimism=0.5.  Sweep both -- if the gap closes, it is tuning.
HORIZONS = (5, 10, 20)
PESSIMISMS = (0.5, 0.0)
METHODS = ("refplan", "bamcts")

GRID = [
    (cell, method, horizon, pessimism)
    for cell in CELLS
    for method in METHODS
    for horizon in HORIZONS
    for pessimism in PESSIMISMS
]


def slug(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def cell_config(cell, output_root: Path):
    population, family, sigma, reward_mode = cell
    cfg = load_config(str(ROOT / "configs" / "motivation_native.yaml"))
    cfg.environment = real_environment_like(cfg.environment, population, family)
    cfg.environment = replace(
        cfg.environment, observation_noise_sigma=sigma, reward_mode=reward_mode
    )
    cfg.validate()
    key = f"{slug(population)}/{family}/sigma_{slug(sigma)}"
    cfg.dataset.output = str(output_root / "datasets" / key / "public.npz")
    cfg.dataset.private_output = str(output_root / "private" / key / "truth.npz")
    return cfg, key


def cmd_row(args):
    cell, method, horizon, pessimism = GRID[args.index]
    output_root = Path(args.output_root)
    cfg, key = cell_config(cell, output_root)
    cfg.planner = replace(cfg.planner, horizon=horizon, pessimism=pessimism)
    cfg.validate()
    tag = f"{method}_h{horizon}_p{slug(pessimism)}"
    cfg.evaluation.output_dir = str(output_root / "evaluation" / key / tag)
    summary = run_method(method, cfg, "learned", regenerate=False)
    record = {
        "index": args.index, "population": cell[0], "family": cell[1],
        "sigma": cell[2], "reward_mode": cell[3], "method": method,
        "horizon": horizon, "pessimism": pessimism,
        "operational_return_mean": summary["operational_return_mean"],
        "collapse_entry_mean": summary["collapse_entry_mean"],
        "fallback_count_mean": summary["fallback_count_mean"],
        "row_seconds": summary.get("row_seconds"),
    }
    target = output_root / "rows" / f"{args.index:03d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(json.dumps(record, indent=2, sort_keys=True))


def cmd_model(args):
    """Quantify the informational asymmetry directly.

    One-step held-out prediction error of (a) the LEARNED dynamics model the general
    methods use, vs (b) the NATIVE mechanistic model, which reads r from the public
    action table and K from species.csv.  If (b) is far more accurate, the natives are
    not 'naive' -- they are near-oracle, and explanation (B) is the right one.
    """

    output_root = Path(args.output_root)
    out = []
    for cell in CELLS:
        cfg, key = cell_config(cell, output_root)
        dataset = ensure_dataset(cfg, regenerate=False)
        train, holdout = dataset.split_by_episode(0.8, seed=116)

        proposal = LearnedLinearProposal.fit(train, cfg.environment, ridge=cfg.model.ridge)
        rng = np.random.default_rng(0)
        # Learned model: mean one-step prediction from the observed state.
        learned, _ = proposal.sample_next(
            holdout.observations,
            holdout.actions,
            np.zeros((len(holdout), 1)),
            np.zeros(len(holdout), dtype=int),
            rng,
            holdout.rho if holdout.rho is not None else None,
            holdout.kappa if holdout.kappa is not None else None,
        )
        # Native mechanistic model: reads exact per-action r and K from the public tables.
        native = predict_ricker_next(
            cfg.environment,
            holdout.observations,
            holdout.actions,
            holdout.kappa if holdout.kappa is not None else 0.0,
        )
        truth = holdout.next_observations
        scale = max(float(cfg.environment.K_ref), 1.0)

        def err(pred):
            p = np.log1p(np.maximum(np.asarray(pred, dtype=float), 0.0) / scale)
            t = np.log1p(np.maximum(truth, 0.0) / scale)
            return float(np.sqrt(np.mean((p - t) ** 2)))

        row = {
            "population": cell[0], "family": cell[1], "sigma": cell[2],
            "holdout_transitions": int(len(holdout)),
            "learned_model_log_rmse": err(learned),
            "native_model_log_rmse": err(native),
        }
        row["accuracy_ratio_learned_over_native"] = (
            row["learned_model_log_rmse"] / max(row["native_model_log_rmse"], 1e-12)
        )
        out.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    target = output_root / "model_quality.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)


def cmd_report(args):
    output_root = Path(args.output_root)
    rows = [json.load(p.open()) for p in sorted((output_root / "rows").glob("*.json"))]
    if not rows:
        raise SystemExit("no rows yet")

    # The native return on the same cell, taken from the completed motivation run.
    native_ref = {}
    run = ROOT / "real_ecology_runs" / "motivation_native_20260711" / "outputs" / "main"
    for path in run.rglob("summary.json"):
        s = json.load(path.open())
        if s["model"] in ("plus_native", "moor_native") and s["reward_mode"] == "safe":
            key = (s["population"], s["environment"], s["sigma_obs"])
            native_ref[key] = max(native_ref.get(key, -1e9), s["operational_return_mean"])

    print("GENERAL-METHOD TUNING AUDIT")
    print("Does more planner budget close the gap to the native baselines?\n")
    for cell in CELLS:
        key = (cell[0], cell[1], cell[2])
        nat = native_ref.get(key)
        sub = [r for r in rows if (r["population"], r["family"], r["sigma"]) == key]
        if not sub:
            continue
        print(f"--- {cell[0]} / {cell[1]} / sigma={cell[2]} / {cell[3]}")
        print(f"    native baseline (best): {nat:.3f}" if nat else "    native baseline: n/a")
        print(f"    {'method':>8} {'horizon':>8} {'pessimism':>10} {'return':>8} {'gap to native':>14}")
        for r in sorted(sub, key=lambda r: (r["method"], r["horizon"], r["pessimism"])):
            gap = r["operational_return_mean"] - nat if nat else float("nan")
            star = "  <-- as run" if (r["horizon"] == 5 and r["pessimism"] == 0.5) else ""
            print(f"    {r['method']:>8} {r['horizon']:>8} {r['pessimism']:>10} "
                  f"{r['operational_return_mean']:>8.3f} {gap:>+14.3f}{star}")
        best = max(sub, key=lambda r: r["operational_return_mean"])
        if nat:
            print(f"    BEST tuned general: {best['operational_return_mean']:.3f} "
                  f"({best['method']} h={best['horizon']} p={best['pessimism']}) "
                  f"-> gap {best['operational_return_mean'] - nat:+.3f}")
        print()

    mq = output_root / "model_quality.json"
    if mq.exists():
        print("\nMODEL QUALITY (one-step held-out log-RMSE; lower is better)")
        print("The general methods LEARN this model; the natives READ r and K from public tables.\n")
        print(f"{'population':>20} {'family':>7} {'learned':>9} {'native':>9} {'x worse':>8}")
        for r in json.load(mq.open()):
            print(f"{r['population']:>20} {r['family']:>7} {r['learned_model_log_rmse']:>9.4f} "
                  f"{r['native_model_log_rmse']:>9.4f} {r['accuracy_ratio_learned_over_native']:>8.1f}x")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", default=str(ROOT / "outputs" / "audit_general"))
    p = sub.add_parser("row", parents=[common]); p.add_argument("index", type=int); p.set_defaults(func=cmd_row)
    p = sub.add_parser("model", parents=[common]); p.set_defaults(func=cmd_model)
    p = sub.add_parser("report", parents=[common]); p.set_defaults(func=cmd_report)
    p = sub.add_parser("count", parents=[common]); p.set_defaults(func=lambda a: print(len(GRID)))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
