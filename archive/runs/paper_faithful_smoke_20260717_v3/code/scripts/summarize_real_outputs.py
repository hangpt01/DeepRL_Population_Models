#!/usr/bin/env python3
"""Write a compact text summary for a real-ecology experiment root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def _load_jsons(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _metric_deltas(summaries: list[dict[str, object]]) -> list[str]:
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in summaries:
        key = (
            row.get("compute_backend_effective", "unknown"),
            row.get("population"),
            row.get("environment"),
            row.get("sigma_obs"),
            row.get("model"),
            row.get("filter"),
        )
        grouped.setdefault(key, {})[str(row.get("reward_mode"))] = row
    metrics = ("persistence_mean", "collapse_entry_mean", "unsafe_fraction_mean", "economic_cost_mean")
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    for pair in grouped.values():
        if "safe" not in pair or "yield" not in pair:
            continue
        for metric in metrics:
            if metric in pair["safe"] and metric in pair["yield"]:
                values[metric].append(float(pair["safe"][metric]) - float(pair["yield"][metric]))
    lines = []
    for metric, deltas in values.items():
        if deltas:
            lines.append(f"safe_minus_yield_{metric}: {mean(deltas):.6g} over {len(deltas)} pairs")
    return lines


def _efficiency_lines(aggregate: dict[str, object]) -> list[str]:
    rows = aggregate.get("efficiency_mean", [])
    if not isinstance(rows, list) or not rows:
        return []
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        seconds = row.get("manifest_row_seconds", row.get("row_seconds"))
        rss = row.get("manifest_row_peak_rss_mb", row.get("peak_rss_mb"))
        planner = row.get("planner_seconds_mean")
        fit = row.get("fit_seconds")
        fields = [
            (
                f"{row.get('compute_backend_effective', 'unknown')}/"
                f"{row.get('reward_mode')}/{row.get('model')}/{row.get('filter')}"
            ),
            f"runs={row.get('runs')}",
        ]
        if seconds is not None:
            fields.append(f"row_s={float(seconds):.3g}")
        if planner is not None:
            fields.append(f"planner_s_ep={float(planner):.3g}")
        if fit is not None:
            fields.append(f"fit_s={float(fit):.3g}")
        if rss is not None:
            fields.append(f"peak_rss_mb={float(rss):.1f}")
        lines.append("  " + "  ".join(fields))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    aggregate_path = Path(args.aggregate) if args.aggregate else root / "aggregate.json"
    output_path = Path(args.output) if args.output else root / "summary.txt"
    summaries = _load_jsons(root / "evaluation")
    gates = _load_jsons(root / "gates")
    calibration = _load_jsons(root / "calibration")
    manifest_rows = {
        path.name: _csv_count(path)
        for path in sorted(root.glob("manifest*.csv"))
    }
    aggregate = {}
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)

    lines = [
        f"output_root: {root}",
        f"completed_summaries: {len(summaries)}",
        f"calibration_jsons: {len(calibration)}",
        f"gate_jsons: {len(gates)}",
        f"gate_passed: {sum(1 for row in gates if row.get('passed'))}/{len(gates)}",
        "",
        "manifest_rows:",
    ]
    lines.extend(f"  {name}: {count}" for name, count in manifest_rows.items())
    if aggregate:
        lines.extend([
            "",
            f"aggregate_runs: {aggregate.get('runs')}",
            f"beats_both_rate_rows: {len(aggregate.get('beats_both_rate', []))}",
        ])
        model_return = aggregate.get("model_return_mean", {})
        if isinstance(model_return, dict):
            lines.append("model_return_mean_backends: " + ", ".join(sorted(model_return)))
        efficiency = _efficiency_lines(aggregate)
        if efficiency:
            lines.extend(["", "efficiency_mean_by_backend_reward_model_filter:"])
            lines.extend(efficiency)
    delta_lines = _metric_deltas(summaries)
    if delta_lines:
        lines.extend(["", "reward_agnostic_safe_minus_yield:"])
        lines.extend("  " + line for line in delta_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
