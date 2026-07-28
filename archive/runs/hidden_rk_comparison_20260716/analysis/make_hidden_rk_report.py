#!/usr/bin/env python3
"""Generate paper tables and figures for the matched full/hidden r/K run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import tempfile

_mpl = Path(tempfile.gettempdir()) / f"hidden_rk_report_mpl_{os.getuid()}"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ANALYSIS = Path(__file__).resolve().parent
RUN = ANALYSIS.parent
FIGURES = ANALYSIS / "hidden_rk_report_figures"
TABLES = ANALYSIS / "hidden_rk_report_tables"
TRACES = ANALYSIS / "illustrative_trajectories"

METHODS = ("refplan", "bamcts", "ogsrl", "moor_native", "plus_native")
GENERAL = ("refplan", "bamcts", "ogsrl")
COMPARATORS = ("moor_native", "plus_native")
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = (0.0, 0.1, 0.2, 0.4)
REWARDS = ("safe", "yield")
REGIMES = ("full", "hidden")
DISPLAY = {
    "refplan": "RefPlan",
    "bamcts": "BA-MCTS",
    "ogsrl": "OGSRL",
    "moor_native": "MOOR-native",
    "plus_native": "PLUS-native",
    "full": "Full",
    "hidden": "Hidden",
    "safe": "Safe",
    "yield": "Yield",
    "ricker": "Ricker",
    "allee": "Allee",
    "theta": "Theta",
    "regime": "Regime",
}
COLORS = {
    "refplan": "#4C78A8",
    "bamcts": "#59A14F",
    "ogsrl": "#B279A2",
    "moor_native": "#F28E2B",
    "plus_native": "#E15759",
    "full": "#7C8DA6",
    "hidden": "#2A9D8F",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
        ("&", r"\&"), ("#", r"\#"), ("$", r"\$"),
    ):
        text = text.replace(old, new)
    return text


def fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def ranked_tex(values, digits: int = 3, signed: bool = False):
    """Format comparable values with best bold and second-best underlined."""
    distinct = sorted({float(value) for value in values}, reverse=True)
    best = distinct[0]
    second = distinct[1] if len(distinct) > 1 else None

    def render(value):
        text = fmt(float(value), digits=digits, signed=signed)
        if float(value) == best:
            return rf"\textbf{{{text}}}"
        if second is not None and float(value) == second:
            return rf"\underline{{{text}}}"
        return text

    return render


def save_figure(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(FIGURES / f"{stem}{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = ANALYSIS / name
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(name: str, caption: str, label: str, columns: str,
                headers: list[str], rows: list[list[str]], long: bool = False,
                font_size: str | None = None) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    if long:
        lines = [
            f"\\begin{{longtable}}{{{columns}}}",
            f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            "\\toprule", " & ".join(headers) + r" \\", "\\midrule", "\\endfirsthead",
            "\\toprule", " & ".join(headers) + r" \\", "\\midrule", "\\endhead",
        ]
        lines.extend(" & ".join(row) + r" \\" for row in rows)
        lines.extend(["\\bottomrule", "\\end{longtable}"])
    else:
        lines = [
            "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}",
            f"\\label{{{label}}}", f"\\begin{{tabular}}{{{columns}}}", "\\toprule",
            " & ".join(headers) + r" \\", "\\midrule",
        ]
        lines.extend(" & ".join(row) + r" \\" for row in rows)
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    if font_size is not None and not long:
        lines.insert(2, f"\\{font_size}")
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_data():
    summaries: dict[str, list[dict[str, object]]] = {}
    by_cell: dict[str, dict[tuple[str, str, str, float], dict[str, dict[str, object]]]] = {}
    for regime in REGIMES:
        rows = []
        for path in (RUN / "outputs" / regime).rglob("summary.json"):
            row = json.loads(path.read_text())
            row["_path"] = str(path)
            rows.append(row)
        if len(rows) != 1440:
            raise RuntimeError(f"expected 1440 {regime} summaries, found {len(rows)}")
        summaries[regime] = rows
        cells: dict[tuple[str, str, str, float], dict[str, dict[str, object]]] = defaultdict(dict)
        for row in rows:
            key = (
                str(row["reward_mode"]), str(row["population"]),
                str(row["environment"]), float(row["sigma_obs"]),
            )
            model = str(row["model"])
            if model in cells[key]:
                raise RuntimeError(f"duplicate {regime} row for {key} {model}")
            cells[key][model] = row
        if len(cells) != 288 or any(set(models) != set(METHODS) for models in cells.values()):
            raise RuntimeError(f"invalid cell/method coverage in {regime}")
        by_cell[regime] = dict(cells)
    if set(by_cell["full"]) != set(by_cell["hidden"]):
        missing_full = sorted(set(by_cell["hidden"]) - set(by_cell["full"]))
        missing_hidden = sorted(set(by_cell["full"]) - set(by_cell["hidden"]))
        raise RuntimeError(f"unmatched cells: full={missing_full}, hidden={missing_hidden}")

    recoverability = {}
    with (RUN / "manifests" / "manifest_hidden.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recoverability[row["population"]] = row["recoverable"].lower() == "true"
    return summaries, by_cell, recoverability


def selected_keys(by_cell, recoverability, reward=None, scope="all", family=None, sigma=None):
    keys = []
    for key in by_cell["full"]:
        mode, population, fam, noise = key
        if reward is not None and mode != reward:
            continue
        if scope == "recoverable" and not recoverability[population]:
            continue
        if scope == "sink" and recoverability[population]:
            continue
        if family is not None and fam != family:
            continue
        if sigma is not None and noise != sigma:
            continue
        keys.append(key)
    return sorted(keys)


def method_return(by_cell, regime, key, method):
    return float(by_cell[regime][key][method]["operational_return_mean"])


def best_general(by_cell, regime, key):
    return max(method_return(by_cell, regime, key, method) for method in GENERAL)


def general_winner(by_cell, regime, key):
    values = [method_return(by_cell, regime, key, method) for method in GENERAL]
    maximum = max(values)
    winners = [method for method, value in zip(GENERAL, values) if value == maximum]
    return winners[0], len(winners) > 1


def gap_stats(by_cell, regime, keys, comparator):
    general = np.asarray([best_general(by_cell, regime, key) for key in keys])
    native = np.asarray([method_return(by_cell, regime, key, comparator) for key in keys])
    gap = general - native
    return {
        "cells": len(keys),
        "best_general_mean": float(np.mean(general)),
        "native_mean": float(np.mean(native)),
        "gap_mean": float(np.mean(gap)),
        "gap_median": float(np.median(gap)),
        "gap_se": float(np.std(gap, ddof=1) / math.sqrt(len(gap))) if len(gap) > 1 else 0.0,
        "wins": int(np.sum(gap > 0)),
        "ties": int(np.sum(gap == 0)),
    }


def generate_tables(by_cell, recoverability, summaries):
    completion = json.loads((ANALYSIS / "completion.json").read_text())
    registration = json.loads((RUN / "manifests" / "registration.json").read_text())
    provenance = (RUN / "PROVENANCE.txt").read_text(encoding="utf-8")
    git_commit = next(
        line.split(":", 1)[1].strip()
        for line in provenance.splitlines()
        if line.startswith("git_commit:")
    )
    overshoot = {
        regime: sum(int(row["actual_rows"]) > int(row["target_rows"]) for row in summaries[regime])
        for regime in REGIMES
    }
    maximum_rows = {
        regime: max(int(row["actual_rows"]) for row in summaries[regime])
        for regime in REGIMES
    }
    manifest_hashes = {
        regime: sha256(RUN / "manifests" / f"manifest_{regime}.csv")
        for regime in REGIMES
    }
    config_hash = sha256(RUN / "code" / "configs" / "hidden_rk.yaml")
    completion_rows = [
        ["Regime", "Known parameter", "Hidden demographics"],
        ["Label", r"\texttt{expose\_rk=full}", r"\texttt{expose\_rk=hidden}"],
        ["Scientific role", "Parameter-known ablation / upper bound", "Corrected uncertainty test"],
        ["Manifest rows", "1,440", "1,440"],
        ["Completed summaries", str(completion["summary_counts"]["full"]), str(completion["summary_counts"]["hidden"])],
        ["Matched cells", "288", "288"],
        ["Target transitions", f"{registration['target_rows']:,} per cell", f"{registration['target_rows']:,} per cell"],
        ["Rows with overshoot", str(overshoot["full"]), str(overshoot["hidden"])],
        ["Maximum actual rows", f"{maximum_rows['full']:,}", f"{maximum_rows['hidden']:,}"],
        ["Slurm array", r"\texttt{58326885}", r"\texttt{58326886}"],
        ["Exit status", "288/288 completed, 0:0", "288/288 completed, 0:0"],
        ["Manifest SHA-256 prefix", rf"\texttt{{{manifest_hashes['full'][:12]}}}", rf"\texttt{{{manifest_hashes['hidden'][:12]}}}"],
        ["Frozen config SHA-256 prefix", rf"\texttt{{{config_hash[:12]}}}", rf"\texttt{{{config_hash[:12]}}}"],
        ["Frozen base commit", rf"\texttt{{{git_commit[:12]}}}", rf"\texttt{{{git_commit[:12]}}}"],
    ]
    write_table(
        "completion.tex", "Completion, provenance, and regime roles.", "tab:completion",
        "p{0.24\\linewidth}p{0.33\\linewidth}p{0.33\\linewidth}",
        completion_rows[0], completion_rows[1:],
    )

    paired_rows = []
    paired_csv = []
    for reward in REWARDS:
        keys = selected_keys(by_cell, recoverability, reward=reward)
        for method in METHODS:
            full = np.asarray([method_return(by_cell, "full", key, method) for key in keys])
            hidden = np.asarray([method_return(by_cell, "hidden", key, method) for key in keys])
            delta = hidden - full
            item = {
                "reward_mode": reward, "method": method, "cells": len(keys),
                "full_mean": float(np.mean(full)), "hidden_mean": float(np.mean(hidden)),
                "hidden_minus_full_mean": float(np.mean(delta)),
                "hidden_minus_full_median": float(np.median(delta)),
                "delta_se": float(np.std(delta, ddof=1) / math.sqrt(len(delta))),
                "hidden_wins": int(np.sum(delta > 0)),
            }
            paired_csv.append(item)
    for reward in REWARDS:
        records = [item for item in paired_csv if item["reward_mode"] == reward]
        full_rank = ranked_tex([item["full_mean"] for item in records])
        hidden_rank = ranked_tex([item["hidden_mean"] for item in records])
        for item in records:
            paired_rows.append([
                DISPLAY[reward], DISPLAY[item["method"]], str(item["cells"]),
                full_rank(item["full_mean"]), hidden_rank(item["hidden_mean"]),
                fmt(item["hidden_minus_full_mean"], signed=True),
                fmt(item["hidden_minus_full_median"], signed=True),
                f"{item['hidden_wins']}/{item['cells']}",
            ])
    write_csv("matched_method_returns.csv", paired_csv)
    write_table(
        "matched_method_returns.tex",
        "Matched-cell method returns and hidden-minus-full deltas (all populations).",
        "tab:matched-methods", "llrrrrrr",
        ["Reward", "Method", "Cells", "Full", "Hidden", r"$\Delta$ mean", r"$\Delta$ med.", "Hidden wins"],
        paired_rows, long=True,
    )

    native_csv = []
    native_rows = []
    for reward in REWARDS:
        keys = selected_keys(by_cell, recoverability, reward=reward)
        for regime in REGIMES:
            for comparator in COMPARATORS:
                stats = gap_stats(by_cell, regime, keys, comparator)
                item = {"reward_mode": reward, "regime": regime, "comparator": comparator, **stats}
                native_csv.append(item)
                rank = ranked_tex([stats["best_general_mean"], stats["native_mean"]])
                native_rows.append([
                    DISPLAY[reward], DISPLAY[regime], DISPLAY[comparator], str(stats["cells"]),
                    rank(stats["best_general_mean"]), rank(stats["native_mean"]),
                    fmt(stats["gap_mean"], signed=True), f"{stats['wins']}/{stats['cells']}",
                ])
    write_csv("native_comparison.csv", native_csv)
    write_table(
        "native_comparison.tex", "Best general method versus each native comparator.",
        "tab:native-comparison", "lllrrrrr",
        ["Reward", "Regime", "Comparator", "Cells", "Best gen.", "Native", "Gap", "Wins"],
        native_rows,
    )

    split_csv = []
    split_rows = []
    for reward in REWARDS:
        for scope in ("recoverable", "sink"):
            keys = selected_keys(by_cell, recoverability, reward=reward, scope=scope)
            for regime in REGIMES:
                moor = gap_stats(by_cell, regime, keys, "moor_native")
                plus = gap_stats(by_cell, regime, keys, "plus_native")
                item = {
                    "reward_mode": reward, "scope": scope, "regime": regime,
                    "cells": len(keys), "best_general_mean": moor["best_general_mean"],
                    "moor_mean": moor["native_mean"], "gap_vs_moor": moor["gap_mean"],
                    "wins_vs_moor": moor["wins"], "plus_mean": plus["native_mean"],
                    "gap_vs_plus": plus["gap_mean"], "wins_vs_plus": plus["wins"],
                }
                split_csv.append(item)
                split_rows.append([
                    DISPLAY[reward], scope.capitalize(), DISPLAY[regime], str(len(keys)),
                    fmt(item["best_general_mean"]), fmt(item["gap_vs_moor"], signed=True),
                    f"{item['wins_vs_moor']}/{len(keys)}", fmt(item["gap_vs_plus"], signed=True),
                    f"{item['wins_vs_plus']}/{len(keys)}",
                ])
    write_csv("recoverability_comparison.csv", split_csv)
    write_table(
        "recoverability_comparison.tex", "Recoverable-population and demographic-sink split.",
        "tab:recoverability", "lllrrrrrr",
        ["Reward", "Scope", "Regime", "Cells", "Best gen.", r"Gap MOOR", "Wins", r"Gap PLUS", "Wins"],
        split_rows, long=True,
    )

    family_csv = []
    family_rows = []
    for family in FAMILIES:
        for regime in REGIMES:
            keys = selected_keys(by_cell, recoverability, reward="safe", scope="recoverable", family=family)
            for comparator in COMPARATORS:
                stats = gap_stats(by_cell, regime, keys, comparator)
                item = {"family": family, "regime": regime, "comparator": comparator, **stats}
                family_csv.append(item)
                family_rows.append([
                    DISPLAY[family], DISPLAY[regime], DISPLAY[comparator], str(stats["cells"]),
                    fmt(stats["gap_mean"], signed=True), f"{stats['wins']}/{stats['cells']}",
                ])
    write_csv("family_comparison_safe_recoverable.csv", family_csv)
    write_table(
        "family_comparison.tex",
        "Safe-mode best-general minus native gaps by simulator family (recoverable populations).",
        "tab:family", "lllrrr", ["Family", "Regime", "Comparator", "Cells", "Mean gap", "Wins"],
        family_rows,
    )

    sigma_csv = []
    sigma_rows = []
    for sigma in SIGMAS:
        for regime in REGIMES:
            keys = selected_keys(by_cell, recoverability, reward="safe", scope="recoverable", sigma=sigma)
            for comparator in COMPARATORS:
                stats = gap_stats(by_cell, regime, keys, comparator)
                item = {"sigma_obs": sigma, "regime": regime, "comparator": comparator, **stats}
                sigma_csv.append(item)
                sigma_rows.append([
                    f"{sigma:.1f}", DISPLAY[regime], DISPLAY[comparator], str(stats["cells"]),
                    fmt(stats["gap_mean"], signed=True), f"{stats['wins']}/{stats['cells']}",
                ])
    write_csv("sigma_comparison_safe_recoverable.csv", sigma_csv)
    write_table(
        "sigma_comparison.tex",
        "Safe-mode best-general minus native gaps by observation noise (recoverable populations).",
        "tab:sigma", "lllrrr", [r"$\sigma_{\rm obs}$", "Regime", "Comparator", "Cells", "Mean gap", "Wins"],
        sigma_rows,
    )
    return {
        "paired": paired_csv, "native": native_csv, "split": split_csv,
        "family": family_csv, "sigma": sigma_csv,
    }


def generate_method_decomposition(by_cell, recoverability):
    composition = []

    def add_composition(regime, reward, breakdown, level, keys):
        counts = {method: 0 for method in GENERAL}
        ties = 0
        for key in keys:
            winner, tied = general_winner(by_cell, regime, key)
            counts[winner] += 1
            ties += int(tied)
        composition.append({
            "regime": regime,
            "reward_mode": reward,
            "breakdown": breakdown,
            "level": level,
            "cells": len(keys),
            **{f"{method}_wins": counts[method] for method in GENERAL},
            "tie_cells": ties,
        })

    for regime in REGIMES:
        for reward in REWARDS:
            add_composition(
                regime, reward, "overall", "all",
                selected_keys(by_cell, recoverability, reward=reward),
            )
            for family in FAMILIES:
                add_composition(
                    regime, reward, "family", family,
                    selected_keys(by_cell, recoverability, reward=reward, family=family),
                )
            for sigma in SIGMAS:
                add_composition(
                    regime, reward, "sigma", f"{sigma:.1f}",
                    selected_keys(by_cell, recoverability, reward=reward, sigma=sigma),
                )
            for scope in ("recoverable", "sink"):
                add_composition(
                    regime, reward, "scope", scope,
                    selected_keys(by_cell, recoverability, reward=reward, scope=scope),
                )
    if any(row["tie_cells"] for row in composition):
        raise RuntimeError("general winner composition contains an exact tie")
    write_csv("general_winner_composition.csv", composition)

    dimension_rows = {"family": [], "sigma": []}
    for dimension, levels in (("family", FAMILIES), ("sigma", SIGMAS)):
        for regime in REGIMES:
            for reward in REWARDS:
                for comparator in COMPARATORS:
                    for method in GENERAL:
                        for level in levels:
                            selector = {dimension: level}
                            keys = selected_keys(
                                by_cell, recoverability, reward=reward, **selector
                            )
                            gaps = np.asarray([
                                method_return(by_cell, regime, key, method)
                                - method_return(by_cell, regime, key, comparator)
                                for key in keys
                            ])
                            dimension_rows[dimension].append({
                                "dimension": dimension,
                                "level": f"{level:.1f}" if dimension == "sigma" else level,
                                "regime": regime,
                                "reward_mode": reward,
                                "comparator": comparator,
                                "method": method,
                                "cells": len(keys),
                                "mean_gap": float(np.mean(gaps)),
                                "median_gap": float(np.median(gaps)),
                                "wins": int(np.sum(gaps > 0)),
                            })
        write_csv(f"general_method_native_gaps_by_{dimension}.csv", dimension_rows[dimension])

    hidden_summary = []
    hidden_table_rows = []
    cell_gaps = []
    for reward in REWARDS:
        keys = selected_keys(by_cell, recoverability, reward=reward)
        winner_counts = {method: 0 for method in GENERAL}
        for key in keys:
            winner, tied = general_winner(by_cell, "hidden", key)
            if tied:
                raise RuntimeError("hidden general summary contains an exact tie")
            winner_counts[winner] += 1
        for method in GENERAL:
            returns = np.asarray([
                method_return(by_cell, "hidden", key, method) for key in keys
            ])
            gaps = {}
            for comparator in COMPARATORS:
                values = np.asarray([
                    method_return(by_cell, "hidden", key, method)
                    - method_return(by_cell, "hidden", key, comparator)
                    for key in keys
                ])
                gaps[comparator] = values
                for key, gap in zip(keys, values):
                    _, population, family, sigma = key
                    cell_gaps.append({
                        "reward_mode": reward,
                        "method": method,
                        "comparator": comparator,
                        "population": population,
                        "family": family,
                        "sigma_obs": sigma,
                        "recoverable": recoverability[population],
                        "gap": float(gap),
                    })
            item = {
                "reward_mode": reward,
                "method": method,
                "cells": len(keys),
                "mean_return": float(np.mean(returns)),
                "median_return": float(np.median(returns)),
                "general_wins": winner_counts[method],
                "wins_over_moor": int(np.sum(gaps["moor_native"] > 0)),
                "mean_margin_moor": float(np.mean(gaps["moor_native"])),
                "wins_over_plus": int(np.sum(gaps["plus_native"] > 0)),
                "mean_margin_plus": float(np.mean(gaps["plus_native"])),
            }
            hidden_summary.append(item)
    for reward in REWARDS:
        records = [item for item in hidden_summary if item["reward_mode"] == reward]
        ranks = {
            "mean_return": ranked_tex([item["mean_return"] for item in records]),
            "median_return": ranked_tex([item["median_return"] for item in records]),
            "general_wins": ranked_tex([item["general_wins"] for item in records], digits=0),
            "wins_over_moor": ranked_tex([item["wins_over_moor"] for item in records], digits=0),
            "mean_margin_moor": ranked_tex(
                [item["mean_margin_moor"] for item in records], signed=True
            ),
            "wins_over_plus": ranked_tex([item["wins_over_plus"] for item in records], digits=0),
            "mean_margin_plus": ranked_tex(
                [item["mean_margin_plus"] for item in records], signed=True
            ),
        }
        for item in records:
            cells = item["cells"]
            hidden_table_rows.append([
                DISPLAY[reward], DISPLAY[item["method"]],
                ranks["mean_return"](item["mean_return"]),
                ranks["median_return"](item["median_return"]),
                f"{ranks['general_wins'](item['general_wins'])}/{cells}",
                f"{ranks['wins_over_moor'](item['wins_over_moor'])}/{cells} / "
                f"{ranks['mean_margin_moor'](item['mean_margin_moor'])}",
                f"{ranks['wins_over_plus'](item['wins_over_plus'])}/{cells} / "
                f"{ranks['mean_margin_plus'](item['mean_margin_plus'])}",
            ])
    write_csv("hidden_general_method_summary.csv", hidden_summary)
    write_csv("hidden_general_native_cell_gaps.csv", cell_gaps)
    write_table(
        "hidden_general_method_summary.tex",
        "Hidden-regime decomposition of the three general methods (all populations).",
        "tab:hidden-general-methods", "llrrrrr",
        ["Reward", "Method", "Mean", "Median", "General wins",
         "MOOR wins / margin", "PLUS wins / margin"],
        hidden_table_rows, font_size="footnotesize",
    )
    return {
        "winner_composition": composition,
        "family_gaps": dimension_rows["family"],
        "sigma_gaps": dimension_rows["sigma"],
        "hidden_summary": hidden_summary,
        "hidden_cell_gaps": cell_gaps,
    }


def safety_diagnostics(by_cell, recoverability):
    representatives = []
    for key in selected_keys(by_cell, recoverability, reward="safe"):
        representatives.append(by_cell["hidden"][key]["refplan"])
    terminated = 0
    transitions = 0
    fallback = 0
    diagnostics = []
    family_counts = defaultdict(lambda: [0, 0])
    for row in representatives:
        public = row.get("public_surrogate_diagnostics", {})
        evaluator = row.get("evaluator_only_surrogate_diagnostics", {})
        fit_rows = int(public.get("fit_rows", 0))
        holdout_rows = int(public.get("holdout_rows", 0))
        term = int(public.get("terminated_fit", 0)) + int(public.get("terminated_holdout", 0))
        total = fit_rows + holdout_rows
        terminated += term
        transitions += total
        family_counts[str(row["environment"])][0] += term
        family_counts[str(row["environment"])][1] += total
        fallback += public.get("risk_fallback", "none") != "none"
        diagnostics.append((public, evaluator))

    def avg_public(name):
        vals = [float(p[name]) for p, _ in diagnostics if name in p and np.isfinite(float(p[name]))]
        return float(np.mean(vals)) if vals else float("nan")

    def pooled_eval(name, rows_name):
        vals = [(float(e[name]), int(e[rows_name])) for _, e in diagnostics if name in e and int(e.get(rows_name, 0)) > 0]
        return math.sqrt(sum(value * value * n for value, n in vals) / sum(n for _, n in vals)) if vals else float("nan")

    result = {
        "cells": len(representatives),
        "terminated_events": terminated,
        "transitions": transitions,
        "terminated_prevalence": terminated / transitions,
        "risk_fallback_cells": fallback,
        "risk_predicted_prevalence_mean": avg_public("risk_holdout_predicted_prevalence"),
        "risk_holdout_brier_mean": avg_public("risk_holdout_brier"),
        "risk_holdout_log_loss_mean": avg_public("risk_holdout_log_loss"),
        "reward_holdout_rmse_mean": avg_public("reward_holdout_rmse"),
        "reward_low_tail_rmse_mean": avg_public("reward_holdout_low_tail_rmse"),
        "evaluator_no_penalty_rmse_pooled": pooled_eval("evaluator_reward_safe_rmse", "evaluator_reward_safe_rows"),
        "evaluator_penalty_applied_rmse_pooled": pooled_eval("evaluator_reward_unsafe_rmse", "evaluator_reward_unsafe_rows"),
        "family_termination_prevalence": {
            family: count / total if total else 0.0
            for family, (count, total) in family_counts.items()
        },
    }
    rows = [
        ["Safe hidden cells", str(result["cells"]), "One surrogate per cell; method duplicates removed"],
        ["Genuine terminations", str(result["terminated_events"]), f"{result['transitions']:,} public transitions"],
        ["Termination prevalence", f"{100 * result['terminated_prevalence']:.5f}\\%", "Public risk target"],
        ["Constant-risk fallback", f"{result['risk_fallback_cells']}/{result['cells']}", "Expected under near-single-class labels"],
        ["Mean predicted risk", f"{100 * result['risk_predicted_prevalence_mean']:.5f}\\%", "Holdout prediction"],
        ["Risk holdout Brier", fmt(result["risk_holdout_brier_mean"], 6), "Public, report-only"],
        ["Reward holdout RMSE", fmt(result["reward_holdout_rmse_mean"]), "Public"],
        ["Low-reward-tail RMSE", fmt(result["reward_low_tail_rmse_mean"]), "Public, report-only"],
        ["No-penalty RMSE", fmt(result["evaluator_no_penalty_rmse_pooled"]), "Evaluator-only stratum"],
        ["Penalty-applied RMSE", fmt(result["evaluator_penalty_applied_rmse_pooled"]), "Evaluator-only stratum"],
    ]
    write_csv("safety_diagnostics.csv", [{k: v for k, v in result.items() if not isinstance(v, dict)}])
    write_table(
        "safety_diagnostics.tex", "Hidden safe-mode public risk and reward-surrogate diagnostics.",
        "tab:safety-diagnostics", "p{0.30\\linewidth}rp{0.43\\linewidth}",
        ["Diagnostic", "Value", "Interpretation"], rows,
    )
    return result


def write_report_macros(tables, safety, recoverability):
    def select(rows, **wanted):
        return next(row for row in rows if all(row[key] == value for key, value in wanted.items()))

    safe_full_moor = select(tables["native"], reward_mode="safe", regime="full", comparator="moor_native")
    safe_hidden_moor = select(tables["native"], reward_mode="safe", regime="hidden", comparator="moor_native")
    safe_hidden_plus = select(tables["native"], reward_mode="safe", regime="hidden", comparator="plus_native")
    yield_hidden_moor = select(tables["native"], reward_mode="yield", regime="hidden", comparator="moor_native")
    yield_hidden_plus = select(tables["native"], reward_mode="yield", regime="hidden", comparator="plus_native")
    safe_ogsrl = select(tables["paired"], reward_mode="safe", method="ogsrl")
    safe_moor = select(tables["paired"], reward_mode="safe", method="moor_native")
    safe_plus = select(tables["paired"], reward_mode="safe", method="plus_native")

    macros = {
        "ReportMatchedCells": "288",
        "ReportUnmatchedCells": "0",
        "ReportRowsPerRegime": "1,440",
        "ReportPopulationCount": str(len(recoverability)),
        "ReportRecoverablePopulationCount": str(sum(recoverability.values())),
        "ReportSinkPopulationCount": str(sum(not value for value in recoverability.values())),
        "SafeFullMoorGap": fmt(safe_full_moor["gap_mean"], signed=True),
        "SafeHiddenMoorGap": fmt(safe_hidden_moor["gap_mean"], signed=True),
        "SafeHiddenMoorWins": f"{safe_hidden_moor['wins']}/{safe_hidden_moor['cells']}",
        "SafeHiddenPlusGap": fmt(safe_hidden_plus["gap_mean"], signed=True),
        "SafeHiddenPlusWins": f"{safe_hidden_plus['wins']}/{safe_hidden_plus['cells']}",
        "YieldHiddenMoorGap": fmt(yield_hidden_moor["gap_mean"], signed=True),
        "YieldHiddenPlusGap": fmt(yield_hidden_plus["gap_mean"], signed=True),
        "SafeOgsrlDelta": fmt(safe_ogsrl["hidden_minus_full_mean"], signed=True),
        "SafeMoorDelta": fmt(safe_moor["hidden_minus_full_mean"], signed=True),
        "SafePlusDelta": fmt(safe_plus["hidden_minus_full_mean"], signed=True),
        "TerminationEvents": str(safety["terminated_events"]),
        "TerminationTransitions": f"{safety['transitions']:,}",
        "TerminationPercent": f"{100 * safety['terminated_prevalence']:.5f}\\%",
        "RiskFallbackCells": f"{safety['risk_fallback_cells']}/{safety['cells']}",
        "RewardHoldoutRmse": fmt(safety["reward_holdout_rmse_mean"]),
        "RewardLowTailRmse": fmt(safety["reward_low_tail_rmse_mean"]),
        "RewardNoPenaltyRmse": fmt(safety["evaluator_no_penalty_rmse_pooled"]),
        "RewardPenaltyRmse": fmt(safety["evaluator_penalty_applied_rmse_pooled"]),
    }
    lines = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    (TABLES / "report_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_general_winner_composition(rows):
    def select(regime, reward, breakdown, level):
        return next(
            row for row in rows
            if row["regime"] == regime
            and row["reward_mode"] == reward
            and row["breakdown"] == breakdown
            and row["level"] == level
        )

    def stacked(ax, specs, labels, title):
        bottom = np.zeros(len(specs))
        x = np.arange(len(specs))
        for method in GENERAL:
            counts = np.asarray([row[f"{method}_wins"] for row in specs])
            totals = np.asarray([row["cells"] for row in specs])
            percentages = 100.0 * counts / totals
            bars = ax.bar(
                x, percentages, bottom=bottom, color=COLORS[method],
                label=DISPLAY[method], width=0.72,
            )
            for bar, count, percentage, base in zip(bars, counts, percentages, bottom):
                if percentage >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, base + percentage / 2,
                        str(int(count)), ha="center", va="center", fontsize=8,
                    )
            bottom += percentages
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("share of cells (\%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    overall_specs = [
        select(regime, reward, "overall", "all")
        for regime in REGIMES for reward in REWARDS
    ]
    stacked(
        axes[0, 0], overall_specs,
        [f"{DISPLAY[r]}\n{DISPLAY[w]}" for r in REGIMES for w in REWARDS],
        "Overall: regime and reward mode",
    )

    family_specs = [
        select("hidden", reward, "family", family)
        for reward in REWARDS for family in FAMILIES
    ]
    stacked(
        axes[0, 1], family_specs,
        [f"{DISPLAY[r][0]}:{DISPLAY[f]}" for r in REWARDS for f in FAMILIES],
        "Hidden regime by family",
    )

    sigma_specs = [
        select("hidden", reward, "sigma", f"{sigma:.1f}")
        for reward in REWARDS for sigma in SIGMAS
    ]
    stacked(
        axes[1, 0], sigma_specs,
        [f"{DISPLAY[r][0]}:{sigma:.1f}" for r in REWARDS for sigma in SIGMAS],
        "Hidden regime by observation noise",
    )

    scope_specs = [
        select("hidden", reward, "scope", scope)
        for reward in REWARDS for scope in ("recoverable", "sink")
    ]
    stacked(
        axes[1, 1], scope_specs,
        [f"{DISPLAY[r]}\n{'Recov.' if s == 'recoverable' else 'Sink'}"
         for r in REWARDS for s in ("recoverable", "sink")],
        "Hidden regime by population scope",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Which method supplies the cellwise best-general envelope?")
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    save_figure(fig, "general_winner_composition")


def plot_general_native_gap_heatmaps(rows, dimension):
    levels = [DISPLAY[level] for level in FAMILIES] if dimension == "family" else [f"{s:.1f}" for s in SIGMAS]
    lookup_levels = list(FAMILIES) if dimension == "family" else [f"{s:.1f}" for s in SIGMAS]

    def matrix(regime, reward, comparator):
        result = np.zeros((len(GENERAL), len(lookup_levels)))
        for method_index, method in enumerate(GENERAL):
            for level_index, level in enumerate(lookup_levels):
                row = next(
                    item for item in rows
                    if item["regime"] == regime
                    and item["reward_mode"] == reward
                    and item["comparator"] == comparator
                    and item["method"] == method
                    and item["level"] == level
                )
                result[method_index, level_index] = row["mean_gap"]
        return result

    matrices = {
        (regime, reward, comparator): matrix(regime, reward, comparator)
        for regime in REGIMES for reward in REWARDS for comparator in COMPARATORS
    }
    bound = max(float(np.max(np.abs(values))) for values in matrices.values())
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.8))
    panel_rows = [("full", "safe"), ("hidden", "safe"), ("full", "yield"), ("hidden", "yield")]
    image = None
    for row_index, (regime, reward) in enumerate(panel_rows):
        for col_index, comparator in enumerate(COMPARATORS):
            ax = axes[row_index, col_index]
            values = matrices[(regime, reward, comparator)]
            image = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels)
            ax.set_yticks(range(len(GENERAL)))
            ax.set_yticklabels([DISPLAY[method] for method in GENERAL])
            if row_index == 0:
                ax.set_title(DISPLAY[comparator])
            if col_index == 0:
                ax.set_ylabel(f"{DISPLAY[regime]} / {DISPLAY[reward]}")
            for method_index in range(values.shape[0]):
                for level_index in range(values.shape[1]):
                    value = values[method_index, level_index]
                    color = "white" if abs(value) > 0.58 * bound else "black"
                    ax.text(
                        level_index, method_index, f"{value:+.2f}",
                        ha="center", va="center", fontsize=8, color=color,
                    )
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.06, top=0.91, hspace=0.38, wspace=0.24)
    colorbar_axis = fig.add_axes([0.90, 0.16, 0.022, 0.68])
    fig.colorbar(image, cax=colorbar_axis, label="general method - native mean return")
    noun = "family" if dimension == "family" else "observation noise"
    fig.suptitle(f"Per-general-method native gaps by {noun} (all matched populations)")
    save_figure(fig, f"general_method_native_gap_by_{dimension}")


def plot_hidden_general_gap_distributions(cell_gaps):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    rng = np.random.default_rng(116)
    scope_colors = {True: "#4C78A8", False: "#E15759"}
    for row_index, reward in enumerate(REWARDS):
        for col_index, comparator in enumerate(COMPARATORS):
            ax = axes[row_index, col_index]
            distributions = []
            for method_index, method in enumerate(GENERAL):
                selected = [
                    row for row in cell_gaps
                    if row["reward_mode"] == reward
                    and row["comparator"] == comparator
                    and row["method"] == method
                ]
                values = np.asarray([row["gap"] for row in selected])
                distributions.append(values)
                for recoverable in (True, False):
                    scope_values = np.asarray([
                        row["gap"] for row in selected
                        if row["recoverable"] is recoverable
                    ])
                    jitter = rng.uniform(-0.12, 0.12, size=len(scope_values))
                    ax.scatter(
                        scope_values, method_index + jitter,
                        s=9, alpha=0.38, color=scope_colors[recoverable], linewidths=0,
                    )
            boxes = ax.boxplot(
                distributions, vert=False, positions=np.arange(len(GENERAL)),
                widths=0.46, patch_artist=True, showfliers=False,
                tick_labels=[DISPLAY[method] for method in GENERAL],
            )
            for box in boxes["boxes"]:
                box.set(facecolor="white", edgecolor="black", alpha=0.72)
            for median in boxes["medians"]:
                median.set(color="black", linewidth=1.4)
            ax.axvline(0, color="black", lw=1.0, ls="--")
            ax.set_title(f"{DISPLAY[reward]} vs {DISPLAY[comparator]}")
            ax.set_xlabel("method return - native return")
            ax.grid(axis="x", alpha=0.22)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=scope_colors[True],
               markersize=6, label="Recoverable"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=scope_colors[False],
               markersize=6, label="Sink"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Hidden-regime cell-level general-method gaps")
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    save_figure(fig, "hidden_general_native_gap_distributions")


def plot_method_returns(by_cell, recoverability):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.36
    for ax, reward in zip(axes, REWARDS):
        keys = selected_keys(by_cell, recoverability, reward=reward)
        for offset, regime in ((-width / 2, "full"), (width / 2, "hidden")):
            means = [np.mean([method_return(by_cell, regime, key, method) for key in keys]) for method in METHODS]
            ses = [np.std([method_return(by_cell, regime, key, method) for key in keys], ddof=1) / math.sqrt(len(keys)) for method in METHODS]
            ax.bar(x + offset, means, width, yerr=ses, capsize=2, label=DISPLAY[regime], color=COLORS[regime])
        ax.set_title(f"{DISPLAY[reward]} reward")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in METHODS], rotation=30, ha="right")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean operational return (all matched cells)")
    axes[1].legend(frameon=False)
    fig.suptitle("Known-parameter and hidden-demographics method returns")
    fig.tight_layout()
    save_figure(fig, "method_returns_full_vs_hidden")


def plot_family_gaps(by_cell, recoverability, comparator):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.36
    for ax, reward in zip(axes, REWARDS):
        for offset, regime in ((-width / 2, "full"), (width / 2, "hidden")):
            values = []
            for family in FAMILIES:
                keys = selected_keys(by_cell, recoverability, reward=reward, scope="recoverable", family=family)
                values.append(gap_stats(by_cell, regime, keys, comparator)["gap_mean"])
            ax.bar(x + offset, values, width, label=DISPLAY[regime], color=COLORS[regime])
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[f] for f in FAMILIES])
        ax.set_title(f"{DISPLAY[reward]} reward")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"best general - {DISPLAY[comparator]}")
    axes[1].legend(frameon=False)
    fig.suptitle(f"Native gap by family, recoverable populations: {DISPLAY[comparator]}")
    fig.tight_layout()
    save_figure(fig, f"general_minus_{comparator}_by_family")


def plot_sigma_gaps(by_cell, recoverability):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    x = np.arange(len(SIGMAS))
    width = 0.36
    for row_index, comparator in enumerate(COMPARATORS):
        for col_index, reward in enumerate(REWARDS):
            ax = axes[row_index, col_index]
            for offset, regime in ((-width / 2, "full"), (width / 2, "hidden")):
                values = []
                for sigma in SIGMAS:
                    keys = selected_keys(by_cell, recoverability, reward=reward, scope="recoverable", sigma=sigma)
                    values.append(gap_stats(by_cell, regime, keys, comparator)["gap_mean"])
                ax.bar(x + offset, values, width, color=COLORS[regime], label=DISPLAY[regime])
            ax.axhline(0, color="black", lw=0.9)
            ax.set_title(f"{DISPLAY[reward]} / {DISPLAY[comparator]}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s:.1f}" for s in SIGMAS])
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("best general - native")
    axes[1, 0].set_ylabel("best general - native")
    axes[1, 0].set_xlabel(r"observation noise $\sigma_{obs}$")
    axes[1, 1].set_xlabel(r"observation noise $\sigma_{obs}$")
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Native gaps by observation noise, recoverable populations")
    fig.tight_layout()
    save_figure(fig, "general_minus_native_by_sigma")


def plot_delta_heatmap(by_cell, recoverability):
    matrix = np.zeros((len(METHODS), len(REWARDS)))
    for j, reward in enumerate(REWARDS):
        keys = selected_keys(by_cell, recoverability, reward=reward)
        for i, method in enumerate(METHODS):
            matrix[i, j] = np.mean([
                method_return(by_cell, "hidden", key, method) - method_return(by_cell, "full", key, method)
                for key in keys
            ])
    bound = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(REWARDS)))
    ax.set_xticklabels([DISPLAY[r] for r in REWARDS])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([DISPLAY[m] for m in METHODS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="hidden - full mean return")
    ax.set_title("Matched-cell effect of hiding demographics")
    fig.tight_layout()
    save_figure(fig, "hidden_minus_full_delta_heatmap")


def plot_hidden_methods(by_cell, recoverability):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(METHODS))
    for ax, reward in zip(axes, REWARDS):
        for offset, method in zip(offsets, METHODS):
            means = []
            for family in FAMILIES:
                keys = selected_keys(by_cell, recoverability, reward=reward, scope="recoverable", family=family)
                means.append(np.mean([method_return(by_cell, "hidden", key, method) for key in keys]))
            ax.bar(x + offset, means, width, label=DISPLAY[method], color=COLORS[method])
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[f] for f in FAMILIES])
        ax.set_title(f"{DISPLAY[reward]} reward")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("hidden-mode mean return (recoverable cells)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("All five methods in the corrected hidden-demographics regime")
    fig.tight_layout()
    save_figure(fig, "all_methods_hidden_by_family")


def plot_safety(diag):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    family_values = [100 * diag["family_termination_prevalence"].get(f, 0.0) for f in FAMILIES]
    axes[0].bar([DISPLAY[f] for f in FAMILIES], family_values, color=["#4C78A8", "#59A14F", "#F28E2B", "#B279A2"])
    axes[0].set_ylabel("genuine termination prevalence (\%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Public termination target")
    axes[0].grid(axis="y", alpha=0.25)

    risk_values = [100 * diag["terminated_prevalence"], 100 * diag["risk_predicted_prevalence_mean"]]
    axes[1].bar(["Observed", "Predicted"], risk_values, color=["#E15759", "#76B7B2"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("prevalence (\%, log scale)")
    axes[1].set_title(f"Risk fallback in {diag['risk_fallback_cells']}/{diag['cells']} cells")
    axes[1].grid(axis="y", alpha=0.25)

    rmse = [
        diag["reward_holdout_rmse_mean"], diag["reward_low_tail_rmse_mean"],
        diag["evaluator_no_penalty_rmse_pooled"], diag["evaluator_penalty_applied_rmse_pooled"],
    ]
    axes[2].bar(["Holdout", "Low tail", "No penalty", "Penalty\napplied"], rmse,
                color=["#4C78A8", "#F28E2B", "#59A14F", "#E15759"])
    axes[2].set_ylabel("reward surrogate RMSE")
    axes[2].set_title("Public and evaluator-only diagnostics")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Hidden safe-mode information channels (report-only diagnostics)")
    fig.tight_layout()
    save_figure(fig, "safety_risk_diagnostics")


def plot_trajectories() -> list[str]:
    outputs = []
    for population, family in (
        ("Amur tiger", "ricker"),
        ("Puerto Rican parrot", "ricker"),
        ("Iberian lynx", "allee"),
        ("Egyptian vulture", "ricker"),
    ):
        slug = population.lower().replace(" ", "_").replace("-", "_")
        paths = {regime: TRACES / f"{slug}_{family}_sigma_0p2_{regime}.npz" for regime in REGIMES}
        if not all(path.exists() for path in paths.values()):
            print(f"trajectory pending: {population}")
            continue
        loaded = {regime: np.load(path, allow_pickle=False) for regime, path in paths.items()}
        metas = {regime: json.loads(str(data["meta"])) for regime, data in loaded.items()}
        fig, axes = plt.subplots(5, 2, figsize=(11.0, 13.2), sharex="col")
        for col, regime in enumerate(REGIMES):
            data = loaded[regime]
            meta = metas[regime]
            t = np.arange(int(meta["horizon"]))
            for method in METHODS:
                color = COLORS[method]
                state = np.nanmedian(data[f"{method}_state_post"], axis=0)
                observation = np.nanmedian(data[f"{method}_observation_post"], axis=0)
                reward = np.nanmedian(data[f"{method}_reward_true"], axis=0)
                action = np.nanmedian(data[f"{method}_action"], axis=0)
                unsafe = np.nanmean(data[f"{method}_private_unsafe"], axis=0)
                risk_values = data[f"{method}_method_risk"]
                risk = (
                    np.nanmedian(risk_values, axis=0)
                    if np.any(np.isfinite(risk_values))
                    else np.full(risk_values.shape[1], np.nan)
                )
                axes[0, col].plot(t, state, color=color, lw=1.4, label=DISPLAY[method])
                axes[1, col].plot(t, observation, color=color, lw=1.2)
                axes[2, col].plot(t, reward, color=color, lw=1.2)
                axes[3, col].step(t, action, color=color, lw=1.2, where="mid")
                axes[4, col].plot(t, unsafe, color=color, lw=1.2)
                if np.any(np.isfinite(risk)):
                    axes[4, col].plot(t, risk, color=color, lw=1.0, ls=":")
            axes[0, col].axhline(meta["safety_threshold_private"], color="black", ls="--", lw=0.9)
            axes[0, col].set_title(f"{DISPLAY[regime]} regime")
            for row in range(5):
                axes[row, col].grid(alpha=0.2)
        labels = ["true abundance", "observation", "true reward", "median action id", "private unsafe fraction\n(dotted: method risk)"]
        for row, label in enumerate(labels):
            axes[row, 0].set_ylabel(label)
        axes[-1, 0].set_xlabel("timestep")
        axes[-1, 1].set_xlabel("timestep")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=5, frameon=False)
        fig.suptitle(
            f"Illustrative only: {population}, {DISPLAY[family]}, sigma=0.2, safe reward\n"
            "Three fixed paired episodes; private safety traces are evaluator-only",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0.035, 1, 0.965])
        stem = f"trajectory_{slug}_{family}_full_vs_hidden"
        save_figure(fig, stem)
        outputs.append(stem)
        for data in loaded.values():
            data.close()
    return outputs


def write_figure_manifest(trajectory_stems):
    figures = [
        ("method_returns_full_vs_hidden", "Side-by-side mean method return in the two regimes, split by reward mode."),
        ("general_winner_composition", "Stacked winner composition for the cellwise best-general envelope, including hidden family, noise, and recoverability slices."),
        ("general_method_native_gap_by_family", "Per-general-method minus native mean-return heatmaps by family, regime, and reward mode."),
        ("general_method_native_gap_by_sigma", "Per-general-method minus native mean-return heatmaps by observation noise, regime, and reward mode."),
        ("hidden_general_native_gap_distributions", "Hidden-mode cell-level general-method minus native distributions, with recoverable and sink cells identified."),
        ("general_minus_moor_native_by_family", "Best-general minus MOOR-native gap by family on recoverable cells."),
        ("general_minus_plus_native_by_family", "Best-general minus PLUS-native gap by family on recoverable cells."),
        ("general_minus_native_by_sigma", "Best-general minus each native comparator by observation noise."),
        ("hidden_minus_full_delta_heatmap", "Matched-cell hidden-minus-full mean return by method and reward mode."),
        ("all_methods_hidden_by_family", "All five methods in hidden mode by family and reward mode."),
        ("safety_risk_diagnostics", "Public termination/risk and public/evaluator-only reward-surrogate diagnostics."),
    ]
    figures.extend((stem, "Illustrative paired-regime per-step trajectory panel; not headline evidence.") for stem in trajectory_stems)
    lines = [
        "# Figure Manifest", "",
        "Generated by:", "",
        "```bash",
        "MPLCONFIGDIR=/tmp/hidden_rk_report_mpl PYTHONPATH=code/src \\",
        "  python analysis/make_hidden_rk_report.py",
        "```", "",
        "Every entry is written as both `.png` and `.pdf` under `hidden_rk_report_figures/`.", "",
    ]
    for stem, description in figures:
        lines.extend([f"## `{stem}`", "", description, ""])
    lines.extend([
        "## Illustrative trajectory source", "",
        "The trajectory inputs were generated by Slurm array `58328792` with:", "",
        "```bash",
        "python analysis/capture_report_trajectories.py <index>",
        "```", "",
        "Indices 0--7 cover four preselected species/cells in both regimes. All eight Slurm tasks completed with exit code `0:0`. Each trace refits from the frozen cached dataset and rolls three fixed seeds (`7001`, `7051`, `7101`). These traces are explicitly excluded from headline statistics.", "",
    ])
    (ANALYSIS / "FIGURE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    summaries, by_cell, recoverability = load_data()
    tables = generate_tables(by_cell, recoverability, summaries)
    method_decomposition = generate_method_decomposition(by_cell, recoverability)
    safety = safety_diagnostics(by_cell, recoverability)
    write_report_macros(tables, safety, recoverability)
    plot_method_returns(by_cell, recoverability)
    plot_general_winner_composition(method_decomposition["winner_composition"])
    plot_general_native_gap_heatmaps(method_decomposition["family_gaps"], "family")
    plot_general_native_gap_heatmaps(method_decomposition["sigma_gaps"], "sigma")
    plot_hidden_general_gap_distributions(method_decomposition["hidden_cell_gaps"])
    plot_family_gaps(by_cell, recoverability, "moor_native")
    plot_family_gaps(by_cell, recoverability, "plus_native")
    plot_sigma_gaps(by_cell, recoverability)
    plot_delta_heatmap(by_cell, recoverability)
    plot_hidden_methods(by_cell, recoverability)
    plot_safety(safety)
    trajectories = plot_trajectories()
    write_figure_manifest(trajectories)
    report = {
        "run": str(RUN),
        "matched_cells_per_regime": 288,
        "unmatched_cells": [],
        "summary_rows_per_regime": {regime: len(rows) for regime, rows in summaries.items()},
        "population_counts": {
            "total": len(recoverability),
            "recoverable": sum(recoverability.values()),
            "sink": sum(not value for value in recoverability.values()),
        },
        "manifest_sha256": {
            regime: sha256(RUN / "manifests" / f"manifest_{regime}.csv") for regime in REGIMES
        },
        "tables": tables,
        "method_decomposition": method_decomposition,
        "safety": safety,
        "trajectory_figures": trajectories,
    }
    (ANALYSIS / "hidden_rk_report_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "ok", "figures": 11 + len(trajectories),
        "trajectory_figures": trajectories, "unmatched_cells": [],
    }, indent=2))


if __name__ == "__main__":
    main()
