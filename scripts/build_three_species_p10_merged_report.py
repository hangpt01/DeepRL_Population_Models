#!/usr/bin/env python3
"""Build the accepted P=10 six-method three-species report.

This script is intentionally outcome-reading.  It may only be run after both
method-specific structural acceptance records pass and outcome access has been
authorized.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


WORKSPACE = Path("/fs04/scratch2/ce25/Claude_DeepRL_Population_Models")
GENERAL_ROOT = Path(
    "/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/"
    "general_phase2e_full_sigma01_02_20260720_v1"
)
P10_ROOT = (
    WORKSPACE
    / "real_ecology_runs/three_species_ecological_p10_correction_20260723_v1"
)
PACKAGE = P10_ROOT / "launch_package"
OUTCOME_DIR = P10_ROOT / "analysis"
TEX_PATH = WORKSPACE / "docs/merged_results_three_species_matched_general_and_ecological.tex"
MATCHED_CSV = OUTCOME_DIR / "MATCHED_P10_144_METHOD_CELLS.csv"
RECEIPT_PATH = OUTCOME_DIR / "MATCHED_P10_144_RECEIPT.json"

SPECIES = ("Amur tiger", "Crab-eating fox", "Egyptian vulture")
SPECIES_CLASS = {
    "Amur tiger": "declining recoverable",
    "Crab-eating fox": "strong-growth recoverable",
    "Egyptian vulture": "demographic sink",
}
FAMILIES = ("ricker", "allee", "theta", "regime")
FAMILY_LABEL = {
    "ricker": "Ricker",
    "allee": "Allee",
    "theta": "Theta-logistic",
    "regime": "Regime-switching",
}
SIGMAS = (0.1, 0.2)
METHODS = (
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
    "plus_adapted_ricker_only_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
)
METHOD_LABEL = {
    "refplan": "RefPlan-inspired",
    "ogsrl": "OGSRL-inspired",
    "bamcts": "BA-MCTS-inspired",
    "ensemble_value_disagreement_pessimism": "EVD",
    "plus_adapted_ricker_only_pbvi": "PLUS P=10",
    "moor_adapted_ricker_misspec_pbvi": "MOOR P=10",
}
SHORT_METHOD_LABEL = {
    "refplan": "RefPlan",
    "ogsrl": "OGSRL",
    "bamcts": "BA-MCTS",
    "ensemble_value_disagreement_pessimism": "EVD",
    "plus_adapted_ricker_only_pbvi": "PLUS P=10",
    "moor_adapted_ricker_misspec_pbvi": "MOOR P=10",
}
GENERAL_METHODS = METHODS[:4]
ECO_METHODS = METHODS[4:]
EXPECTED_SEEDS = tuple(
    base + offset
    for base in (7001, 7051, 7101, 7151, 7201)
    for offset in range(4)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    return float(value)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row["population"]),
        str(row["environment"]),
        round(float(row["sigma_obs"]), 1),
        str(row["method"]),
    )


def load_acceptance() -> dict[str, dict[str, Any]]:
    result = {}
    for group in ("plus", "moor"):
        path = P10_ROOT / f"acceptance_{group}_p10.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["decision"] == "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
        assert payload["completed_rows"] == payload["expected_rows"] == 24
        assert payload["return_fields_opened"] is False
        assert not payload["failures"] and not payload["incomplete_reasons"]
        result[group] = {"path": str(path), "sha256": sha256(path), **payload}
    return result


def authoritative_dataset_map() -> dict[tuple[str, str, float], str]:
    maps = []
    for group in ("plus", "moor"):
        path = PACKAGE / f"manifests/{group}_p10_plan_24.csv"
        rows = read_csv(path)
        assert len(rows) == 24
        maps.append(
            {
                (
                    row["population"],
                    row["environment"],
                    round(float(row["sigma_obs"]), 1),
                ): row["authoritative_dataset_sha256"]
                for row in rows
            }
        )
    assert maps[0] == maps[1]
    assert len(maps[0]) == 24
    return maps[0]


def general_rows(dataset_map: dict[tuple[str, str, float], str]) -> list[dict[str, Any]]:
    path = GENERAL_ROOT / "analysis/matched_results/cell_method_metrics.csv"
    source = read_csv(path)
    selected = [
        row
        for row in source
        if row["reward_mode"] == "safe"
        and row["population"] in SPECIES
        and row["method"] in GENERAL_METHODS
    ]
    assert len(selected) == 96
    rows = []
    for row in selected:
        cell = (
            row["population"],
            row["environment"],
            round(float(row["sigma_obs"]), 1),
        )
        assert int(row["episodes"]) == 20
        rows.append(
            {
                "population": row["population"],
                "population_class": SPECIES_CLASS[row["population"]],
                "environment": row["environment"],
                "sigma_obs": float(row["sigma_obs"]),
                "reward_mode": "safe",
                "collapse_penalty": 10.0,
                "method": row["method"],
                "method_label": METHOD_LABEL[row["method"]],
                "episodes": 20,
                "evaluation_horizon": 50,
                "discount": 0.95,
                "operational_return_mean": f(row["mean_return"]),
                "operational_return_sd": f(row["return_sd_episodes"]),
                "collapse_rate": f(row["collapse_rate"]),
                "unsafe_fraction": f(row["unsafe_fraction"]),
                "mvp_breach_rate": f(row["mvp_breach_rate"]),
                "persistence_mean": f(row["persistence_rate"]),
                "min_population_mean": f(row["mean_episode_min_population"]),
                "economic_cost_mean": f(row["mean_economic_cost"]),
                "dataset_sha256": dataset_map[cell],
                "source": "accepted frozen general-RL P=10",
                "episodes_path": "",
                "episodes_sha256": "",
                "action_entropy_mean": "",
                "constant_episode_policy": "",
                "pbvi_action_margin_min": "",
            }
        )
    return rows


def diagnostics(artifact_root: Path) -> tuple[float, str]:
    path = artifact_root / "faithful_artifacts/pbvi_policy_diagnostics.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["last_action_values"], dtype=np.float64)
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    actions = ";".join(str(int(value)) for value in np.argmax(values, axis=1))
    return float(np.min(margins)), actions


def ecological_rows(
    dataset_map: dict[tuple[str, str, float], str]
) -> list[dict[str, Any]]:
    rows = []
    for group, method in zip(("plus", "moor"), ECO_METHODS):
        summaries = sorted((P10_ROOT / group / "evaluation").rglob("summary.json"))
        assert len(summaries) == 24
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary["reward_mode"] == "safe"
            assert summary["model"] == method
            assert int(summary["episodes"]) == 20
            assert int(summary["n_steps_mean"]) == 50
            assert int(summary["num_actions"]) == 11
            cell = (
                summary["population"],
                summary["environment"],
                round(float(summary["sigma_obs"]), 1),
            )
            assert summary["dataset_sha256"] == dataset_map[cell]
            episodes_path = summary_path.with_name("episodes.csv")
            episodes = read_csv(episodes_path)
            assert len(episodes) == 20
            assert tuple(int(row["seed"]) for row in episodes) == EXPECTED_SEEDS
            assert all(int(row["n_steps"]) == 50 for row in episodes)
            returns = [f(row["operational_return"]) for row in episodes]
            assert close(statistics.mean(returns), f(summary["operational_return_mean"]))
            assert close(statistics.stdev(returns), f(summary["operational_return_std"]))
            entropies = [f(row["action_entropy"]) for row in episodes]
            margin, actions = diagnostics(summary_path.parent)
            rows.append(
                {
                    "population": summary["population"],
                    "population_class": SPECIES_CLASS[summary["population"]],
                    "environment": summary["environment"],
                    "sigma_obs": float(summary["sigma_obs"]),
                    "reward_mode": "safe",
                    "collapse_penalty": 10.0,
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "episodes": 20,
                    "evaluation_horizon": 50,
                    "discount": 0.95,
                    "operational_return_mean": f(summary["operational_return_mean"]),
                    "operational_return_sd": f(summary["operational_return_std"]),
                    "collapse_rate": f(summary["collapse_entry_mean"]),
                    "unsafe_fraction": f(summary["unsafe_fraction_mean"]),
                    "mvp_breach_rate": f(summary["mvp_breach_mean"]),
                    "persistence_mean": f(summary["persistence_mean"]),
                    "min_population_mean": f(summary["min_true_state_mean"]),
                    "economic_cost_mean": f(summary["economic_cost_mean"]),
                    "dataset_sha256": summary["dataset_sha256"],
                    "source": "accepted P=10 ecological correction",
                    "episodes_path": str(episodes_path),
                    "episodes_sha256": sha256(episodes_path),
                    "action_entropy_mean": statistics.mean(entropies),
                    "constant_episode_policy": all(
                        abs(value) < 1e-12 for value in entropies
                    ),
                    "pbvi_action_margin_min": margin,
                    "pbvi_argmax_actions": actions,
                }
            )
    assert len(rows) == 48
    return rows


def verify_coverage(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 144
    keys = [key(row) for row in rows]
    assert len(set(keys)) == 144
    expected = {
        (population, family, sigma, method)
        for population in SPECIES
        for family in FAMILIES
        for sigma in SIGMAS
        for method in METHODS
    }
    assert set(keys) == expected
    for population in SPECIES:
        for family in FAMILIES:
            for sigma in SIGMAS:
                cell = [
                    row
                    for row in rows
                    if row["population"] == population
                    and row["environment"] == family
                    and row["sigma_obs"] == sigma
                ]
                assert len(cell) == 6
                assert len({row["dataset_sha256"] for row in cell}) == 1
                assert all(
                    row["episodes"] == 20
                    and row["evaluation_horizon"] == 50
                    and row["discount"] == 0.95
                    and row["collapse_penalty"] == 10.0
                    for row in cell
                )


def write_matched_csv(rows: list[dict[str, Any]]) -> None:
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    columns = (
        "population",
        "population_class",
        "environment",
        "sigma_obs",
        "reward_mode",
        "collapse_penalty",
        "method",
        "method_label",
        "episodes",
        "evaluation_horizon",
        "discount",
        "operational_return_mean",
        "operational_return_sd",
        "collapse_rate",
        "unsafe_fraction",
        "mvp_breach_rate",
        "persistence_mean",
        "min_population_mean",
        "economic_cost_mean",
        "dataset_sha256",
        "source",
        "episodes_path",
        "episodes_sha256",
        "action_entropy_mean",
        "constant_episode_policy",
        "pbvi_action_margin_min",
        "pbvi_argmax_actions",
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            SPECIES.index(row["population"]),
            FAMILIES.index(row["environment"]),
            row["sigma_obs"],
            METHODS.index(row["method"]),
        ),
    )
    with MATCHED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)


def lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, str], dict[str, Any]]:
    return {key(row): row for row in rows}


def result_cell(row: dict[str, Any]) -> str:
    return (
        rf"\shortstack{{{row['operational_return_mean']:.3f}\\"
        rf"$\pm$ {row['operational_return_sd']:.3f}}}"
    )


def six_method_table(
    rows_by_key: dict[tuple[str, str, float, str], dict[str, Any]],
    population: str,
) -> str:
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}lc*{6}{c}@{}}",
        r"\toprule",
        r"Family & Noise & RefPlan & OGSRL & BA-MCTS & EVD & PLUS P=10 & MOOR P=10 \\",
        r"\midrule",
    ]
    for family in FAMILIES:
        for sigma in SIGMAS:
            values = [
                result_cell(rows_by_key[(population, family, sigma, method)])
                for method in METHODS
            ]
            lines.append(
                f"{FAMILY_LABEL[family]} & {sigma:.1f} & "
                + " & ".join(values)
                + r" \\"
            )
        if family != FAMILIES[-1]:
            lines.append(r"\addlinespace[1.5pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                rf"\caption{{{tex(population)} ({tex(SPECIES_CLASS[population])}): "
                r"fair matched safe-reward comparison. Each entry is mean discounted "
                r"return $\pm$ within-cell sample SD, $n=20$.}"
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def shared_metric_table(
    rows_by_key: dict[tuple[str, str, float, str], dict[str, Any]],
    field: str,
    title: str,
) -> str:
    lines = [
        r"\begin{landscape}",
        r"\begin{center}\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{longtable}{@{}llc*{6}{r}@{}}",
        rf"\caption{{{title}. Values are 20-episode cell means.}}\\",
        r"\toprule",
        r"Species & Family & Noise & RefPlan & OGSRL & BA-MCTS & EVD & PLUS P=10 & MOOR P=10 \\",
        r"\midrule\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Species & Family & Noise & RefPlan & OGSRL & BA-MCTS & EVD & PLUS P=10 & MOOR P=10 \\",
        r"\midrule\endhead",
        r"\midrule\multicolumn{9}{r}{Continued on next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for population in SPECIES:
        for family in FAMILIES:
            for sigma in SIGMAS:
                values = [
                    rows_by_key[(population, family, sigma, method)][field]
                    for method in METHODS
                ]
                lines.append(
                    f"{tex(population)} & {FAMILY_LABEL[family]} & {sigma:.1f} & "
                    + " & ".join(f"{float(value):.3f}" for value in values)
                    + r" \\"
                )
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\end{longtable}", r"\end{center}", r"\end{landscape}"])
    return "\n".join(lines)


def summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[p]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Species & Method & Mean & Cell SD & Range & Cells \\",
        r"\midrule",
    ]
    for population in SPECIES:
        for method in METHODS:
            values = [
                row["operational_return_mean"]
                for row in rows
                if row["population"] == population and row["method"] == method
            ]
            assert len(values) == 8
            lines.append(
                f"{tex(population)} & {tex(SHORT_METHOD_LABEL[method])} & "
                f"{statistics.mean(values):.3f} & {statistics.stdev(values):.3f} & "
                f"[{min(values):.3f}, {max(values):.3f}] & 8 "
                + r"\\"
            )
        lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Species-separated descriptive summary across the eight matched family--noise cells. Cell SD is dispersion of the eight cell means, not an evaluation-episode SD. The sink is not pooled with recoverable species.}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def ecological_diagnostics(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Species & Adapted method & Mean entropy & Constant-episode cells & Minimum PBVI margin \\",
        r"\midrule",
    ]
    for population in SPECIES:
        for method in ECO_METHODS:
            selected = [
                row
                for row in rows
                if row["population"] == population and row["method"] == method
            ]
            lines.append(
                f"{tex(population)} & {tex(SHORT_METHOD_LABEL[method])} & "
                f"{statistics.mean(float(row['action_entropy_mean']) for row in selected):.3f} & "
                f"{sum(bool(row['constant_episode_policy']) for row in selected)}/8 & "
                f"{min(float(row['pbvi_action_margin_min']) for row in selected):.4f} "
                + r"\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Method-specific ecological policy diagnostics. ``Constant-episode'' means every episode in that cell had zero within-episode action entropy; it does not prove algorithmic equivalence. PBVI margin is the smallest final action-value top-one/top-two gap across candidates and cells.}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def yield_summary() -> str:
    path = GENERAL_ROOT / "analysis/matched_results/cell_method_metrics.csv"
    rows = [
        row
        for row in read_csv(path)
        if row["reward_mode"] == "yield"
        and row["population"] in SPECIES
        and row["method"] in GENERAL_METHODS
    ]
    assert len(rows) == 96
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Species & General-RL method & Mean across 8 cells & Cell SD & Cells \\",
        r"\midrule",
    ]
    for population in SPECIES:
        for method in GENERAL_METHODS:
            values = [
                f(row["mean_return"])
                for row in rows
                if row["population"] == population and row["method"] == method
            ]
            lines.append(
                f"{tex(population)} & {tex(SHORT_METHOD_LABEL[method])} & "
                f"{statistics.mean(values):.3f} & {statistics.stdev(values):.3f} & 8 "
                + r"\\"
            )
        lines.append(r"\addlinespace[1.5pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Supplementary general-RL-only yield summary. PLUS and MOOR were not evaluated for yield and are intentionally absent.}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def long_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{landscape}",
        r"\begin{center}\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}lllclrrrrrr@{}}",
        r"\caption{Complete accepted P=10 safe comparison: all 144 method-cells.}\\",
        r"\toprule",
        r"Species & Family & Noise & Method & Return mean & SD & Unsafe & Persist. & Collapse & Min.\ $N$ & Cost \\",
        r"\midrule\endfirsthead",
        r"\multicolumn{11}{c}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"Species & Family & Noise & Method & Return mean & SD & Unsafe & Persist. & Collapse & Min.\ $N$ & Cost \\",
        r"\midrule\endhead",
        r"\midrule\multicolumn{11}{r}{Continued on next page}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            SPECIES.index(row["population"]),
            FAMILIES.index(row["environment"]),
            row["sigma_obs"],
            METHODS.index(row["method"]),
        ),
    )
    for row in ordered:
        lines.append(
            f"{tex(row['population'])} & {FAMILY_LABEL[row['environment']]} & "
            f"{row['sigma_obs']:.1f} & {tex(row['method_label'])} & "
            f"{row['operational_return_mean']:.3f} & "
            f"{row['operational_return_sd']:.3f} & "
            f"{row['unsafe_fraction']:.3f} & {row['persistence_mean']:.3f} & "
            f"{row['collapse_rate']:.3f} & {row['min_population_mean']:.3f} & "
            f"{row['economic_cost_mean']:.3f} "
            + r"\\"
        )
    lines.extend([r"\end{longtable}", r"\end{center}", r"\end{landscape}"])
    return "\n".join(lines)


def build_tex(rows: list[dict[str, Any]], acceptance: dict[str, dict[str, Any]]) -> str:
    by_key = lookup(rows)
    pieces = [
        r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=17mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{pdflscape}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\setlist{nosep}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.12}
\title{Three-species matched general-RL and ecological comparison\\Safe reward, common \(P=10\)}
\author{Frozen general-RL results with accepted P=10 adapted PLUS and MOOR correction}
\date{24 July 2026}
\begin{document}
\maketitle

\begin{abstract}
This report gives the fair matched safe-reward comparison requested for Amur tiger,
Crab-eating fox, and Egyptian vulture.  It combines 96 accepted frozen general-RL
method-cells with 48 newly planned and evaluated ecological method-cells under the
same \(P=10\) objective: 3 species \(\times\) 4 hidden families \(\times\) 2
observation-noise levels \(\times\) 6 methods \(=144\) method-cells.  The adapted
PLUS and MOOR policies were re-planned for \(P=10\); this is not a re-scoring of
their historical \(P=5\) policies.  Every result uses 20 evaluation episodes.
\end{abstract}

\tableofcontents

\section{Primary fair matched safe-return results}
These are the first result tables by design.  Ricker is the in-family control for
the two form-committed ecological methods; Allee, theta-logistic, and
regime-switching are out-of-family tests.  PLUS and MOOR denote the registered
adapted implementations, not exact paper reproductions.  Species remain separate:
Amur tiger and Crab-eating fox are recoverable cases, whereas Egyptian vulture is
a demographic sink.
""",
        six_method_table(by_key, "Amur tiger"),
        six_method_table(by_key, "Crab-eating fox"),
        six_method_table(by_key, "Egyptian vulture"),
        r"""
\clearpage
\section{Evaluation-comparability audit}
The comparison passes the evaluator-definition audit.  Exact dataset identity
alone was not treated as sufficient.  The general, PLUS, and MOOR runtimes have
byte-identical evaluator, environment, reward, action, real-data, and public type
implementations where those components determine evaluation.  The method-specific
configuration differences affect fitting and planning, not the common hidden
evaluation simulator.

\begin{table}[ht]
\centering\small
\begin{tabular}{@{}p{3.3cm}p{11.2cm}@{}}
\toprule
Item & Verified common definition\\
\midrule
Reward & \(R_t=s_{t+1}/(s_{t+1}+K_{\rm ref})-c(a_t)-10\,\mathbf{1}[s_{t+1}\le s_{\rm safe}]\), with \(\alpha=1\), true next state, occupancy penalty, and no additional scaling.\\
Return & \(\sum_{t=0}^{49}0.95^t R_t\); all tables use the same 50-step horizon and discount \(0.95\).\\
Initial state & Registered deterministic population-specific \(N_0\): evaluation uses low-start probability \(0\) and initial log-SD \(0\).\\
Hidden simulator & Exact registered population, Ricker/Allee/theta-logistic/regime-switching identity, parameters, process randomness, and authoritative private dataset hash.\\
Observation & Same log-normal observation model and cell-specific \(\sigma_{\rm obs}\in\{0.1,0.2\}\).\\
Episodes & Five blocks \(7001,7051,7101,7151,7201\), four episodes per block, giving seeds 7001--7004, 7051--7054, 7101--7104, 7151--7154, and 7201--7204.\\
Actions & Identical ordered 11-action table, population/family-specific effects, cost, and action-to-intervention mapping.\\
Safety metrics & Collapse entry is a transition from above to at/below \(s_{\rm safe}\); unsafe fraction counts unsafe states over the 51-state episode trace; persistence is final state \(>s_{\rm safe}\); abundance metrics use the same true-state trace.\\
\bottomrule
\end{tabular}
\caption{Common-evaluator audit. All 24 species--family--noise cells also have one exact authoritative dataset SHA-256 shared by all six methods.}
\end{table}

\subsection{Common 11-action semantics}
\begin{center}\small
\begin{tabular}{@{}cll@{}}
\toprule
ID & Intervention & Public channel\\
\midrule
0 & Do nothing & none\\
1 & Sustainable harvest & demographic rate\\
2 & Aggressive harvest & demographic rate\\
3 & Predator/disease control & demographic rate\\
4 & Breeding/recruitment support & demographic rate\\
5 & Moderate restoration & carrying capacity\\
6 & Intensive restoration & carrying capacity\\
7 & Integrated conservation (light) & rate + capacity\\
8 & Adaptive conservation trial & rate + capacity\\
9 & Flagship conservation programme & rate + capacity\\
10 & Translocation & direct state increment\\
\bottomrule
\end{tabular}
\end{center}
Action effects are resolved from the same authoritative table for each population
and hidden family; IDs are categorical and are not treated as an ordinal scale.

\section{Shared six-method outcomes}
Unsafe occupancy, persistence, and collapse entry are defined identically and are
therefore compared directly.  For a sink already below its safety threshold,
collapse entry can be zero while unsafe occupancy is one and persistence is zero;
collapse entry must not be interpreted alone.
""",
        shared_metric_table(by_key, "unsafe_fraction", "Unsafe fraction"),
        shared_metric_table(by_key, "persistence_mean", "Persistence"),
        shared_metric_table(by_key, "collapse_rate", "Collapse-entry rate"),
        r"\section{Species-separated summary}",
        summary_table(rows),
        r"""
\section{Method-specific diagnostics and interpretation}
The ecological methods use reward-independent fitted demographic caches but newly
built reward-dependent POMDP components and newly solved \(P=10\) PBVI policies.
PLUS uses eight Ricker candidates with a uniform prior; MOOR uses its registered
adapted Ricker-misspecified route.  Neither method receives the hidden family
label.  Equal or nearly equal returns can arise from constant or nearly constant
policies and must not be called algorithmic equivalence.
""",
        ecological_diagnostics(rows),
        r"""
The four general methods retain their own diagnostics (for example OGSRL fallback)
outside the shared tables.  PBVI margin, candidate entropy, fallback count, and
action entropy are method-specific mechanisms and are not inserted as if they
were common outcomes.  Descriptive return differences are not multiplicity-
adjusted rankings.

\section{Supplementary general-RL-only yield results}
The \(P=10\) correction is safe-only.  The historical general-RL yield outcomes
remain valid within their own reward mode, but no PLUS or MOOR yield policies were
planned or evaluated.  Yield is therefore never placed in a six-method table.
""",
        yield_summary(),
        r"""
\section{Provenance and acceptance chronology}
All 216 reward-independent fitted-model slots were reused after complete cache
identity and artifact-hash validation: 192 PLUS candidate slots and 24 MOOR
fitted-model slots; zero fits were recomputed.  Reuse did not treat a \(P=5\)
full-configuration cache as a \(P=10\) policy cache.  All 24 PLUS and all 24 MOOR
reward-dependent POMDP/planner/evaluation rows were executed under \(P=10\).

Slurm arrays \texttt{58493916} (PLUS) and \texttt{58493918} (MOOR) completed
24/24 tasks with exit status 0:0.  Return-blind acceptance jobs
\texttt{58493967} and \texttt{58493968} then returned
\texttt{PASS\_LIMITED\_STRUCTURAL\_ACCEPTANCE}, 24/24 each, with
\texttt{return\_fields\_opened=false}.  Outcomes were opened only after those
gates passed and the user returned with explicit instruction to fill this report.

\begin{description}[style=nextline]
\item[P=10 experiment root.]
{\tiny\path{/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1/}}
\item[Launch-package digest.]
\texttt{13f6fcdf12450efb09f3c62c60db05c4c41007f64356a3999eb5d38d7e34368d}
\item[PLUS manifest/config.]
\texttt{cfe43f35ce4ff1658958a2173026ca70c7164878f2408b4b246673016dc0d2a6}\\
\texttt{10a064344e07cac16a7b2d5bd717bdce109c8970148de4295d6ff3532297fa68}
\item[MOOR manifest/config.]
\texttt{f890c9a7a535fe903bf7a40ecf992a9881341b84103b9e0dc7e10be002fc6d57}\\
\texttt{a3655a846d951439177da350ddf40ddc6d4962b42b317582dd29e72ca05323bd}
\item[Fit-reuse ledger.]
\texttt{5410b0e4fd71312af601cbdd4c8f11115469483780f26c202fbe232916187ac3}
\end{description}

\appendix
\section{Historical \(P=5\) ecological experiment---not matched returns}
The earlier ecological outputs are preserved unchanged as adapted/inspired results
under a different reward.  Their policies were planned with \(P=5\), so their
returns are not inserted into, ranked with, or used to fill any \(P=10\) matched
table.  The earlier global Ricker-only PLUS experiment also remains honestly
classified as 71/72 because index 51 timed out.  Those historical artifacts and
acceptance receipts were neither overwritten nor relabelled by this correction.

\section{Complete 144-cell audit table}
""",
        long_table(rows),
        r"""
\section{Reporting boundaries}
\begin{enumerate}
\item This is a selected three-species benchmark, not a nine-population result.
\item Ricker is a correct-form inductive bias for the ecological methods; the
other three hidden families deliberately test form misspecification.  Results
must be reported family by family.
\item The claim supported here is a comparison of form-committed adapted
ecological planning with form-flexible general MBRL under the registered
conditions, not a universal statement that one class beats the other.
\item Within-cell SD is conditional on one fitted policy and one registered
dataset; it is not uncertainty over independent datasets or refits.
\item PLUS and MOOR are registered adapted implementations, not exact paper
reproductions.
\end{enumerate}

\end{document}
""",
    ]
    return "\n\n".join(pieces)


def main() -> None:
    acceptance = load_acceptance()
    dataset_map = authoritative_dataset_map()
    rows = general_rows(dataset_map) + ecological_rows(dataset_map)
    verify_coverage(rows)
    write_matched_csv(rows)
    TEX_PATH.write_text(build_tex(rows, acceptance), encoding="utf-8")
    receipt = {
        "receipt_schema": "three_species_matched_p10_outcome_report_v1",
        "acceptance_gate": {
            group: {
                "decision": payload["decision"],
                "completed_rows": payload["completed_rows"],
                "expected_rows": payload["expected_rows"],
                "return_fields_opened_at_gate": payload["return_fields_opened"],
                "sha256": payload["sha256"],
            }
            for group, payload in acceptance.items()
        },
        "outcomes_opened_after_acceptance": True,
        "outcome_opening_authorization": (
            "User returned after completion and instructed Codex to check success "
            "and fill docs/merged_results_three_species_matched_general_and_ecological.tex"
        ),
        "coverage": {
            "general_p10_safe_rows": 96,
            "plus_p10_safe_rows": 24,
            "moor_p10_safe_rows": 24,
            "total_safe_method_cells": 144,
            "species": list(SPECIES),
            "families": list(FAMILIES),
            "sigma_obs": list(SIGMAS),
            "episodes_per_cell": 20,
        },
        "comparability_audit": {
            "decision": "PASS_FAIR_MATCHED_COMPARISON",
            "collapse_penalty": 10.0,
            "evaluation_horizon": 50,
            "discount": 0.95,
            "evaluation_episode_seeds": list(EXPECTED_SEEDS),
            "exact_dataset_hash_per_cell_across_six_methods": True,
            "evaluator_sha256": "125ec6073b203e34dc10d7182082e9439d577c9fa819953d280b3401128136bb",
            "environment_sha256": "c7079a34155c482ed449359c72b7089ac04dbc64f148776347f54a75066b4b06",
            "reward_sha256": "356d126d23fedc69e3632b923b6e8a9a071896813f7ddf9f5ad509823a8b33d8",
            "actions_sha256": "b591e63937a3eefe5570de5adf2d479b83b3783e5e83dde864eecbbeaee9a673",
            "realdata_sha256": "f293ed77f79208446fb04df3498b68941f15fb7df0979e8b8d3362ef8ed546d1",
        },
        "matched_csv": str(MATCHED_CSV),
        "matched_csv_sha256": sha256(MATCHED_CSV),
        "tex": str(TEX_PATH),
        "tex_sha256": sha256(TEX_PATH),
        "historical_p5_mixed_into_matched_tables": False,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
