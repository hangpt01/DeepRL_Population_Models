#!/usr/bin/env python3
"""After-any finalizer: registered calculations and sealing only; runs no method."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from recovery_common import END_TO_END, EVD, LABEL, activity, sha, source_only_hash, strict_json

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/stageb_sigma02_corrected_recovery1_20260809"
BEST_FIXED = {"tiger": {"action": 10, "return": 4.224420899026084},
              "fox": {"action": 1, "return": 10.153605271956147}}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interval(values: np.ndarray) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(20260808))
    indices = rng.integers(0, len(values), size=(100000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025, method="linear")),
            float(np.quantile(means, 0.975, method="linear"))]


def residual_comparison(o: dict[str, Any], t: dict[str, Any]) -> dict[str, Any]:
    left = np.asarray(o["per_member_raw"], dtype=np.float64)
    right = np.asarray(t["per_member_raw"], dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError("Arm O/T residual member cardinality mismatch")
    difference = right - left
    ratios = [float(b / a) if a > 0 else None for a, b in zip(left, right)]
    return {"arm_o_raw": left.tolist(), "arm_t_raw": right.tolist(),
            "t_minus_o": difference.tolist(), "absolute_difference": np.abs(difference).tolist(),
            "guarded_t_over_o_ratio": ratios,
            "ratio_guard": "Arm O denominator > 0; no numerical magnitude threshold",
            "arm_o_floor_active": o["floor_active"], "arm_t_floor_active": t["floor_active"]}


def posterior_summary(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text())
    if not data["episodes"] or data["episodes"][0]["initial_posterior"] is None:
        return None
    initial, terminal, trajectories = [], [], []
    for episode in data["episodes"]:
        initial.append(float(episode["initial_posterior"]["entropy"]))
        trace = [float(step["posterior_after_observe"]["entropy"]) for step in episode["steps"]]
        trajectories.append(trace)
        terminal.append(trace[-1])
    return {"initial_entropy_raw": initial, "per_episode_per_step_entropy": trajectories,
            "terminal_entropy_raw": terminal, "initial_entropy_mean": float(np.mean(initial)),
            "terminal_entropy_mean": float(np.mean(terminal)),
            "terminal_effective_model_count_mean": float(np.mean(np.exp(terminal)))}


def task_summary(task: dict[str, Any], o_receipt: dict[str, Any],
                 t_receipt: dict[str, Any]) -> dict[str, Any]:
    o_rows, t_rows = csv_rows(Path(o_receipt["episodes_path"])), csv_rows(Path(t_receipt["episodes_path"]))
    if len(o_rows) != 20 or len(t_rows) != 20:
        raise RuntimeError("paired episode cardinality mismatch")
    for o, t in zip(o_rows, t_rows):
        if (o["episode"], o["seed"], o["block_seed"]) != (t["episode"], t["seed"], t["block_seed"]):
            raise RuntimeError("paired identity mismatch")
    oe = json.loads(Path(o_receipt["timestep_evidence"]).read_text())
    te = json.loads(Path(t_receipt["timestep_evidence"]).read_text())
    crn_mismatches, paired_action_differences = [], []
    for episode, (left, right) in enumerate(zip(oe["episodes"], te["episodes"])):
        if left["seed"] != right["seed"] or len(left["timesteps"]) != len(right["timesteps"]):
            crn_mismatches.append({"episode": episode, "reason": "identity_or_length"}); continue
        differing = 0
        for step, (a, b) in enumerate(zip(left["timesteps"], right["timesteps"])):
            for key in ("process_rng_before_sha256", "process_rng_after_sha256",
                        "observation_rng_before_sha256", "observation_rng_after_sha256",
                        "process_draw_count_before", "process_draw_count_after",
                        "observation_draw_count_before", "observation_draw_count_after"):
                if a[key] != b[key]:
                    crn_mismatches.append({"episode": episode, "step": step, "field": key})
            differing += int(a["action"] != b["action"])
        paired_action_differences.append(differing / max(len(left["timesteps"]), 1))
    if crn_mismatches:
        raise RuntimeError(f"common-random-number evidence mismatch: {crn_mismatches[:3]}")
    o_values = np.asarray([float(row["true_return"]) for row in o_rows])
    t_values = np.asarray([float(row["true_return"]) for row in t_rows])
    difference = t_values - o_values

    def evidence_metrics(data: dict[str, Any]) -> dict[str, Any]:
        component = {key: [] for key in ("benefit", "cost", "safety", "total", "pre", "post")}
        collapse, first, unsafe, penalty_duration = [], [], [], []
        all_actions = []
        for episode in data["episodes"]:
            rows = episode["timesteps"]; all_actions.extend(row["action"] for row in rows)
            entries = [row["timestep"] for row in rows if row["collapse_entry_indicator"]]
            first_step = entries[0] if entries else None
            collapse.append(bool(entries)); first.append(first_step)
            unsafe.append(sum(row["unsafe_indicator"] for row in rows))
            penalty_duration.append(sum(row["safety_penalty_term"] != 0.0 for row in rows))
            component["benefit"].append(sum(row["discounted_benefit_reward_term"] for row in rows))
            component["cost"].append(sum(row["discounted_action_cost_term"] for row in rows))
            component["safety"].append(sum(row["discounted_safety_penalty_term"] for row in rows))
            component["total"].append(sum(row["discounted_total_true_reward"] for row in rows))
            component["pre"].append(sum(row["discounted_total_true_reward"] for row in rows
                                        if first_step is None or row["timestep"] < first_step))
            component["post"].append(sum(row["discounted_total_true_reward"] for row in rows
                                         if first_step is not None and row["timestep"] >= first_step))
        return {"discounted_component_raw": component,
                "discounted_component_means": {key: float(np.mean(value)) for key, value in component.items()},
                "collapse_rate": float(np.mean(collapse)), "first_collapse_timing_raw": first,
                "first_collapse_timing_mean_when_present": float(np.mean([x for x in first if x is not None])) if any(x is not None for x in first) else None,
                "unsafe_duration_raw": unsafe, "unsafe_duration_mean": float(np.mean(unsafe)),
                "safety_penalty_duration_raw": penalty_duration,
                "safety_penalty_duration_mean": float(np.mean(penalty_duration)),
                "activity": activity(all_actions), "full_action_sequences":
                [[row["action"] for row in episode["timesteps"]] for episode in data["episodes"]]}

    om, tm = evidence_metrics(oe), evidence_metrics(te)
    oc = json.loads(Path(o_receipt["pre_return_component_receipt"]).read_text())
    tc = json.loads(Path(t_receipt["pre_return_component_receipt"]).read_text())
    residuals = residual_comparison(oc["residual_sigma"], tc["residual_sigma"]) \
        if task["method"] in {"refplan", "ogsrl", "bamcts"} else None
    op, tp = posterior_summary(Path(o_receipt["policy_evidence"])), posterior_summary(Path(t_receipt["policy_evidence"]))
    posterior = None
    if task["method"] == "bamcts":
        posterior = {"arm_o": op, "arm_t": tp,
                     "terminal_entropy_t_minus_o": float(tp["terminal_entropy_mean"] - op["terminal_entropy_mean"]),
                     "terminal_entropy_t_over_o": float(tp["terminal_entropy_mean"] / op["terminal_entropy_mean"]) if op["terminal_entropy_mean"] > 0 else None,
                     "complete_realized_trajectory": True}
    if task["method"] in ECO:
        if o_receipt["fitted_artifact"]["pickle_sha256"] != t_receipt["fitted_artifact"]["pickle_sha256"]:
            raise RuntimeError("ecological policy bytes differ across arms")
    fixed = BEST_FIXED[task["cell"]]
    return {"status_label": LABEL, "cell": task["cell"], "method": task["method"],
        "evd_separate_raw_logged_reward_objective": task["method"] == EVD,
        "arm_o_absolute_return_raw": o_values.tolist(), "arm_t_absolute_return_raw": t_values.tolist(),
        "arm_o_mean_return": float(o_values.mean()), "arm_t_mean_return": float(t_values.mean()),
        "paired_t_minus_o_raw": difference.tolist(), "paired_t_minus_o_mean": float(difference.mean()),
        "paired_percentile_bootstrap_95_interval": interval(difference),
        "positive_negative_zero_counts": {"positive": int(np.sum(difference > 0)),
                                          "negative": int(np.sum(difference < 0)),
                                          "zero": int(np.sum(difference == 0))},
        "normalized_secondary": None if abs(float(t_values.mean())) < 1.0 else
            float(difference.mean() / max(abs(float(t_values.mean())), 1.0)),
        "arm_o": om, "arm_t": tm,
        "paired_action_difference_fraction_raw": paired_action_differences,
        "paired_action_difference_fraction_mean": float(np.mean(paired_action_differences)),
        "crn_evidence": {"result": "PASS", "mismatches": []},
        "best_fixed_anchor": fixed,
        "adaptive_headroom": {"arm_o": float(o_values.mean() - fixed["return"]),
                              "arm_t": float(t_values.mean() - fixed["return"])},
        "residual_diagnostics": residuals, "bamcts_realized_posterior": posterior,
        "refplan_predictive_dispersion": {"arm_o": oc["refplan_predictive_dispersion"],
                                           "arm_t": tc["refplan_predictive_dispersion"]}
                                           if task["method"] == "refplan" else None,
        "component_change": {"arm_o_artifact_sha256": o_receipt["fitted_artifact"]["pickle_sha256"],
                             "arm_t_artifact_sha256": t_receipt["fitted_artifact"]["pickle_sha256"],
                             "matched_surrogate_sha256": oc["matched_surrogate_sha256"],
                             "interpretation_label": tc["interpretation_label"]},
        "scientific_hypothesis_interpretation": None}


def main() -> None:
    for name in ("PROVISIONAL_CORRECTED_RESULTS.json", "PROVISIONAL_CORRECTED_RESULTS.md",
                 "CORRECTED_FINALIZER_RECEIPT.json", "CORRECTED_COMPLETION_REPORT.md",
                 "CORRECTED_OUTPUT_HASHES.sha256"):
        if (OUT / name).exists():
            raise RuntimeError(f"finalizer collision: {name}")
    manifest = json.loads((DOC / "CORRECTED_TASK_MANIFEST.json").read_text())
    registration = json.loads((DOC / "CORRECTED_STAGEB_REGISTRATION.json").read_text())
    completed: dict[str, dict[str, Any]] = {"tiger": {}, "fox": {}}
    failures, contract_failures = [], []
    for task in manifest["tasks"]:
        opath = OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json"
        tpath = OUT / "arm_t_tasks" / task["task_id"] / "task_receipt.json"
        if not opath.exists() or not tpath.exists():
            failures.append({"task": task, "reason": "missing task receipt"}); continue
        ore, tre = json.loads(opath.read_text()), json.loads(tpath.read_text())
        if ore.get("exit_status") != 0 or tre.get("exit_status") != 0:
            failures.append({"task": task, "arm_o_exit": ore.get("exit_status"),
                             "arm_t_exit": tre.get("exit_status"),
                             "failure": tre.get("failure") or ore.get("failure")}); continue
        try:
            completed[task["cell"]][task["method"]] = task_summary(task, ore, tre)
        except Exception as exc:
            contract_failures.append({"task": task, "reason": str(exc)})
    sources = {track: source_only_hash(ROOT, track) for track in ("ecological", "general")}
    if sources != registration["frozen_source_only_hashes"]:
        contract_failures.append({"reason": "frozen source hash mismatch", "observed": sources})
    status = "FAIL" if contract_failures else "PARTIAL" if failures else "PASS"
    result = {"schema_version": "corrected_recovery_results_v1", "status_label": LABEL,
              "scientific_interpretation_performed": False, "species_pooled": False,
              "cells": completed, "task_failures": failures,
              "contract_or_information_boundary_failures": contract_failures,
              "frozen_source_hashes": sources, "finalizer_job_id": os.getenv("SLURM_JOB_ID")}
    strict_json(OUT / "PROVISIONAL_CORRECTED_RESULTS.json", result)
    lines = ["# Provisional corrected Stage B recovery results", "", LABEL, "",
             "Tiger and fox are separate; EVD is separately labelled. No hypothesis interpretation was performed.", ""]
    for cell in ("tiger", "fox"):
        lines.extend([f"## {cell}", ""])
        for method, row in completed[cell].items():
            lines.extend([f"### {method}", "", LABEL, "",
                          f"- Arm O mean: `{row['arm_o_mean_return']}`",
                          f"- Arm T mean: `{row['arm_t_mean_return']}`",
                          f"- Paired T−O: `{row['paired_t_minus_o_mean']}`",
                          f"- Paired 95% interval: `{row['paired_percentile_bootstrap_95_interval']}`",
                          f"- Interpretation label: `{row['component_change']['interpretation_label']}`", ""])
    (OUT / "PROVISIONAL_CORRECTED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    strict_json(OUT / "CORRECTED_FINALIZER_RECEIPT.json", {"schema_version": "corrected_recovery_finalizer_v1",
        "status": status, "completed_tasks": sum(len(x) for x in completed.values()),
        "task_failures": len(failures), "contract_failures": len(contract_failures),
        "methods_run_by_finalizer": 0, "completed_unix": time.time(),
        "job_id": os.getenv("SLURM_JOB_ID"), "dependency": os.getenv("SLURM_JOB_DEPENDENCY")})
    verdict = ("PASS — CORRECTED STAGE B OUTPUTS SEALED; READY FOR INDEPENDENT AUDIT" if status == "PASS"
               else "PARTIAL — CORRECTED STAGE B TASK FAILURE; NO RETRY AUTHORIZED" if status == "PARTIAL"
               else "FAIL — CORRECTED STAGE B CONTRACT OR INFORMATION BOUNDARY FAILED")
    (OUT / "CORRECTED_COMPLETION_REPORT.md").write_text(
        "# Corrected Stage B completion report\n\n" + LABEL +
        f"\n\nCompleted tasks: `{sum(len(x) for x in completed.values())}/12`. "
        f"Task failures: `{len(failures)}`. Contract failures: `{len(contract_failures)}`.\n\n" + verdict + "\n",
        encoding="utf-8")
    checksum = OUT / "CORRECTED_OUTPUT_HASHES.sha256"
    paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text("# Self-excluded manifest; repository-relative paths.\n" + "".join(
        f"{sha(path)}  {path.relative_to(ROOT)}\n" for path in paths), encoding="utf-8")


if __name__ == "__main__":
    main()
