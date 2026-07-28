#!/usr/bin/env python3
"""Limited structural acceptance for the frozen 32-cell diagnostic.

This module intentionally refuses summary/return/comparison/ranking inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from phase2_launch import (
    FIT_SHA256,
    PLAN_SHA256,
    RUNTIME_DIGEST,
    METHODS,
    fit_cell,
    read_rows,
    sha256_file,
    slug,
    sigma_slug,
    verify_runtime,
)


VERSION = "acceptance_v2_frozen_runtime_observable_20260719"
FORBIDDEN_PATH_NAMES = {"summary.json", "comparative_summary.json", "ranking.json"}
FORBIDDEN_FIELD_FRAGMENTS = ("operational_return", "true_return", "survival_return", "ranking")
NOT_EVALUABLE = [
    "3: per-start L-BFGS termination status",
    "4b: projected gradient and free/bound-active coordinates",
    "5: parameter-boundary occupancy",
    "7b: holdout/train error ratio",
    "9: PLUS posterior history",
    "10b: run-specific PBVI deterministic repeat",
]
CRITERION_MAPPING = [
    {
        "original": 1,
        "classification": "EVALUABLE GATE",
        "amendment": "frozen identity, privacy audit, method and planner provenance",
    },
    {
        "original": 2,
        "classification": "EVALUABLE GATE",
        "amendment": "selected fitted objectives finite",
    },
    {
        "original": 3,
        "classification": "NOT EVALUABLE",
        "amendment": "per-start termination status not emitted",
    },
    {
        "original": 4,
        "classification": "DESCRIPTIVE ONLY",
        "amendment": "traces/raw gradients reported; projected bound-aware gradient unavailable",
    },
    {
        "original": 5,
        "classification": "NOT EVALUABLE",
        "amendment": "boundary occupancy not emitted",
    },
    {
        "original": 6,
        "classification": "DESCRIPTIVE ONLY",
        "amendment": "start objectives reported without registered agreement tolerance",
    },
    {
        "original": 7,
        "classification": "DESCRIPTIVE ONLY",
        "amendment": "holdout SSE reported without comparable train ratio",
    },
    {
        "original": 8,
        "classification": "DESCRIPTIVE ONLY",
        "amendment": "candidate composition and hashes reported",
    },
    {
        "original": 9,
        "classification": "NOT EVALUABLE",
        "amendment": "posterior history not emitted outside forbidden summaries",
    },
    {
        "original": 10,
        "classification": "EVALUABLE GATE + NOT EVALUABLE remainder",
        "amendment": "finite/legal action values gated; deterministic repeat unavailable",
    },
    {
        "original": 11,
        "classification": "EVALUABLE GATE",
        "amendment": "emitted receipt/model/POMDP/planner hashes cross-checked",
    },
    {
        "original": 12,
        "classification": "EVALUABLE GATE",
        "amendment": "all receipts, artifacts, and successful task exits required",
    },
    {
        "original": 13,
        "classification": "EVALUABLE GATE",
        "amendment": "Slurm core-hours, memory, states, and exits checked",
    },
]


def guarded_json(path: Path) -> dict[str, Any]:
    if path.name in FORBIDDEN_PATH_NAMES:
        raise RuntimeError(f"acceptance is forbidden from opening {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    def scan(value: Any, location: str = "root") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                    raise RuntimeError(f"forbidden return/ranking field at {location}.{key}")
                scan(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan(item, f"{location}[{index}]")

    scan(payload)
    return payload


def artifact_root(run_root: Path, row: dict[str, str]) -> Path | None:
    base = (
        run_root
        / "evaluation"
        / f"regime_{row['expose_rk']}"
        / slug(row["population"])
        / row["environment"]
        / f"sigma_{sigma_slug(row['sigma_obs'])}"
    )
    matches = list(base.glob(f"**/{row['method']}/{row['filter']}/faithful_artifacts"))
    return matches[0] if len(matches) == 1 else None


def parse_rss_mb(value: str) -> float | None:
    if not value:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value.strip(), re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).upper()
    return (
        number
        * {"": 1 / 1024**2, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}[unit]
    )


def slurm_records(
    fit_jobs: list[str], plan_jobs: list[str], sacct_file: Path | None
) -> list[dict[str, str]]:
    job_ids = [*fit_jobs, *plan_jobs]
    if sacct_file:
        lines = sacct_file.read_text(encoding="utf-8").splitlines()
    else:
        completed = subprocess.run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                ",".join(job_ids),
                "--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,MaxRSS,ExitCode",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        lines = completed.stdout.splitlines()
    records = []
    for line in lines:
        fields = line.split("|")
        job_pattern = "|".join(re.escape(job_id) for job_id in job_ids)
        if len(fields) >= 6 and re.fullmatch(rf"(?:{job_pattern})_\d+", fields[0]):
            records.append(
                dict(
                    zip(
                        ("job_id", "state", "elapsed_raw", "alloc_cpus", "max_rss", "exit_code"),
                        fields[:6],
                    )
                )
            )
    return records


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    fit_rows, plan_rows = read_rows(args.fit_manifest), read_rows(args.plan_manifest)
    gates: list[dict[str, Any]] = []
    descriptive: list[dict[str, Any]] = []
    incomplete = False

    def gate(name: str, passed: bool, evidence: Any, scope: str = "whole array") -> None:
        gates.append(
            {
                "name": name,
                "classification": "EVALUABLE GATE",
                "passed": bool(passed),
                "evidence": evidence,
                "failure_stop_scope": scope,
            }
        )

    try:
        runtime = verify_runtime(args.code_root)
        gate(
            "pre-run frozen runtime identity", runtime["runtime_digest"] == RUNTIME_DIGEST, runtime
        )
    except Exception as exc:
        gate("pre-run frozen runtime identity", False, str(exc))
    gate(
        "fit manifest identity",
        sha256_file(args.fit_manifest) == FIT_SHA256,
        sha256_file(args.fit_manifest),
    )
    gate(
        "plan manifest identity",
        sha256_file(args.plan_manifest) == PLAN_SHA256,
        sha256_file(args.plan_manifest),
    )
    row_results = []
    for position, (fit_row, plan_row) in enumerate(zip(fit_rows, plan_rows)):
        result: dict[str, Any] = {
            "position": position,
            "method": fit_row["method"],
            "cell": fit_cell(fit_row),
            "checks": [],
        }

        def row_gate(name: str, passed: bool, evidence: Any) -> None:
            result["checks"].append({"name": name, "passed": bool(passed), "evidence": evidence})

        receipt_path = (
            args.run_root
            / "fit_receipts"
            / fit_cell(fit_row)
            / fit_row["method"]
            / "fit_receipt.json"
        )
        if not receipt_path.is_file():
            incomplete = True
            row_gate("fit receipt exists", False, str(receipt_path))
            row_results.append(result)
            continue
        receipt = guarded_json(receipt_path)
        row_gate(
            "fit receipt identity",
            receipt.get("method") == fit_row["method"]
            and receipt.get("cell") == fit_cell(fit_row)
            and receipt.get("return_fields_opened") is False,
            {"method": receipt.get("method"), "cell": receipt.get("cell")},
        )
        row_gate(
            "complete-episode overshoot",
            int(receipt.get("target_rows", -1)) == 4000
            and 0 <= int(receipt.get("overshoot_rows", -1)) < 25,
            {
                k: receipt.get(k)
                for k in ("target_rows", "actual_rows", "overshoot_rows", "episode_count")
            },
        )
        artifacts = artifact_root(args.run_root, plan_row)
        if artifacts is None:
            incomplete = True
            row_gate("unique plan artifact exists", False, None)
            row_results.append(result)
            continue
        fit_payload = guarded_json(artifacts / "faithful_fit.json")
        privacy = guarded_json(artifacts / "privacy_audit.json")
        planners = guarded_json(artifacts / "planner_provenance.json")
        fits = fit_payload.get("fits", [])
        objectives = [item.get("objective") for item in fits]
        row_gate(
            "privacy passed",
            privacy.get("status") == "passed" and not privacy.get("forbidden_name_hits"),
            privacy.get("status"),
        )
        row_gate(
            "registered method implementation",
            fit_payload.get("method_impl_version") == METHODS[fit_row["method"]],
            fit_payload.get("method_impl_version"),
        )
        row_gate(
            "finite selected objectives",
            bool(fits)
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in objectives),
            objectives,
        )
        expected_count = int(fit_row["candidate_count"])
        row_gate(
            "registered candidate count",
            len(fits) == expected_count == int(fit_payload.get("candidate_count", -1)),
            len(fits),
        )
        if fit_row["method"].startswith("plus_"):
            forms: dict[str, int] = {}
            persistence: dict[str, int] = {}
            for item in fits:
                forms[item["form"]] = forms.get(item["form"], 0) + 1
                if item["form"] == "regime":
                    persistence[f"{float(item['fixed_regime_persistence']):.2f}"] = (
                        persistence.get(f"{float(item['fixed_regime_persistence']):.2f}", 0) + 1
                    )
            row_gate(
                "PLUS family allocation",
                forms == {"ricker": 4, "allee": 4, "theta": 4, "regime": 4}
                and persistence == {"0.80": 1, "0.90": 2, "0.97": 1},
                {"forms": forms, "regime_persistence": persistence},
            )
        else:
            row_gate(
                "MOOR single Ricker",
                len(fits) == 1 and fits[0].get("form") == "ricker",
                [item.get("form") for item in fits],
            )
        receipt_params = list(receipt.get("parameter_hashes", []))
        fit_params = [item.get("parameter_hash") for item in fits]
        receipt_transitions = {item.get("transition_data_hash") for item in fits}
        row_gate(
            "receipt/model/transition consistency",
            receipt_params == fit_params
            and receipt.get("transition_data_hash") in receipt_transitions
            and len(receipt_transitions) == 1,
            {
                "parameter_hashes_match": receipt_params == fit_params,
                "transition_hashes": sorted(str(x) for x in receipt_transitions),
            },
        )
        regime_hashes = {item.get("regime_law_hash") for item in fits}
        row_gate(
            "regime-law consistency",
            None not in regime_hashes and len(regime_hashes) == 1,
            sorted(str(x) for x in regime_hashes),
        )
        planner_items = planners.get("planners", [])
        pomdp_metadata = [
            guarded_json(path) for path in sorted(artifacts.glob("pomdp_model_*.json"))
        ]
        row_gate(
            "planner/POMDP consistency",
            len(planner_items) == len(fits) == len(pomdp_metadata)
            and all(
                not x.get("external_invocation") and x.get("name") == "pbvi" for x in planner_items
            )
            and [x.get("model_hash") for x in planner_items]
            == [x.get("model_hash") for x in pomdp_metadata],
            {"planner_count": len(planner_items), "pomdp_count": len(pomdp_metadata)},
        )
        policy_path = artifacts / "pbvi_policy_diagnostics.npz"
        legal = False
        shape = None
        if policy_path.is_file():
            with np.load(policy_path, allow_pickle=False) as data:
                values = np.asarray(data["last_action_values"])
                shape = list(values.shape)
                legal = (
                    values.shape == (expected_count, int(fit_row["num_actions"]))
                    and np.all(np.isfinite(values))
                    and np.all(
                        (np.argmax(values, axis=1) >= 0)
                        & (np.argmax(values, axis=1) < int(fit_row["num_actions"]))
                    )
                )
        row_gate("finite PBVI values and legal argmax", legal, shape)
        descriptive.append(
            {
                "position": position,
                "criterion": "4/6/7/8 observable diagnostics",
                "selected_gradient_norms": [x.get("selected_gradient_norm") for x in fits],
                "start_objectives": [x.get("start_objectives") for x in fits],
                "objective_trace_lengths": [
                    [len(t) for t in x.get("start_objective_traces", [])] for x in fits
                ],
                "holdout_normalized_survey_sse": [
                    x.get("holdout_normalized_survey_sse") for x in fits
                ],
                "candidate_parameter_hashes": fit_params,
            }
        )
        row_results.append(result)
    gate(
        "all registered artifact rows complete",
        len(row_results) == 64 and all(len(item["checks"]) >= 8 for item in row_results),
        {"rows_checked": len(row_results)},
        "whole array completion gate",
    )
    records = slurm_records(args.fit_job_id, args.plan_job_id, args.sacct_file)
    task_records = {item["job_id"]: item for item in records}
    tasks_per_job = 64 if len(args.fit_job_id) == 1 else 32
    expected_ids = {
        f"{job_id}_{i}"
        for job_id in [*args.fit_job_id, *args.plan_job_id]
        for i in range(tasks_per_job)
    }
    complete_ids = {
        job
        for job, item in task_records.items()
        if item["state"].split()[0] == "COMPLETED" and item["exit_code"].startswith("0:0")
    }
    billed = sum(
        float(item["elapsed_raw"] or 0) * float(item["alloc_cpus"] or 0) / 3600 for item in records
    )
    rss_values = [value for item in records if (value := parse_rss_mb(item["max_rss"])) is not None]
    gate(
        "Slurm completion and successful exits",
        expected_ids == complete_ids,
        {
            "expected": 128,
            "completed_successfully": len(expected_ids & complete_ids),
            "missing_or_failed": sorted(expected_ids - complete_ids),
        },
        "whole array completion gate",
    )
    gate(
        "approved allocated-core-hour ceiling",
        billed <= 550.0,
        {
            "allocated_core_hours": billed,
            "warning_threshold": 400,
            "protective_stop": 500,
            "hard_ceiling": 550,
        },
    )
    gate(
        "per-task memory ceiling",
        not rss_values or max(rss_values) <= 4096,
        {"max_rss_mb": max(rss_values) if rss_values else None, "limit_mb": 4096},
        "one cell",
    )
    any_fail = any(not item["passed"] for item in gates) or any(
        not check["passed"] for row in row_results for check in row["checks"]
    )
    if incomplete or len(records) < 128:
        decision = "INCOMPLETE"
    elif any_fail:
        decision = "FAIL_STRUCTURAL_ACCEPTANCE"
    else:
        decision = "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
    return {
        "acceptance_version": VERSION,
        "amendment_provenance": {
            "amendment_time": "2026-07-19T14:16:11+10:00",
            "reason": "original table required diagnostics not emitted by frozen runtime",
            "original_table_preserved": "docs/fix_implement_ecology_baseline/PHASE2_PREREGISTRATION_AND_BUDGET_PACKAGE.md",
            "jobs_submitted_before_amendment": False,
            "returns_inspected_before_amendment": False,
        },
        "decision": decision,
        "limited_structural_acceptance_passed": decision == "PASS_LIMITED_STRUCTURAL_ACCEPTANCE",
        "evaluated_gates": gates,
        "original_vs_amended_criterion_mapping": CRITERION_MAPPING,
        "row_results": row_results,
        "descriptive_only": descriptive,
        "not_evaluable": NOT_EVALUABLE,
        "limitations": [
            "not full scientific acceptance",
            "optimizer convergence and boundary occupancy are not fully observable",
            "run-specific posterior behaviour and PBVI repeatability are not fully observable",
            "does not authorize the 288-cell experiment",
            "4,000-transition adequacy remains unresolved",
            "safe-mode results remain information-limited",
        ],
        "return_fields_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--fit-job-id", action="append", required=True)
    parser.add_argument("--plan-job-id", action="append", required=True)
    parser.add_argument("--sacct-file", type=Path)
    args = parser.parse_args()
    payload = evaluate(args)
    output = args.run_root / "acceptance.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "acceptance_path": str(output),
                "decision": payload["decision"],
                "return_fields_opened": False,
            },
            indent=2,
        )
    )
    if payload["decision"] == "FAIL_STRUCTURAL_ACCEPTANCE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
