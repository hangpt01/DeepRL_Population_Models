#!/usr/bin/env python3
"""Prospective tiger-only E1 grid-161 replication using the audited fox runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
for entry in (str(SCRIPT_DIR), str(REPO_ROOT), str(ECOLOGY_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import e1_fox_grid161_run as base  # noqa: E402


TIGER = "Amur tiger"
SCHEMA = "e1_tiger_only_grid161_scientific_v1"
BASE_OUTPUT = REPO_ROOT / "outputs" / "e1_tiger_only_grid161"
REGISTRATION_PREFIX = "E1_TIGER_RUN_REGISTRATION_"
FOX_FORBIDDEN_K_REF = 41.0
FOX_FORBIDDEN_S_SAFE = 10.25

AUTHORISING_COMMAND = r"""RUN THE TIGER E1 REPLICATION NOW.

This is a prospective second-species extension motivated by the completed fox result. Fox returns are already known; no tiger E1 return has been opened.

Do not modify, rerun or merge the fox experiment.

SPECIES AND CELLS

Species:
- tiger, using the repository’s canonical tiger species identifier

Population model:
- Ricker

Cells:
- tiger × Ricker × observation noise 0.1
- tiger × Ricker × observation noise 0.2

ARMS
- A1: registered tiger model + true state
- A2: fitted tiger model + true state
- A3: registered tiger model + noisy observation
- A4: fitted tiger model + noisy observation

FIXED NUMERICAL CONVENTION
Use exactly the fox E1 convention: state_bins=161; PBVI horizon=5; belief_points=32; noisy-arm observation branches=7; true-state arms use exact predictive support; gamma=0.95; explicit source true reward; occupancy penalty; alpha=1.0; P=10; repaired log-space filtering for A3/A4; registered absorbing-zero convention for A1/A3; frozen fitted-model convention for A2/A4.

Derive N0, K_base, K_max, K_ref, s_safe, action table and costs, registered transition parameters, fitted artifacts, survey_scale and reset distribution from the tiger configuration. Hard stop if fox K_ref=41 or s_safe=10.25 is present in a tiger policy or reward.

REGISTRATION
Before any tiger return exists, create and hash an immutable tiger registration recording this complete command, timestamp, code/environment hashes, that fox results were already known, no directional tiger prediction is claimed, tiger is a replication/control rather than a pooled extension, no fox/tiger pooling is permitted, and results remain conditional on grid 161 and PBVI horizon 5/32 beliefs.

ENVIRONMENT AND ARCHITECTURE
Use /fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10; require Python 3.10.14, NumPy 2.2.6, Slurm partition comp, constraint EPYC9534, and one architecture only.

EPISODES
Use block seeds [7001,7051,7101,7151,7201], episodes_per_seed=4, exactly twenty paired derived episodes per tiger cell and arm, identical environmental streams across A1-A4 within cell/episode, and independent planning RNG.

RUN
Use eight tasks (two cells by four arms). Each task builds one policy, evaluates all twenty episodes, checkpoints after every episode, reuses the frozen tiger fit with zero refits, and preserves trajectories, actions, returns, counters and hashes.

OUTPUT
Write only to outputs/e1_tiger_only_grid161/. Never mix with or overwrite outputs/e1_fox_only_grid161/. Create the registration, raw episode records, task receipts, run receipt, audit manifest, logs, reproduction instructions and Claude read-only audit prompt as in the fox run.

MINIMAL PRE-RUN CHECK
Verify the tiger model/action table and reward parameters; eight 161-bin policies; A2/A4 shared fit; zero refits; finite normalized beliefs; repaired A3/A4 filter; true-state evaluation reward; twenty paired identities; EPYC9534 architecture. Do not run a grid sweep, V7 or V8.

After successful execution, arrange an afterok-dependent finalizer exactly as in the fox run. Do not statistically aggregate until an independent Claude audit returns PASS — SAFE TO AGGREGATE. Report job IDs, completion status, output path, receipts and audit prompt. Do not interpret tiger results before audit."""

REGISTERED_DEVIATION = {
    "state_bins": 161,
    "replaces_frozen_state_bins": 41,
    "reason": (
        "The prospective tiger replication adopts the completed fox E1 numerical "
        "convention before any tiger E1 return is opened."
    ),
    "conditions": [
        "Tiger E1 is conditional on the 161-bin numerical convention.",
        "It is not directly comparable with accepted PLUS/MOOR on planner budget.",
        "It does not establish tiger grid convergence.",
        "The higher-dimensional belief representation may materially increase compute.",
    ],
}

PREREGISTERED_STATEMENTS = [
    "Fox E1 results were already known before this tiger registration.",
    "No directional tiger prediction is claimed.",
    "Tiger is a prospective replication/control, not a pooled extension.",
    "No fox/tiger pooling is permitted.",
    "Results remain conditional on tiger, Ricker, state_bins=161 and PBVI horizon-5/32-belief planning.",
]


def _configure_base() -> None:
    base.SCHEMA = SCHEMA
    base.FOX = TIGER
    base.BASE_OUTPUT = BASE_OUTPUT
    base.REGISTRATION_PREFIX = REGISTRATION_PREFIX
    base.REGISTERED_DEVIATION = REGISTERED_DEVIATION
    base.PREREGISTERED_STATEMENTS = PREREGISTERED_STATEMENTS
    base.TASKS = tuple(
        (base.core.cell_id(TIGER, sigma), arm)
        for sigma in base.SIGMAS
        for arm in base.ARMS
    )


_configure_base()
TASKS = base.TASKS


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


_base_build_cell = base.build_cell


def assert_tiger_cell(cell: dict[str, Any]) -> None:
    sigma = float(cell["sigma"])
    env = cell["env_cfg"]
    expected = base.core.real_environment(
        TIGER,
        "ricker",
        observation_noise_sigma=sigma,
        process_noise_sigma=0.0,
        initial_log_sigma=0.0,
        low_start_probability=0.0,
        horizon=base.EVALUATION_HORIZON,
        expose_rk="full",
        reward_mode="safe",
        collapse_penalty=10.0,
        safety_penalty_mode="occupancy",
        alpha=1.0,
    )
    fields = (
        "population", "kind", "N0", "K_base", "K_max", "K_ref",
        "safety_threshold", "mvp_threshold", "collapse_penalty", "alpha",
    )
    for field in fields:
        if getattr(env, field) != getattr(expected, field):
            raise AssertionError(f"tiger environment drift in {field}")
    if env.population != TIGER or not cell["cell"].startswith("tiger_ricker_sigma_"):
        raise AssertionError("canonical tiger identifier/cell route absent")
    if env.K_ref == FOX_FORBIDDEN_K_REF or env.safety_threshold == FOX_FORBIDDEN_S_SAFE:
        raise AssertionError("fox reward constant present in tiger environment")
    tiger_actions = base.core.real_action_table(TIGER, "ricker", env.data_dir)
    fox_env = base.core.real_environment("Crab-eating fox", "ricker", observation_noise_sigma=sigma)
    fox_actions = base.core.real_action_table("Crab-eating fox", "ricker", fox_env.data_dir)
    if base.action_hash(cell["actions"]) != base.action_hash(tiger_actions):
        raise AssertionError("tiger action table mismatch")
    if base.action_hash(cell["actions"]) == base.action_hash(fox_actions):
        raise AssertionError("fox action table present in tiger cell")
    reward = base.core.build_reward(env)
    if reward.K_ref != env.K_ref or reward.K_ref == FOX_FORBIDDEN_K_REF:
        raise AssertionError("fox K_ref present in tiger reward")
    if cell["fitted_provenance"]["cache_key"] != base.core.FIT_CACHE_KEYS[(TIGER, sigma)]:
        raise AssertionError("wrong fitted artifact cache key")
    if cell["fitted_provenance"]["fits_executed"] != 0:
        raise AssertionError("tiger fit was recomputed")
    for arm, policy in cell["policies"].items():
        pomdp = policy.pomdp
        if pomdp.e1_env_cfg.population != TIGER:
            raise AssertionError(f"{arm}: fox/non-tiger environment in policy")
        if pomdp.e1_env_cfg.K_ref != env.K_ref or pomdp.e1_env_cfg.safety_threshold != env.safety_threshold:
            raise AssertionError(f"{arm}: tiger reward constants drift")
        if pomdp.e1_env_cfg.K_ref == FOX_FORBIDDEN_K_REF or pomdp.e1_env_cfg.safety_threshold == FOX_FORBIDDEN_S_SAFE:
            raise AssertionError(f"{arm}: fox constant present in tiger policy")
        if pomdp.e1_reward.K_ref != env.K_ref:
            raise AssertionError(f"{arm}: source reward K_ref mismatch")


def build_cell(sigma: float, requested_arms: tuple[str, ...] = base.ARMS) -> dict[str, Any]:
    cell = _base_build_cell(sigma, requested_arms)
    assert_tiger_cell(cell)
    return cell


base.build_cell = build_cell


def model_payload(model: Any) -> dict[str, Any]:
    return _jsonable(asdict(model)) | {
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameter_hash": model.parameter_hash(),
    }


def derived_tiger_configuration() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sigma in base.SIGMAS:
        cell = build_cell(sigma)
        env = cell["env_cfg"]
        result[cell["cell"]] = {
            "canonical_species_identifier": TIGER,
            "environment": {
                key: getattr(env, key)
                for key in (
                    "N0", "K_base", "K_max", "K_ref", "safety_threshold",
                    "mvp_threshold", "collapse_penalty", "alpha",
                    "safety_penalty_mode", "observation_noise_sigma",
                )
            },
            "action_table": [asdict(action) for action in cell["actions"]],
            "action_table_hash": base.action_hash(cell["actions"]),
            "registered_model": model_payload(cell["policies"]["A1"].pomdp.model),
            "fitted_model": model_payload(cell["policies"]["A2"].pomdp.model),
            "fitted_artifact": cell["fitted_provenance"],
            "A2_A4_same_fitted_object": (
                cell["policies"]["A2"].pomdp.model
                is cell["policies"]["A4"].pomdp.model
            ),
            "initial_prior_routes": {
                "A1": "true-state delta",
                "A2": "true-state delta",
                "A3": "registered nearest-bin delta at N0/survey_scale",
                "A4": "fitted base log-normal reset with genuine positive reset_log_scale",
            },
            "fox_constant_exclusion": {
                "forbidden_K_ref": FOX_FORBIDDEN_K_REF,
                "forbidden_s_safe": FOX_FORBIDDEN_S_SAFE,
                "passed": True,
            },
        }
    return result


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "e1_tiger_grid161_finalize.py",
        SCRIPT_DIR / "e1_tiger_grid161_aggregate.py",
        SCRIPT_DIR / "e1_fox_grid161_run.py",
        SCRIPT_DIR / "e1_phase1_parity.py",
        SCRIPT_DIR / "e1_phase2_core_validation.py",
        SCRIPT_DIR / "e1_phase5_remediated_validation.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "planners" / "pbvi.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "faithful_pomdp.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "envs.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "reward.py",
    ]
    return {str(path.relative_to(REPO_ROOT)): base.sha256_file(path) for path in paths}


base.code_hashes = code_hashes


def task_command(run_dir: Path, index: int) -> str:
    return (
        f"PYTHONDONTWRITEBYTECODE=1 {Path(sys.executable).resolve()} "
        f"{Path(__file__).resolve()} task --run-dir {run_dir.resolve()} --task-index {index}"
    )


base.task_command = task_command


def slurm_script_text() -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=e1tiger161
#SBATCH --partition={base.EXPECTED_PARTITION}
#SBATCH --constraint={base.EXPECTED_CONSTRAINT}
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec {Path(sys.executable).resolve()} {Path(__file__).resolve()} task --run-dir "$1" --task-index "$SLURM_ARRAY_TASK_ID"
"""


base.slurm_script_text = slurm_script_text


def finalizer_slurm_text() -> str:
    return """#!/bin/bash
#SBATCH --job-name=e1tiger161-finalize
#SBATCH --partition=comp
#SBATCH --constraint=EPYC9534
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
grep -q 'EPYC 9534' /proc/cpuinfo
/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 "$1/logs/e1_tiger_grid161_finalize.py" finalize --run-dir "$1"
/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10 "$1/logs/e1_tiger_grid161_finalize.py" seal --run-dir "$1" >/dev/null
"""


def canonical_config() -> dict[str, Any]:
    config = {
        "schema": SCHEMA,
        "species": TIGER,
        "population_model": "ricker",
        "cells": [base.core.cell_id(TIGER, sigma) for sigma in base.SIGMAS],
        "arms": {
            "A1": "registered tiger transition model + true state",
            "A2": "fitted tiger transition model + true state",
            "A3": "registered tiger transition model + noisy observation",
            "A4": "fitted tiger transition model + noisy observation",
        },
        "planner": {
            "name": "PBVI",
            "state_bins": base.STATE_BINS,
            "capacity_bins": base.CAPACITY_BINS,
            "observation_bins": base.OBSERVATION_BINS,
            "horizon": base.HORIZON,
            "belief_points": base.BELIEF_POINTS,
            "noisy_observation_branches": base.OBSERVATION_BRANCHES,
            "true_state_branch_convention": "exact predictive support and weights",
            "gamma": base.GAMMA,
        },
        "reward": {
            "route": "explicit source true reward", "mode": "safe",
            "penalty": "occupancy", "alpha": 1.0, "P": 10.0,
        },
        "environment": {"horizon": base.EVALUATION_HORIZON, "process_noise": 0.0},
        "block_seeds": list(base.BLOCK_SEEDS),
        "episodes_per_seed": base.EPISODES_PER_SEED,
        "derived_episode_identities": list(base.DERIVED_EPISODES),
        "planning_seeds_by_cell": {
            base.core.cell_id(TIGER, sigma): base.core.PLANNING_SEEDS[base.core.cell_id(TIGER, sigma)]
            for sigma in base.SIGMAS
        },
        "slurm": {"partition": "comp", "constraint": "EPYC9534", "tasks": 8},
        "registered_deviation": REGISTERED_DEVIATION,
        "derived_tiger_configuration": derived_tiger_configuration(),
        "fox_tiger_pooling_permitted": False,
    }
    return config


def reproduce_markdown(run_dir: Path, control: dict[str, Any]) -> str:
    py = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    return f"""# E1 tiger-only grid-161 reproduction commands

These commands are non-destructive. Existing immutable episode files are never replaced.

## Reconstruct each policy (pre-return checks only)

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {script} check --run-dir {run_dir}
```

## Reproduce or resume every episode

```bash
{control['submit_command']}
```

## Recompute file hashes

```bash
find {run_dir} -type f -print0 | sort -z | xargs -0 sha256sum
```

## Verify pairing and completeness without interpretation

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {script} verify --run-dir {run_dir}
```

## Aggregate only after independent audit PASS — SAFE TO AGGREGATE

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {SCRIPT_DIR / 'e1_tiger_grid161_aggregate.py'} --run-dir {run_dir} --output {run_dir / 'analysis_later'}
```
"""


base.reproduce_markdown = reproduce_markdown


def prepare() -> dict[str, Any]:
    started = time.perf_counter()
    if BASE_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing tiger namespace: {BASE_OUTPUT}")
    if not (REPO_ROOT / "outputs" / "e1_fox_only_grid161").is_dir():
        raise AssertionError("completed fox namespace missing; tiger extension context cannot be recorded")
    run_dir = BASE_OUTPUT.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    environment = base.assert_environment()
    git_before = base.git_snapshot()
    identities = base.episode_identity_payload()
    config = canonical_config()
    config_digest = base.canonical_digest(config)
    all_hashes: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    fit_pairing: dict[str, Any] = {}
    for sigma in base.SIGMAS:
        cell = build_cell(sigma)
        cid = cell["cell"]
        if cell["policies"]["A2"].pomdp.model is not cell["policies"]["A4"].pomdp.model:
            raise AssertionError(f"{cid}: A2/A4 fitted object identity failure")
        if len(cell["actions"]) != 11:
            raise AssertionError(f"{cid}: wrong tiger action count")
        fit_pairing[cid] = {
            "same_object_in_precheck": True,
            "cache_key": cell["fitted_provenance"]["cache_key"],
            "array_hash": cell["fitted_provenance"]["array_hash"],
            "fits_executed": 0,
        }
        checks[cid] = {}
        all_hashes[cid] = {}
        for arm in base.ARMS:
            policy = cell["policies"][arm]
            pomdp = policy.pomdp
            if pomdp.config.state_bins != 161:
                raise AssertionError(f"{cid}/{arm}: state_bins drift")
            if pomdp.objective_flag != "explicit_source_true_reward":
                raise AssertionError(f"{cid}/{arm}: true reward route absent")
            if arm in ("A1", "A2"):
                belief = base.true_state_belief(policy, cell)
                repaired_route = False
                branch_convention: str | int = "exact_predictive_support"
            else:
                belief = pomdp.initial_belief(float(cell["env_cfg"].N0))
                updated, _ = pomdp.runtime_update(belief, 0, float(cell["env_cfg"].N0))
                base.check_belief(updated, f"{cid}/{arm}/repaired_runtime")
                lookahead, _ = pomdp.update(belief, 0, float(cell["env_cfg"].N0))
                base.check_belief(lookahead, f"{cid}/{arm}/repaired_lookahead")
                repaired_route = isinstance(pomdp, base.p5.RemediatedNoisyStatePOMDP)
                if not repaired_route:
                    raise AssertionError(f"{cid}/{arm}: repaired filter class absent")
                branch_convention = 7
            all_hashes[cid][arm] = base.pomdp_hashes(cell, arm)
            checks[cid][arm] = {
                "policy_constructed": True,
                "state_bins": pomdp.config.state_bins,
                "belief": base.check_belief(belief, f"{cid}/{arm}/initial"),
                "repaired_logspace_filter": repaired_route,
                "true_reward_route": pomdp.objective_flag,
                "branch_convention": branch_convention,
                "planning_seed": cell["planning_seed"],
                "species": pomdp.e1_env_cfg.population,
                "K_ref": pomdp.e1_env_cfg.K_ref,
                "s_safe": pomdp.e1_env_cfg.safety_threshold,
                "fox_constants_absent": True,
            }
    for cid in all_hashes:
        if all_hashes[cid]["A2"]["fit_array_hash"] != all_hashes[cid]["A4"]["fit_array_hash"]:
            raise AssertionError(f"{cid}: A2/A4 fit hash mismatch")
        if all_hashes[cid]["A1"]["kernel_hash"] != all_hashes[cid]["A3"]["kernel_hash"]:
            raise AssertionError(f"{cid}: A1/A3 kernel mismatch")
        if all_hashes[cid]["A2"]["kernel_hash"] != all_hashes[cid]["A4"]["kernel_hash"]:
            raise AssertionError(f"{cid}: A2/A4 kernel mismatch")
    planning_seeds = set(config["planning_seeds_by_cell"].values())
    if planning_seeds.intersection(base.DERIVED_EPISODES):
        raise AssertionError("planning and evaluation seeds overlap")

    task_slurm = logs / "e1_tiger_grid161.slurm"
    finalizer_py = logs / "e1_tiger_grid161_finalize.py"
    finalizer_slurm = logs / "e1_tiger_grid161_finalize.slurm"
    base.write_text_new(task_slurm, slurm_script_text(), 0o755)
    base.write_text_new(finalizer_py, (SCRIPT_DIR / "e1_tiger_grid161_finalize.py").read_text(encoding="utf-8"), 0o555)
    base.write_text_new(finalizer_slurm, finalizer_slurm_text(), 0o755)
    submit_command = (
        f"sbatch --parsable --partition=comp --constraint=EPYC9534 --array=0-7%8 "
        f"--output={run_dir}/logs/slurm-%A_%a.out --error={run_dir}/logs/slurm-%A_%a.err "
        f"{task_slurm} {run_dir}"
    )
    task_map = {
        str(index): {
            "cell": cid,
            "arm": arm,
            "command": task_command(run_dir, index),
            "expected_hashes": all_hashes[cid][arm],
        }
        for index, (cid, arm) in enumerate(TASKS)
    }
    registration = {
        "schema": f"{SCHEMA}_registration",
        "timestamp_utc": base.utc_now(),
        "authorising_command_verbatim": AUTHORISING_COMMAND,
        "result_directory": str(run_dir),
        "repository_commit": git_before["commit"],
        "git_status_before": git_before,
        "environment": environment,
        "environment_digest": environment["environment_digest"],
        "code_hashes": code_hashes(),
        "canonical_configuration": config,
        "configuration_digest": config_digest,
        "derived_episode_identities": identities,
        "planning_rng_independent_of_evaluation_rng": True,
        "planning_seeds": config["planning_seeds_by_cell"],
        "evaluation_seeds": list(base.DERIVED_EPISODES),
        "fit_pairing": fit_pairing,
        "pre_return_checks": checks,
        "architecture_precheck": {
            "partition": "comp", "constraint": "EPYC9534",
            "one_architecture_enforced_by_all_task_commands": True,
            "runtime_cpu_assertion_required": "AMD EPYC 9534",
        },
        "task_map": task_map,
        "complete_submission_command": submit_command,
        "exact_task_commands": {key: value["command"] for key, value in task_map.items()},
        "preregistered_statements": PREREGISTERED_STATEMENTS,
        "registered_deviation": REGISTERED_DEVIATION,
        "fox_results_known_before_registration": True,
        "fox_tiger_pooling_permitted": False,
        "tiger_returns_computed_or_opened_before_registration": False,
    }
    registration_path = run_dir / f"{REGISTRATION_PREFIX}{base.stamp()}.json"
    base.write_json_new(registration_path, registration, 0o444)
    registration_hash = base.sha256_file(registration_path)
    control = {
        "schema": f"{SCHEMA}_control",
        "registration_path": str(registration_path),
        "registration_sha256": registration_hash,
        "configuration_digest": config_digest,
        "episode_identity_sha256": identities["sha256"],
        "result_directory": str(run_dir),
        "submit_command": submit_command,
        "task_map": task_map,
    }
    base.write_json_new(run_dir / base.RUN_CONTROL_NAME, control, 0o444)
    base.write_json_new(logs / "pre_run_check.json", {
        "status": "PASS", "checks": checks, "fit_pairing": fit_pairing,
        "derived_tiger_configuration": config["derived_tiger_configuration"],
        "architecture": registration["architecture_precheck"],
        "elapsed_seconds": time.perf_counter() - started,
    })
    base.write_json_new(logs / "episode_identities.json", identities)
    base.write_text_new(logs / "environment_version.txt", json.dumps(environment, indent=2, sort_keys=True) + "\n")
    base.write_text_new(logs / "git_status_before.txt", "\n".join(git_before["status_lines"]) + "\n")
    result = {
        "status": "PASS", "run_dir": str(run_dir),
        "registration_path": str(registration_path),
        "registration_sha256": registration_hash,
        "derived_episode_identities": list(base.DERIVED_EPISODES),
        "episode_identity_sha256": identities["sha256"],
        "submit_command": submit_command,
        "finalizer_slurm": str(finalizer_slurm),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def assert_registered_code(run_dir: Path) -> None:
    control = base.load_json(run_dir / base.RUN_CONTROL_NAME)
    registration = base.load_json(Path(control["registration_path"]))
    if registration["code_hashes"] != code_hashes():
        raise AssertionError("registered tiger code hashes changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("task", "verify", "check"):
        child = sub.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
        if name == "task":
            child.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "task":
        try:
            assert_registered_code(args.run_dir.resolve())
            base.run_task(args.run_dir.resolve(), args.task_index)
        except BaseException as error:
            base.failure_record(args.run_dir.resolve(), args.task_index, error)
            raise
    elif args.command == "verify":
        assert_registered_code(args.run_dir.resolve())
        base.verify(args.run_dir.resolve())
    else:
        assert_registered_code(args.run_dir.resolve())
        base.check_existing(args.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
