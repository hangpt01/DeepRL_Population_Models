#!/usr/bin/env python3
"""Phase 5: remediated, no-return E1 V1--V5 validation instrumentation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
for entry in (str(REPO_ROOT), str(ECOLOGY_SRC), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import e1_phase2_core_validation as core  # noqa: E402
from e1_phase1_parity import (  # noqa: E402
    LIKELIHOOD_FLOOR,
    POSTERIOR_L1_TOLERANCE,
    RegisteredPOMDP,
    diagnostic_wrapper_code_digest,
    stable_model_log_likelihood,
)
from real_ecology_benchmark.faithful_pomdp import (  # noqa: E402
    CandidateBelief,
    CandidatePOMDP,
)


SCHEMA = "e1_phase5_remediated_v1_v5_v1"
CONTROLLING_DIGEST = "c44bc6d514c346fd190a0c60b1bf107b6f250fe0834a6fa7a45674c845983f49"
CONTROLLING_RECEIPT = (
    REPO_ROOT
    / ".verification"
    / "e1_py310_numpy226"
    / "phase2_core"
    / "PHASE2_CORE_PHASE3_RECEIPT.json"
)
PHASE1_RECEIPT = (
    REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase1" / "E1_RECEIPT.json"
)
DEFAULT_OUTPUT = REPO_ROOT / ".verification" / "e1_phase5_remediation" / "v1_v5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Phase 5 output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Phase 5 temporary output collision: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def assert_environment() -> dict[str, str]:
    actual = (platform.python_version(), np.__version__)
    if actual != ("3.10.14", "2.2.6"):
        raise AssertionError(f"wrong Phase 5 environment: {actual}")
    return {
        "python": actual[0],
        "numpy": actual[1],
        "executable": str(Path(sys.executable).resolve()),
    }


def empty_route_counts() -> dict[str, int | float]:
    return {
        "calls": 0,
        "finite_log_evidence": 0,
        "structural_impossible_support": 0,
        "numerical_underflow": 0,
        "old_density_evidence_at_or_below_floor": 0,
        "uniform_fallback": 0,
        "predicted_fallback": 0,
        "non_normalized_posterior": 0,
        "nan_or_infinity": 0,
        "maximum_normalization_error": 0.0,
    }


class Phase5Counters:
    def __init__(self) -> None:
        self.runtime_filtering_calls = 0
        self.identity_conditioning_calls = 0
        self.diagnostic_representative_observation_calls = 0
        self.base_candidate_update_calls = 0
        self.frozen_initial_belief_calls = 0
        self.frozen_representative_observation_calls = 0
        self.runtime = empty_route_counts()
        self.lookahead = empty_route_counts()

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_filtering_calls": self.runtime_filtering_calls,
            "diagnostic_identity_conditioning_calls": self.identity_conditioning_calls,
            "diagnostic_representative_observation_calls": (
                self.diagnostic_representative_observation_calls
            ),
            "base_CandidatePOMDP_update_calls": self.base_candidate_update_calls,
            "frozen_initial_belief_calls": self.frozen_initial_belief_calls,
            "frozen_representative_observations_calls": (
                self.frozen_representative_observation_calls
            ),
            "runtime_repaired_update": dict(self.runtime),
            "pbvi_lookahead_repaired_update": dict(self.lookahead),
        }


class RemediatedTrueStatePOMDP(core.TrueStateE1POMDP):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.phase5 = Phase5Counters()

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        self.phase5.identity_conditioning_calls += 1
        return super().update(belief, action, observation)

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.phase5.diagnostic_representative_observation_calls += 1
        return super().representative_observations(predicted, branch_count)


class RemediatedNoisyStatePOMDP(core.NoisyStateE1POMDP):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.phase5 = Phase5Counters()

    def posterior_from_prediction(
        self, predicted: np.ndarray, observation: float
    ) -> tuple[np.ndarray, float, bool]:
        posterior, log_evidence, impossible = super().posterior_from_prediction(
            predicted, observation
        )
        route = self.phase5.runtime if self._runtime_update_active else self.phase5.lookahead
        route["calls"] += 1
        density = self.model.observation_likelihood(
            float(observation), self.state_abundances()
        )
        old_evidence = float(np.dot(predicted, density))
        log_likelihood = np.tile(
            stable_model_log_likelihood(
                self.model, float(observation), self.abundance_grid
            ),
            self.regime_count,
        )
        positive = predicted > 0.0
        numerical_underflow = bool(
            np.any(positive & (density == 0.0) & np.isfinite(log_likelihood))
        )
        normalization_error = abs(float(np.sum(posterior)) - 1.0)
        uniform = np.full(len(posterior), 1.0 / len(posterior))
        uniform_fallback = bool(impossible and np.array_equal(posterior, uniform))
        predicted_fallback = bool(impossible and np.array_equal(posterior, predicted))
        unexpected_nonfinite = bool(
            np.any(~np.isfinite(predicted))
            or np.any(~np.isfinite(posterior))
            or math.isnan(float(log_evidence))
            or float(log_evidence) == math.inf
        )
        route["finite_log_evidence"] += int(math.isfinite(float(log_evidence)))
        route["structural_impossible_support"] += int(impossible)
        route["numerical_underflow"] += int(numerical_underflow)
        route["old_density_evidence_at_or_below_floor"] += int(
            old_evidence <= LIKELIHOOD_FLOOR or not math.isfinite(old_evidence)
        )
        route["uniform_fallback"] += int(uniform_fallback)
        route["predicted_fallback"] += int(predicted_fallback)
        route["non_normalized_posterior"] += int(normalization_error > 1.0e-12)
        route["nan_or_infinity"] += int(unexpected_nonfinite)
        route["maximum_normalization_error"] = max(
            float(route["maximum_normalization_error"]), normalization_error
        )
        return posterior, log_evidence, impossible


class RemediatedTrueStatePolicy(core.TrueStatePolicy):
    def act(self, belief: Any, observation: float) -> int:
        self.pomdp.phase5.runtime_filtering_calls += 1
        return super().act(belief, observation)


class RemediatedNoisyStatePolicy(core.NoisyStatePolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.used_initial_belief: CandidateBelief | None = None
        self.initial_counter_snapshot: dict[str, Any] | None = None

    def act(self, belief: Any, observation: float) -> int:
        was_none = self.internal_belief is None
        action = super().act(belief, observation)
        if was_none:
            if self.internal_belief is None:
                raise AssertionError("initial belief was not constructed")
            self.used_initial_belief = CandidateBelief(
                self.internal_belief.probabilities.copy(),
                self.internal_belief.capacity,
                self.internal_belief.previous_observation,
                self.internal_belief.current_observation,
                self.internal_belief.timestep,
            )
            self.initial_counter_snapshot = self.pomdp.phase5.as_dict()
        return action


class FrozenBaseSpies:
    """Temporary call-site spies restored before receipt construction."""

    def __init__(self) -> None:
        self.original_update = CandidatePOMDP.update
        self.original_initial = CandidatePOMDP.initial_belief
        self.original_representative = CandidatePOMDP.representative_observations

    def __enter__(self) -> "FrozenBaseSpies":
        original_update = self.original_update
        original_initial = self.original_initial
        original_representative = self.original_representative

        def update_spy(instance: Any, *args: Any, **kwargs: Any):
            if hasattr(instance, "phase5"):
                instance.phase5.base_candidate_update_calls += 1
            return original_update(instance, *args, **kwargs)

        def initial_spy(instance: Any, *args: Any, **kwargs: Any):
            if hasattr(instance, "phase5"):
                instance.phase5.frozen_initial_belief_calls += 1
            return original_initial(instance, *args, **kwargs)

        def representative_spy(instance: Any, *args: Any, **kwargs: Any):
            if hasattr(instance, "phase5"):
                instance.phase5.frozen_representative_observation_calls += 1
            return original_representative(instance, *args, **kwargs)

        CandidatePOMDP.update = update_spy
        CandidatePOMDP.initial_belief = initial_spy
        CandidatePOMDP.representative_observations = representative_spy
        return self

    def __exit__(self, *_args: Any) -> None:
        CandidatePOMDP.update = self.original_update
        CandidatePOMDP.initial_belief = self.original_initial
        CandidatePOMDP.representative_observations = self.original_representative


def build_cells() -> dict[str, dict[str, Any]]:
    cells = core.build_arms()
    config = core.planner_config()
    for cid, cell in cells.items():
        registered = cell["policies"]["A3"].pomdp.model
        fitted = cell["policies"]["A4"].pomdp.model
        reg_context = core.method_context(cell["actions"], cell["sigma"], registered.survey_scale)
        fit_context = core.method_context(cell["actions"], cell["sigma"], fitted.survey_scale)
        seed = cell["planning_seed"]
        a1 = RemediatedTrueStatePOMDP(
            registered, reg_context, config, seed, cell["env_cfg"], "A1"
        )
        a2 = RemediatedTrueStatePOMDP(
            fitted, fit_context, config, seed, cell["env_cfg"], "A2"
        )
        a3 = RemediatedNoisyStatePOMDP(
            registered, reg_context, config, seed, cell["env_cfg"], "A3", True
        )
        a4 = RemediatedNoisyStatePOMDP(
            fitted, fit_context, config, seed, cell["env_cfg"], "A4", False
        )
        if a2.model is not a4.model:
            raise AssertionError("remediated A2/A4 model object not shared")
        cell["policies"] = {
            "A1": RemediatedTrueStatePolicy("A1", a1, seed),
            "A2": RemediatedTrueStatePolicy("A2", a2, seed),
            "A3": RemediatedNoisyStatePolicy("A3", a3, seed),
            "A4": RemediatedNoisyStatePolicy("A4", a4, seed),
        }
    return cells


def validate_real_counters(
    cells: dict[str, dict[str, Any]], episodes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cid, cell in cells.items():
        result[cid] = {}
        for arm, policy in cell["policies"].items():
            pomdp = policy.pomdp
            counts = pomdp.phase5.as_dict()
            steps = episodes[cid][arm]["steps"]
            if arm in {"A1", "A2"}:
                if counts["runtime_filtering_calls"] != steps:
                    raise AssertionError(f"{cid}/{arm} runtime filtering not counted")
                if counts["diagnostic_identity_conditioning_calls"] <= 0:
                    raise AssertionError(f"{cid}/{arm} PBVI identity calls absent")
                if counts["diagnostic_representative_observation_calls"] <= 0:
                    raise AssertionError(f"{cid}/{arm} representative calls absent")
                if counts["frozen_initial_belief_calls"] != 0:
                    raise AssertionError(f"{cid}/{arm} unexpectedly called frozen prior")
            else:
                runtime = counts["runtime_repaired_update"]
                lookahead = counts["pbvi_lookahead_repaired_update"]
                if runtime["calls"] != steps or lookahead["calls"] <= 0:
                    raise AssertionError(f"{cid}/{arm} repaired update accounting failed")
                if runtime["calls"] != pomdp.deployed_logspace_updates:
                    raise AssertionError(f"{cid}/{arm} runtime counter mismatch")
                if lookahead["calls"] != pomdp.lookahead_logspace_updates:
                    raise AssertionError(f"{cid}/{arm} lookahead counter mismatch")
                for route_name, route in (("runtime", runtime), ("lookahead", lookahead)):
                    failures = {
                        key: route[key]
                        for key in (
                            "structural_impossible_support",
                            "uniform_fallback",
                            "predicted_fallback",
                            "non_normalized_posterior",
                            "nan_or_infinity",
                        )
                        if route[key] != 0
                    }
                    if failures:
                        raise AssertionError(f"{cid}/{arm}/{route_name} failures: {failures}")
                    if route["finite_log_evidence"] != route["calls"]:
                        raise AssertionError(f"{cid}/{arm}/{route_name} nonfinite evidence")
            if counts["base_CandidatePOMDP_update_calls"] != 0:
                raise AssertionError(f"{cid}/{arm} called frozen base update")
            result[cid][arm] = {
                "status": "PASS",
                "steps": steps,
                **counts,
            }
    return {"status": "PASS", "cells": result}


def initial_prior_evidence(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cid, cell in cells.items():
        result[cid] = {}
        for arm, policy in cell["policies"].items():
            if arm in {"A1", "A2"}:
                result[cid][arm] = {
                    "initial_belief_called": False,
                    "used_initial_belief": None,
                    "frozen_initial_belief_calls": policy.pomdp.phase5.frozen_initial_belief_calls,
                }
                continue
            belief = policy.used_initial_belief
            if belief is None or policy.initial_counter_snapshot is None:
                raise AssertionError(f"{cid}/{arm} missing already-used initial belief")
            raw_grid = policy.pomdp.state_abundances() * policy.pomdp.model.survey_scale
            argmax = int(np.argmax(belief.probabilities))
            grid_value_raw = float(raw_grid[argmax])
            raw_mean = float(np.dot(belief.probabilities, raw_grid))
            n0 = float(cell["env_cfg"].N0)
            result[cid][arm] = {
                "initial_belief_called": True,
                "recorded_without_second_initial_belief_call": True,
                "argmax_index": argmax,
                "argmax_grid_value_raw": grid_value_raw,
                "corrected_argmax_representation_error": abs(grid_value_raw - n0),
                "raw_belief_mean": raw_mean,
                "raw_belief_mean_error": abs(raw_mean - n0),
                "probability_sum": float(belief.probabilities.sum()),
                "counter_snapshot_immediately_after_used_prior": policy.initial_counter_snapshot,
                "final_frozen_initial_belief_calls": (
                    policy.pomdp.phase5.frozen_initial_belief_calls
                ),
            }
    return result


def semantic_identity_check(
    controlling: dict[str, Any], cells: dict[str, dict[str, Any]],
    hashes: dict[str, Any], episodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[str] = []
    checked = 0
    for cid in sorted(cells):
        for arm in core.ARMS:
            old_row = controlling["hash_and_counter_table"][cid][arm]
            new_row = hashes["arms"][cid][arm]
            for key in (
                "model_parameter_hash", "transition_kernel_hash",
                "abundance_grid_hash", "capacity_grid_hash", "planning_seed",
                "survey_scale",
            ):
                checked += 1
                if old_row[key] != new_row[key]:
                    mismatches.append(f"{cid}/{arm}/{key}")
            checked += 1
            if (
                controlling["validation_episodes"][cid][arm]["actions_hash"]
                != episodes[cid][arm]["actions_hash"]
            ):
                mismatches.append(f"{cid}/{arm}/actions_hash")
    if mismatches:
        raise AssertionError(f"semantic/action drift: {mismatches}")
    if hashes["actual_registered_transition_provenance"] != controlling[
        "actual_registered_transition_provenance"
    ]:
        raise AssertionError("registered transition provenance drift")
    return {
        "status": "PASS",
        "comparisons": checked,
        "mismatches": mismatches,
        "all_sixteen_action_hashes_unchanged": True,
        "registered_transition_provenance_unchanged": True,
        "accepted_wrapper_digest_unchanged": (
            diagnostic_wrapper_code_digest() == core.ACCEPTED_WRAPPER_DIGEST
        ),
        "reward_route": "explicit_source_true_reward",
        "filter_route": "accepted repaired log-space implementation wrapped only by counters",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    environment = assert_environment()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Phase 5 output namespace is not empty: {args.output}")
    with CONTROLLING_RECEIPT.open("r", encoding="utf-8") as handle:
        controlling = json.load(handle)
    if controlling.get("build_validation_digest") != CONTROLLING_DIGEST:
        raise AssertionError("controlling V1--V5 digest mismatch")
    with PHASE1_RECEIPT.open("r", encoding="utf-8") as handle:
        phase1 = json.load(handle)
    if phase1.get("overall_status") != "PASS":
        raise AssertionError("declared-environment Phase 1 receipt is not passing")

    started_utc = now()
    started = time.perf_counter()
    cells = build_cells()
    with FrozenBaseSpies():
        hashes = core.arm_hash_table(cells)
        v1 = core.validate_v1()
        episodes: dict[str, dict[str, Any]] = {}
        for cid, cell in cells.items():
            episodes[cid] = {}
            for arm in core.ARMS:
                episodes[cid][arm] = core.validation_episode(cell, arm)
        v2_semantic = core.validate_v2(cells, episodes)
        v3 = core.validate_v3(cells)
        v4 = core.validate_v4(cells, hashes)
        v5_semantic = core.validate_v5(cells)
        real_counters = validate_real_counters(cells, episodes)
        prior_evidence = initial_prior_evidence(cells)

    semantic_identity = semantic_identity_check(
        controlling, cells, hashes, episodes
    )
    validations = {
        "V1": v1,
        "V2": v2_semantic,
        "V3": v3,
        "V4": v4,
        "V5": v5_semantic,
    }
    if any(value["status"] != "PASS" for value in validations.values()):
        raise AssertionError("remediated V1--V5 semantic validation failed")
    instrumentation_source = "".join(
        inspect.getsource(item)
        for item in (
            Phase5Counters, RemediatedTrueStatePOMDP,
            RemediatedNoisyStatePOMDP, RemediatedTrueStatePolicy,
            RemediatedNoisyStatePolicy, FrozenBaseSpies,
        )
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_utc": now(),
        "controlling_build_validation_digest": CONTROLLING_DIGEST,
        "controlling_receipt_sha256": sha256_file(CONTROLLING_RECEIPT),
        "phase1_receipt_sha256": sha256_file(PHASE1_RECEIPT),
        "environment": environment,
        "validations": validations,
        "semantic_identity": semantic_identity,
        "validation_episodes": episodes,
        "real_call_site_counters": real_counters,
        "initial_prior_evidence": prior_evidence,
        "instrumentation": {
            "strategy": (
                "actual subclass entry points plus temporary frozen CandidatePOMDP "
                "method spies; spies restored before receipt construction"
            ),
            "digest": hashlib.sha256(instrumentation_source.encode()).hexdigest(),
            "base_spies_restored": (
                CandidatePOMDP.update is FrozenBaseSpies().original_update
                and CandidatePOMDP.initial_belief is FrozenBaseSpies().original_initial
                and CandidatePOMDP.representative_observations
                is FrozenBaseSpies().original_representative
            ),
            "counter_zero_interpretation": (
                "zero base-call fields are accepted only alongside exact accounting "
                "of every dispatched subclass update and installed base-method spies"
            ),
        },
        "no_return_boundary": {
            "scientific_returns_read": False,
            "evaluation_rewards_read_or_accumulated": False,
            "contrasts_or_confidence_intervals_computed": False,
            "registered_evaluation_seed_artifacts_read": False,
        },
        "execution": {
            "started_utc": started_utc,
            "finished_utc": now(),
            "elapsed_seconds": time.perf_counter() - started,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    digest_payload = dict(payload)
    digest_payload.pop("created_utc")
    digest_payload.pop("execution")
    payload["remediation_digest"] = canonical_digest(digest_payload)
    write_json_new(args.output / "PHASE5_REMEDIATED_V1_V5_RECEIPT.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "remediation_digest": payload["remediation_digest"],
        "elapsed_seconds": payload["execution"]["elapsed_seconds"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
