#!/usr/bin/env python3
"""Post-failure E1 five-grid characterisation without evaluation returns."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
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

import e1_phase2_core_validation as core  # noqa: E402
import e1_phase3_remainder as phase3  # noqa: E402
from e1_phase1_parity import (  # noqa: E402
    ABSORBING_ZERO_EQUATION_VERSION,
    FIT_CACHE_KEYS,
    diagnostic_wrapper_code_digest,
)
from real_ecology_benchmark.envs import ContinuousEcologyEnv  # noqa: E402
from real_ecology_benchmark.faithful_pomdp import CandidateBelief  # noqa: E402
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner  # noqa: E402


SCHEMA = "e1_phase5_postfailure_grid_characterisation_v1"
CONTROLLING_DIGEST = "c44bc6d514c346fd190a0c60b1bf107b6f250fe0834a6fa7a45674c845983f49"
CONTROLLING_RECEIPT = (
    REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase2_core"
    / "PHASE2_CORE_PHASE3_RECEIPT.json"
)
V6_FAILURE_RECEIPT = REPO_ROOT / ".verification" / "e1_phase3_remainder" / "V6_DISCRETISATION.json"
DEFAULT_OUTPUT = REPO_ROOT / ".verification" / "e1_phase5_remediation" / "grid_characterisation"
GRIDS = (41, 61, 81, 121, 161)
SEEDS = (63001, 63002, 63003, 63004, 63005)
ARMS = ("A3", "A4")
PRIORS = ("deployed", "barycentric_control")
PAIR_SEQUENCE = tuple(zip(GRIDS[:-1], GRIDS[1:]))
POPULATION = "Crab-eating fox"
SIGMA = 0.1
CELL_ID = "fox_ricker_sigma_0.1"
ROOTS = 50
GAMMA = 0.95
NEAR_TIE_RELATIVE_THRESHOLD = 0.01
CONTROL_LABEL = "POST-FAILURE CONTROL — PRIOR EXPLANATION ALREADY FALSIFIED AT SEED 63001"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Phase 5 output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Phase 5 temporary output collision: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def assert_environment() -> dict[str, str]:
    actual = (platform.python_version(), np.__version__)
    if actual != ("3.10.14", "2.2.6"):
        raise AssertionError(f"wrong Phase 5 environment: {actual}")
    controlling = load_json(CONTROLLING_RECEIPT)
    if controlling["build_validation_digest"] != CONTROLLING_DIGEST:
        raise AssertionError("controlling V1--V5 receipt drift")
    failure = load_json(V6_FAILURE_RECEIPT)
    if failure.get("status") != "FAILED_ACTION_SENSITIVITY":
        raise AssertionError("genuine V6 failure receipt is absent or changed")
    return {
        "python": actual[0],
        "numpy": actual[1],
        "executable": str(Path(sys.executable).resolve()),
        "controlling_receipt_sha256": sha256_file(CONTROLLING_RECEIPT),
        "v6_failure_receipt_sha256": sha256_file(V6_FAILURE_RECEIPT),
    }


def rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def ranking(values: np.ndarray) -> list[int]:
    return np.argsort(-values, kind="stable").astype(int).tolist()


def top_margin(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


class CharacterisationPOMDP(core.NoisyStateE1POMDP):
    """Accepted noisy POMDP with passive counters on the actual call sites."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.p5_calls = Counter()
        self.p5_impossible = Counter()
        self.p5_max_normalization_error = 0.0

    def initial_belief(self, observation: float) -> CandidateBelief:
        self.p5_calls["initial_belief"] += 1
        return super().initial_belief(observation)

    def transition_matrix(self, capacity: float, action: int) -> np.ndarray:
        self.p5_calls["transition_matrix"] += 1
        return super().transition_matrix(capacity, action)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        self.p5_calls["predict"] += 1
        return super().predict(belief, action)

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.p5_calls["representative_observations"] += 1
        return super().representative_observations(predicted, branch_count)

    def posterior_from_prediction(
        self, predicted: np.ndarray, observation: float
    ) -> tuple[np.ndarray, float, bool]:
        route = "runtime" if self._runtime_update_active else "lookahead"
        self.p5_calls[f"{route}_posterior"] += 1
        posterior, log_evidence, impossible = super().posterior_from_prediction(
            predicted, observation
        )
        error = abs(float(np.sum(posterior)) - 1.0)
        self.p5_max_normalization_error = max(self.p5_max_normalization_error, error)
        self.p5_impossible[route] += int(impossible)
        if impossible:
            raise AssertionError(f"{route} observation fallback/impossible support")
        if error > 1.0e-12:
            raise AssertionError(f"{route} posterior normalization failure: {error}")
        if not np.all(np.isfinite(posterior)) or not math.isfinite(float(log_evidence)):
            raise AssertionError(f"{route} posterior/evidence NaN or infinity")
        return posterior, log_evidence, impossible

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        route = "runtime" if self._runtime_update_active else "lookahead"
        self.p5_calls[f"{route}_update"] += 1
        result = super().update(belief, action, observation)
        if not np.all(np.isfinite(result[0].probabilities)):
            raise AssertionError(f"{route} update produced NaN or infinity")
        return result


def collapse_branches(observations: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    grouped: dict[float, float] = {}
    for observation, weight in zip(observations, weights):
        key = float(observation)
        grouped[key] = grouped.get(key, 0.0) + float(weight)
    collapsed = np.asarray(list(grouped.values()), dtype=np.float64)
    if abs(float(collapsed.sum()) - 1.0) > 1.0e-12 or np.any(collapsed < 0.0):
        raise AssertionError("representative branch weights invalid")
    return {
        "requested": int(len(observations)),
        "distinct": int(len(grouped)),
        "observations": list(grouped),
        "collapsed_weights": collapsed.tolist(),
        "effective_branch_count_inverse_simpson": float(1.0 / np.sum(collapsed ** 2)),
    }


class CharacterisationPlanner(phase3.AuditedPointBasedPlanner):
    """Deployed PBVI plus branch-collapse instrumentation, without route changes."""

    def action_values(self, belief: CandidateBelief) -> np.ndarray:
        records: list[dict[str, Any]] = []
        original = self.model.representative_observations

        def recorder(predicted: np.ndarray, branch_count: int):
            observations, weights = original(predicted, branch_count)
            row = collapse_branches(
                np.asarray(observations, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
            )
            row["predictive_support_size"] = int(np.count_nonzero(predicted > 0.0))
            records.append(row)
            return observations, weights

        self.model.representative_observations = recorder
        try:
            values = super().action_values(belief)
        finally:
            self.model.representative_observations = original
        graph_call_count = self.config.horizon * self.config.belief_points
        if len(records) <= graph_call_count + self.model.context.num_actions:
            raise AssertionError("representative-observation instrumentation incomplete")
        graph = records[:graph_call_count]
        backup = records[graph_call_count:]
        root_actions = records[-self.model.context.num_actions:]
        for action, row in enumerate(root_actions):
            row["action"] = action
        histogram = Counter(row["distinct"] for row in records)
        signature_histogram = Counter(
            json.dumps(row["collapsed_weights"], separators=(",", ":")) for row in records
        )
        effective = np.asarray(
            [row["effective_branch_count_inverse_simpson"] for row in records],
            dtype=np.float64,
        )
        self.last_audit.update({
            "representative_observation_requested_branch_counts": sorted(
                {row["requested"] for row in records}
            ),
            "representative_observation_call_count": len(records),
            "distinct_branch_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
            "branch_weight_signature_histogram": dict(sorted(signature_histogram.items())),
            "effective_branch_count_inverse_simpson_min": float(effective.min()),
            "effective_branch_count_inverse_simpson_mean": float(effective.mean()),
            "effective_branch_count_inverse_simpson_max": float(effective.max()),
            "reachable_graph_support_coverage": float(
                np.mean([row["distinct"] / row["requested"] for row in graph])
            ),
            "backup_support_coverage": float(
                np.mean([row["distinct"] / row["requested"] for row in backup])
            ),
            "root_action_branches": root_actions,
            "predictive_support_size_min": min(row["predictive_support_size"] for row in records),
            "predictive_support_size_max": max(row["predictive_support_size"] for row in records),
        })
        return values


def planner_config(state_bins: int) -> Any:
    config = replace(core.planner_config(), state_bins=int(state_bins))
    config.validate()
    expected = {
        "state_bins": state_bins,
        "capacity_bins": 9,
        "observation_bins": 41,
        "transition_samples": 256,
        "observation_samples": 256,
        "belief_points": 32,
        "observation_branches": 7,
        "horizon": 5,
    }
    actual = asdict(config)
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError(f"planner configuration drift: {key}")
    return config


def build_pomdp(base: dict[str, Any], arm: str, state_bins: int) -> CharacterisationPOMDP:
    source = base["policies"][arm].pomdp
    model = source.model
    context = core.method_context(base["actions"], SIGMA, model.survey_scale)
    pomdp = CharacterisationPOMDP(
        model,
        context,
        planner_config(state_bins),
        base["planning_seed"],
        base["env_cfg"],
        arm,
        registered_prior=(arm == "A3"),
    )
    if pomdp.model.parameter_hash() != source.model.parameter_hash():
        raise AssertionError("model treatment drift")
    return pomdp


def barycentric_prior(pomdp: CharacterisationPOMDP, observation: float, n0: float) -> CandidateBelief:
    calls_before = int(pomdp.p5_calls["initial_belief"])
    raw_grid = pomdp.abundance_grid * pomdp.model.survey_scale
    if n0 < raw_grid[0] or n0 > raw_grid[-1]:
        raise AssertionError("N0 outside barycentric grid")
    upper = int(np.searchsorted(raw_grid, n0, side="left"))
    probabilities = np.zeros(pomdp.hidden_count, dtype=np.float64)
    if upper < len(raw_grid) and float(raw_grid[upper]) == float(n0):
        probabilities[upper] = 1.0
    else:
        if upper == 0 or upper == len(raw_grid):
            raise AssertionError("barycentric bracket missing")
        lower = upper - 1
        fraction = (n0 - raw_grid[lower]) / (raw_grid[upper] - raw_grid[lower])
        probabilities[lower] = 1.0 - fraction
        probabilities[upper] = fraction
    if pomdp.regime_count == 2:
        probabilities = np.concatenate((0.5 * probabilities, 0.5 * probabilities))
    belief = CandidateBelief(
        probabilities,
        pomdp.model.initial_capacity,
        float(observation),
        float(observation),
        0,
    )
    raw_states = pomdp.state_abundances() * pomdp.model.survey_scale
    if abs(float(np.sum(belief.probabilities)) - 1.0) > 1.0e-15:
        raise AssertionError("barycentric mass does not sum to one")
    if np.any(belief.probabilities < 0.0):
        raise AssertionError("negative barycentric weight")
    if abs(float(np.dot(belief.probabilities, raw_states)) - n0) > 1.0e-12:
        raise AssertionError("barycentric raw mean is not N0")
    if int(pomdp.p5_calls["initial_belief"]) != calls_before:
        raise AssertionError("control prior accidentally called base/deployed initial_belief")
    return belief


def control_prior_unit_tests(base: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for arm in ARMS:
        for bins in GRIDS:
            pomdp = build_pomdp(base, arm, bins)
            belief = barycentric_prior(
                pomdp, float(base["env_cfg"].N0), float(base["env_cfg"].N0)
            )
            rows[f"{arm}_{bins}"] = {
                "mass_sum": float(belief.probabilities.sum()),
                "minimum_weight": float(belief.probabilities.min()),
                "raw_mean": float(np.dot(
                    belief.probabilities,
                    pomdp.state_abundances() * pomdp.model.survey_scale,
                )),
                "base_or_deployed_initial_belief_calls": int(pomdp.p5_calls["initial_belief"]),
            }
    test = np.asarray([0.0, 1.0, 2.0])
    exact = np.zeros(3)
    target = 1.0
    upper = int(np.searchsorted(test, target, side="left"))
    if test[upper] == target:
        exact[upper] = 1.0
    if not np.array_equal(exact, np.asarray([0.0, 1.0, 0.0])):
        raise AssertionError("exact grid coincidence did not produce delta")
    return {
        "status": "PASS",
        "control_label": CONTROL_LABEL,
        "mass_sum_tolerance": 1.0e-15,
        "raw_mean_tolerance": 1.0e-12,
        "exact_grid_coincidence_delta": True,
        "base_log_normal_prior_accidentally_called": False,
        "rows": rows,
    }


def initial_belief(
    pomdp: CharacterisationPOMDP, prior: str, observation: float, n0: float
) -> CandidateBelief:
    if prior == "deployed":
        return pomdp.initial_belief(observation)
    if prior == "barycentric_control":
        return barycentric_prior(pomdp, observation, n0)
    raise ValueError(prior)


def action_table_digest(actions: tuple[Any, ...]) -> str:
    rows = []
    for action in actions:
        rows.append(jsonable(asdict(action) if hasattr(action, "__dataclass_fields__") else vars(action)))
    return canonical_digest(rows)


def run_identity(
    pomdp: CharacterisationPOMDP, base: dict[str, Any], arm: str, bins: int, prior: str
) -> dict[str, Any]:
    raw_grid = pomdp.abundance_grid * pomdp.model.survey_scale
    identity = {
        "arm": arm,
        "state_bins": bins,
        "prior_convention": prior,
        "control_label": CONTROL_LABEL if prior == "barycentric_control" else None,
        "model_parameter_hash": pomdp.model.parameter_hash(),
        "abundance_grid_hash": array_hash(pomdp.abundance_grid),
        "raw_abundance_grid_hash": array_hash(raw_grid),
        "raw_grid": raw_grid.tolist(),
        "raw_grid_endpoints": [float(raw_grid[0]), float(raw_grid[-1])],
        "positive_grid_spacing_convention": "geomspace(1e-4, max(capacity_ceiling*1.5,1.0), state_bins-1)",
        "capacity_grid_hash": array_hash(core.capacity_grid(pomdp)),
        "capacity_grid": core.capacity_grid(pomdp).tolist(),
        "planner_config": asdict(pomdp.config),
        "planning_seed": pomdp.seed,
        "validation_seed_is_query_bank_seed": True,
        "action_table_hash": action_table_digest(base["actions"]),
        "action_cost_hash": array_hash(np.asarray(pomdp.context.action_costs)),
        "reward_route": pomdp.objective_flag,
        "initial_prior_route": (
            "registered nearest-bin delta" if prior == "deployed" and arm == "A3"
            else "fitted genuine log-normal reset prior conditioned on observation"
            if prior == "deployed"
            else "mean-preserving two-bin barycentric diagnostic control"
        ),
        "observation_update_route": "accepted repaired log-space CandidatePOMDP route",
        "survey_scale": pomdp.model.survey_scale,
        "absorbing_zero_equation_version": (
            ABSORBING_ZERO_EQUATION_VERSION if arm == "A3" else "fitted_kernel_no_added_guard"
        ),
        "diagnostic_wrapper_code_digest": diagnostic_wrapper_code_digest(),
        "characterisation_wrapper_digest": hashlib.sha256(
            (inspect.getsource(CharacterisationPOMDP) + inspect.getsource(CharacterisationPlanner)).encode()
        ).hexdigest(),
        "fit_cache_key": FIT_CACHE_KEYS[(POPULATION, SIGMA)] if arm == "A4" else None,
    }
    identity["identity_digest"] = canonical_digest(identity)
    return identity


def counter_snapshot(pomdp: CharacterisationPOMDP, planner: CharacterisationPlanner) -> dict[str, Any]:
    return {
        "pomdp": dict(pomdp.p5_calls),
        "true_reward_calls": int(pomdp.true_reward_calls),
        "kernel_build_count": int(pomdp.kernel_build_count),
        "planner_invocations": int(planner.invocation_count),
    }


def counter_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    keys = set(after["pomdp"]) | set(before["pomdp"])
    return {
        "pomdp": {key: int(after["pomdp"].get(key, 0) - before["pomdp"].get(key, 0)) for key in sorted(keys)},
        "true_reward_calls": int(after["true_reward_calls"] - before["true_reward_calls"]),
        "kernel_build_count": int(after["kernel_build_count"] - before["kernel_build_count"]),
        "planner_invocations": int(after["planner_invocations"] - before["planner_invocations"]),
    }


def evaluate_root(
    pomdp: CharacterisationPOMDP,
    planner: CharacterisationPlanner,
    belief: CandidateBelief,
    truth: float,
    step: int,
    initial_snap_error: float,
    identity_digest: str,
    query_bank_digest: str,
) -> tuple[dict[str, Any], int]:
    normalization_error = abs(float(belief.probabilities.sum()) - 1.0)
    if normalization_error > 1.0e-12 or not np.all(np.isfinite(belief.probabilities)):
        raise AssertionError("root belief fallback/normalization/finite failure")
    before = counter_snapshot(pomdp, planner)
    started = time.perf_counter()
    q = np.asarray(planner.action_values(belief), dtype=np.float64)
    elapsed = time.perf_counter() - started
    if q.shape != (11,) or not np.all(np.isfinite(q)):
        raise AssertionError("planner Q vector is not eleven finite values")
    action = int(np.argmax(q))
    raw_states = pomdp.state_abundances() * pomdp.model.survey_scale
    raw_mean = float(np.dot(belief.probabilities, raw_states))
    predicted = pomdp.predict(belief, action)
    after = counter_snapshot(pomdp, planner)
    if not np.all(np.isfinite(predicted)) or abs(float(predicted.sum()) - 1.0) > 1.0e-12:
        raise AssertionError("selected-action prediction invalid")
    record = {
        "root": step,
        "q_values": q.tolist(),
        "mean_q": float(q.mean()),
        "maximum_q": float(q.max()),
        "action_centred_advantages": (q - float(q.max())).tolist(),
        "selected_action": action,
        "ranking": ranking(q),
        "top_two_margin": top_margin(q),
        "raw_belief_mean": raw_mean,
        "common_simulator_truth": float(truth),
        "absolute_belief_mean_error": abs(raw_mean - truth),
        "relative_belief_mean_error": abs(raw_mean - truth) / max(abs(float(truth)), 1.0e-12),
        "zero_mass": float(belief.probabilities[0]),
        "normalization_error": normalization_error,
        "belief_probabilities": belief.probabilities.tolist(),
        "predictive_support_size_selected_action": int(np.count_nonzero(predicted > 0.0)),
        "representative_observation_requested_branch_count": 7,
        "number_of_distinct_representative_observations": {
            "minimum": min(int(k) for k in planner.last_audit["distinct_branch_histogram"]),
            "maximum": max(int(k) for k in planner.last_audit["distinct_branch_histogram"]),
        },
        "distinct_branch_histogram": planner.last_audit["distinct_branch_histogram"],
        "branch_weights": planner.last_audit["root_action_branches"],
        "effective_branch_count": {
            "definition": "inverse Simpson after aggregating duplicate representative observations",
            "minimum": planner.last_audit["effective_branch_count_inverse_simpson_min"],
            "mean": planner.last_audit["effective_branch_count_inverse_simpson_mean"],
            "maximum": planner.last_audit["effective_branch_count_inverse_simpson_max"],
        },
        "reachable_graph_support_coverage": planner.last_audit["reachable_graph_support_coverage"],
        "backup_support_coverage": planner.last_audit["backup_support_coverage"],
        "duplicate_belief_rate": planner.last_audit["duplicate_point_rate"],
        "nearest_point_distance": {
            "minimum": planner.last_audit["nearest_point_distance_min"],
            "mean": planner.last_audit["nearest_point_distance_mean"],
            "maximum": planner.last_audit["nearest_point_distance_max"],
        },
        "initial_N0_snapping_error": initial_snap_error,
        "planner_model_calls": counter_delta(after, before),
        "runtime_seconds": elapsed,
        "peak_rss_mib": rss_mib(),
        "identity_digest": identity_digest,
        "query_bank_digest": query_bank_digest,
    }
    return record, action


def preliminary_query_digest(seed: int, arm: str) -> str:
    return canonical_digest({"arm": arm, "seed": seed, "roots": ROOTS, "source": "deployed_41"})


def create_query_bank(
    base: dict[str, Any], arm: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], CharacterisationPOMDP]:
    pomdp = build_pomdp(base, arm, 41)
    planner = CharacterisationPlanner(pomdp, pomdp.config, GAMMA, pomdp.seed)
    identity = run_identity(pomdp, base, arm, 41, "deployed")
    env = ContinuousEcologyEnv(base["env_cfg"])
    reset = env.reset(seed)
    n0 = float(base["env_cfg"].N0)
    belief = initial_belief(pomdp, "deployed", float(reset.observation), n0)
    used_initial = belief.probabilities.copy()
    raw_grid = pomdp.state_abundances() * pomdp.model.survey_scale
    snap_error = abs(float(raw_grid[int(np.argmax(used_initial))]) - n0)
    roots: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    preliminary = preliminary_query_digest(seed, arm)
    current = reset
    for step in range(ROOTS):
        root = {
            "root": step,
            "truth": float(current.evaluator_info["state"]),
            "observation": float(current.observation),
            "public_context": jsonable(current.public_info),
            "belief_capacity": float(belief.capacity),
            "belief_timestep": int(belief.timestep),
        }
        record, action = evaluate_root(
            pomdp, planner, belief, root["truth"], step, snap_error,
            identity["identity_digest"], preliminary,
        )
        root["baseline_41_action"] = action
        common_context = {
            "truth": root["truth"], "observation": root["observation"],
            "public_context": root["public_context"],
            "belief_capacity": root["belief_capacity"],
            "belief_timestep": root["belief_timestep"],
        }
        record["common_root_context"] = common_context
        record["common_root_context_digest"] = canonical_digest(common_context)
        roots.append(root)
        records.append(record)
        result = env.step(action)
        if step < ROOTS - 1:
            belief, _ = pomdp.runtime_update(belief, action, float(result.observation))
        current = result
        if result.done and step != ROOTS - 1:
            raise AssertionError("query-bank environment ended early")
    query_payload = {
        "arm": arm,
        "seed": seed,
        "source": "deployed 41-bin policy",
        "finer_recommendations_applied": False,
        "reward_or_return_accumulated_or_saved": False,
        "roots": roots,
    }
    query_digest = canonical_digest(query_payload)
    for record in records:
        record["query_bank_digest"] = query_digest
    run = finalize_run(
        pomdp, planner, base, arm, seed, 41, "deployed", identity,
        query_digest, records, used_initial, snap_error,
    )
    return roots, run, pomdp


def replay_run(
    base: dict[str, Any], arm: str, seed: int, bins: int, prior: str,
    roots: list[dict[str, Any]], query_digest: str,
) -> dict[str, Any]:
    pomdp = build_pomdp(base, arm, bins)
    planner = CharacterisationPlanner(pomdp, pomdp.config, GAMMA, pomdp.seed)
    identity = run_identity(pomdp, base, arm, bins, prior)
    n0 = float(base["env_cfg"].N0)
    belief = initial_belief(pomdp, prior, roots[0]["observation"], n0)
    used_initial = belief.probabilities.copy()
    raw_grid = pomdp.state_abundances() * pomdp.model.survey_scale
    snap_error = abs(float(raw_grid[int(np.argmax(used_initial))]) - n0)
    records = []
    for step, root in enumerate(roots):
        if int(belief.timestep) != root["belief_timestep"]:
            raise AssertionError("common replay timestep mismatch")
        if abs(float(belief.capacity) - root["belief_capacity"]) > 1.0e-12:
            raise AssertionError("common replay capacity mismatch")
        if abs(float(belief.current_observation) - root["observation"]) > 1.0e-12:
            raise AssertionError("common replay observation mismatch")
        record, _ = evaluate_root(
            pomdp, planner, belief, root["truth"], step, snap_error,
            identity["identity_digest"], query_digest,
        )
        common_context = {
            "truth": root["truth"], "observation": root["observation"],
            "public_context": root["public_context"],
            "belief_capacity": root["belief_capacity"],
            "belief_timestep": root["belief_timestep"],
        }
        record["common_root_context"] = common_context
        record["common_root_context_digest"] = canonical_digest(common_context)
        records.append(record)
        if step < ROOTS - 1:
            belief, _ = pomdp.runtime_update(
                belief, int(root["baseline_41_action"]), float(roots[step + 1]["observation"])
            )
    return finalize_run(
        pomdp, planner, base, arm, seed, bins, prior, identity,
        query_digest, records, used_initial, snap_error,
    )


def finalize_run(
    pomdp: CharacterisationPOMDP, planner: CharacterisationPlanner,
    base: dict[str, Any], arm: str, seed: int, bins: int, prior: str,
    identity: dict[str, Any], query_digest: str, records: list[dict[str, Any]],
    used_initial: np.ndarray, snap_error: float,
) -> dict[str, Any]:
    actions = np.asarray([row["selected_action"] for row in records], dtype=np.float64)
    raw_states = pomdp.state_abundances() * pomdp.model.survey_scale
    if pomdp.p5_impossible["runtime"] or pomdp.p5_impossible["lookahead"]:
        raise AssertionError("fallback route reached")
    if pomdp.p5_max_normalization_error > 1.0e-12:
        raise AssertionError("posterior normalization failure")
    identity["transition_kernel_hash"] = core.full_transition_kernel_hash(pomdp)
    identity["raw_transition_matrix_hash"] = phase3.raw_transition_matrix_hash(pomdp)
    identity["identity_digest_with_kernel"] = canonical_digest(identity)
    for record in records:
        record["identity_digest"] = identity["identity_digest_with_kernel"]
    return {
        "arm": arm,
        "validation_seed": seed,
        "state_bins": bins,
        "prior_convention": prior,
        "control_label": CONTROL_LABEL if prior == "barycentric_control" else None,
        "query_bank_digest": query_digest,
        "identity": identity,
        "initial_prior": {
            "probabilities": used_initial.tolist(),
            "probability_sum": float(used_initial.sum()),
            "raw_mean": float(np.dot(used_initial, raw_states)),
            "N0": float(base["env_cfg"].N0),
            "N0_snapping_error": snap_error,
            "initial_belief_calls": int(pomdp.p5_calls["initial_belief"]),
        },
        "records": records,
        "action_sequence_hash": array_hash(actions),
        "planner_model_calls_total": {
            "pomdp": dict(pomdp.p5_calls),
            "true_reward_calls": int(pomdp.true_reward_calls),
            "kernel_build_count": int(pomdp.kernel_build_count),
            "planner_invocations": int(planner.invocation_count),
        },
        "maximum_posterior_normalization_error": pomdp.p5_max_normalization_error,
        "fallback_counts": dict(pomdp.p5_impossible),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in records)),
        "peak_rss_mib": max(row["peak_rss_mib"] for row in records),
        "scientific_returns_read_or_accumulated": False,
    }


def union_support_belief_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    lx = np.asarray(left["identity"]["raw_grid"], dtype=np.float64)
    rx = np.asarray(right["identity"]["raw_grid"], dtype=np.float64)
    lp = np.asarray(left["record"]["belief_probabilities"], dtype=np.float64)
    rp = np.asarray(right["record"]["belief_probabilities"], dtype=np.float64)
    grid = np.unique(np.concatenate((lx, rx)))
    lcdf = np.asarray([lp[lx <= value].sum() for value in grid])
    rcdf = np.asarray([rp[rx <= value].sum() for value in grid])
    return {
        "belief_wasserstein_1": float(np.sum(np.abs(lcdf[:-1] - rcdf[:-1]) * np.diff(grid))),
        "common_step_cdf_maximum_difference": float(np.max(np.abs(lcdf - rcdf))),
        "raw_belief_mean_difference": abs(
            left["record"]["raw_belief_mean"] - right["record"]["raw_belief_mean"]
        ),
    }


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left["query_bank_digest"] != right["query_bank_digest"]:
        raise AssertionError("mismatched common histories")
    if left["arm"] != right["arm"] or left["prior_convention"] != right["prior_convention"]:
        raise AssertionError("pair arm/prior contamination")
    disagreements = []
    ranking_disagreements = []
    centered_differences = []
    belief_rows = []
    branch_changes = []
    for lrow, rrow in zip(left["records"], right["records"]):
        if lrow["root"] != rrow["root"] or lrow["common_simulator_truth"] != rrow["common_simulator_truth"]:
            raise AssertionError("mismatched common replay root")
        lq = np.asarray(lrow["q_values"])
        rq = np.asarray(rrow["q_values"])
        difference = float(np.max(np.abs((lq - lq.max()) - (rq - rq.max()))))
        centered_differences.append(difference)
        metrics = union_support_belief_metrics(
            {"identity": left["identity"], "record": lrow},
            {"identity": right["identity"], "record": rrow},
        )
        metrics["root"] = lrow["root"]
        metrics["left_truth_anchored_belief_error"] = lrow["absolute_belief_mean_error"]
        metrics["right_truth_anchored_belief_error"] = rrow["absolute_belief_mean_error"]
        belief_rows.append(metrics)
        branch_changes.append({
            "root": lrow["root"],
            "left_effective_mean": lrow["effective_branch_count"]["mean"],
            "right_effective_mean": rrow["effective_branch_count"]["mean"],
            "left_distinct_histogram": lrow["distinct_branch_histogram"],
            "right_distinct_histogram": rrow["distinct_branch_histogram"],
        })
        if lrow["ranking"] != rrow["ranking"]:
            ranking_disagreements.append(lrow["root"])
        if lrow["selected_action"] != rrow["selected_action"]:
            lscale = max(abs(lrow["maximum_q"]), 1.0e-12)
            rscale = max(abs(rrow["maximum_q"]), 1.0e-12)
            disagreements.append({
                "root": lrow["root"],
                "left_action": lrow["selected_action"],
                "right_action": rrow["selected_action"],
                "left_q_values": lrow["q_values"],
                "right_q_values": rrow["q_values"],
                "left_ranking": lrow["ranking"],
                "right_ranking": rrow["ranking"],
                "left_margin": lrow["top_two_margin"],
                "right_margin": rrow["top_two_margin"],
                "left_relative_margin": lrow["top_two_margin"] / lscale,
                "right_relative_margin": rrow["top_two_margin"] / rscale,
                "near_tied_by_declared_rule": bool(
                    lrow["top_two_margin"] / lscale <= NEAR_TIE_RELATIVE_THRESHOLD
                    or rrow["top_two_margin"] / rscale <= NEAR_TIE_RELATIVE_THRESHOLD
                ),
            })
    values = np.asarray(centered_differences)
    return {
        "left_bins": left["state_bins"],
        "right_bins": right["state_bins"],
        "action_disagreement_count": len(disagreements),
        "action_disagreement_locations": [row["root"] for row in disagreements],
        "fraction_changed_actions": len(disagreements) / ROOTS,
        "action_disagreements": disagreements,
        "ranking_disagreement_count": len(ranking_disagreements),
        "ranking_disagreement_locations": ranking_disagreements,
        "maximum_centred_q_difference": float(values.max()),
        "median_centred_q_difference": float(np.median(values)),
        "difference_in_mean_max_q": float(
            np.mean([row["maximum_q"] for row in right["records"]])
            - np.mean([row["maximum_q"] for row in left["records"]])
        ),
        "belief_metrics_by_root": belief_rows,
        "belief_wasserstein_1_max": max(row["belief_wasserstein_1"] for row in belief_rows),
        "belief_wasserstein_1_median": float(np.median([row["belief_wasserstein_1"] for row in belief_rows])),
        "common_step_cdf_maximum_difference": max(
            row["common_step_cdf_maximum_difference"] for row in belief_rows
        ),
        "raw_belief_mean_difference_max": max(row["raw_belief_mean_difference"] for row in belief_rows),
        "truth_anchored_belief_error": {
            "left_mean": float(np.mean([row["left_truth_anchored_belief_error"] for row in belief_rows])),
            "right_mean": float(np.mean([row["right_truth_anchored_belief_error"] for row in belief_rows])),
        },
        "branch_collapse_changes": branch_changes,
        "direct_vector_l1_or_tv_applied_to_unequal_grids": False,
        "unequal_grid_metric_support": "union of raw abundance grid supports",
    }


def treatment_isolation(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for prior in PRIORS:
        reference = runs[prior]["41"]["identity"]
        for bins in GRIDS[1:]:
            candidate = runs[prior][str(bins)]["identity"]
            config_diff = {
                key for key in reference["planner_config"]
                if reference["planner_config"][key] != candidate["planner_config"][key]
            }
            if config_diff != {"state_bins"}:
                raise AssertionError(f"treatment drift in planner config: {config_diff}")
            held = (
                "model_parameter_hash", "raw_grid_endpoints", "capacity_grid_hash",
                "planning_seed", "action_table_hash", "action_cost_hash", "reward_route",
                "observation_update_route", "survey_scale", "diagnostic_wrapper_code_digest",
                "characterisation_wrapper_digest", "fit_cache_key",
            )
            mismatch = [key for key in held if reference[key] != candidate[key]]
            if mismatch:
                raise AssertionError(f"treatment drift beyond state_bins: {mismatch}")
            rows.append({
                "prior": prior, "left_bins": 41, "right_bins": bins,
                "config_difference": sorted(config_diff), "held_field_mismatches": mismatch,
            })
    return {
        "status": "PASS",
        "only_state_bins_varies_within_each_prior_comparison": True,
        "labelled_prior_control_is_the_only_cross_prior_change": True,
        "observation_bins": 41,
        "observation_branches": 7,
        "horizon": 5,
        "belief_points": 32,
        "rows": rows,
    }


def run_task(arm: str, seed: int, output: Path) -> dict[str, Any]:
    environment = assert_environment()
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("task outside authorised arm/seed set")
    started = time.perf_counter()
    base = core.build_arms()[CELL_ID]
    unit_tests = control_prior_unit_tests(base)
    roots, deployed_41, _ = create_query_bank(base, arm, seed)
    query_payload = {
        "arm": arm, "seed": seed, "source": "deployed 41-bin policy",
        "finer_recommendations_applied": False,
        "reward_or_return_accumulated_or_saved": False,
        "roots": roots,
    }
    query_digest = canonical_digest(query_payload)
    if deployed_41["query_bank_digest"] != query_digest:
        raise AssertionError("query-bank digest mutation")
    runs: dict[str, dict[str, Any]] = {prior: {} for prior in PRIORS}
    runs["deployed"]["41"] = deployed_41
    for prior in PRIORS:
        for bins in GRIDS:
            if prior == "deployed" and bins == 41:
                continue
            runs[prior][str(bins)] = replay_run(
                base, arm, seed, bins, prior, roots, query_digest
            )
    isolation = treatment_isolation(runs)
    controlling = load_json(CONTROLLING_RECEIPT)
    controlling_hash = controlling["validation_episodes"][CELL_ID][arm]["actions_hash"]
    if seed == core.VALIDATION_SEED and runs["deployed"]["41"]["action_sequence_hash"] != controlling_hash:
        raise AssertionError("controlling 41-bin action hash was not reproduced")
    comparisons: dict[str, Any] = {prior: {} for prior in PRIORS}
    for prior in PRIORS:
        for left, right in PAIR_SEQUENCE:
            comparisons[prior][f"{left}_to_{right}"] = compare_pair(
                runs[prior][str(left)], runs[prior][str(right)]
            )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_utc": now(),
        "environment": environment,
        "scope": {
            "arm": arm, "cell": CELL_ID, "validation_seed": seed,
            "grids": list(GRIDS), "priors": list(PRIORS), "roots": ROOTS,
            "post_failure_descriptive_not_preregistered_confirmation": True,
            "control_label": CONTROL_LABEL,
        },
        "query_bank": query_payload,
        "query_bank_digest": query_digest,
        "control_prior_unit_tests": unit_tests,
        "treatment_isolation": isolation,
        "runs": runs,
        "successive_grid_comparisons": comparisons,
        "no_return_boundary": {
            "scientific_evaluation_returns_read": False,
            "reward_or_return_accumulated_or_saved": False,
            "V7_or_V8_run": False,
            "contrasts_confidence_intervals_or_scientific_jobs_run": False,
        },
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": rss_mib(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    digest_payload = dict(payload)
    digest_payload.pop("created_utc")
    digest_payload.pop("execution")
    payload["task_digest"] = canonical_digest(digest_payload)
    path = output / "tasks" / f"{arm.lower()}_seed_{seed}.json"
    write_json_new(path, payload)
    print(json.dumps({
        "status": "PASS", "task": f"{arm}/{seed}",
        "task_digest": payload["task_digest"], "output": str(path.resolve()),
        "elapsed_seconds": payload["execution"]["elapsed_seconds"],
    }, indent=2, sort_keys=True))
    return payload


def aggregate(output: Path) -> dict[str, Any]:
    environment = assert_environment()
    tasks: dict[str, Any] = {}
    all_disagreements = []
    summaries: dict[str, Any] = {arm: {prior: {} for prior in PRIORS} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = output / "tasks" / f"{arm.lower()}_seed_{seed}.json"
            task = load_json(path)
            if task.get("status") != "PASS" or task["scope"]["arm"] != arm:
                raise AssertionError(f"invalid task output: {path}")
            tasks[f"{arm}_{seed}"] = {
                "path": str(path.resolve()), "sha256": sha256_file(path),
                "task_digest": task["task_digest"],
            }
    for arm in ARMS:
        for prior in PRIORS:
            pair_rows = {}
            grid_errors = {str(grid): [] for grid in GRIDS}
            grid_effective = {str(grid): [] for grid in GRIDS}
            for left, right in PAIR_SEQUENCE:
                name = f"{left}_to_{right}"
                per_seed = []
                for seed in SEEDS:
                    task = load_json(Path(tasks[f"{arm}_{seed}"]["path"]))
                    pair = task["successive_grid_comparisons"][prior][name]
                    per_seed.append({
                        "seed": seed,
                        "action_disagreement_count": pair["action_disagreement_count"],
                        "action_disagreement_locations": pair["action_disagreement_locations"],
                        "fraction_changed_actions": pair["fraction_changed_actions"],
                        "ranking_disagreement_count": pair["ranking_disagreement_count"],
                        "ranking_disagreement_locations": pair["ranking_disagreement_locations"],
                        "maximum_centred_q_difference": pair["maximum_centred_q_difference"],
                        "median_centred_q_difference": pair["median_centred_q_difference"],
                        "difference_in_mean_max_q": pair["difference_in_mean_max_q"],
                        "belief_wasserstein_1_max": pair["belief_wasserstein_1_max"],
                        "common_step_cdf_maximum_difference": pair["common_step_cdf_maximum_difference"],
                        "raw_belief_mean_difference_max": pair["raw_belief_mean_difference_max"],
                        "truth_anchored_belief_error": pair["truth_anchored_belief_error"],
                    })
                    for row in pair["action_disagreements"]:
                        all_disagreements.append({
                            "arm": arm, "prior": prior, "seed": seed,
                            "grid_pair": name, **row,
                        })
                pair_rows[name] = {
                    "per_seed": per_seed,
                    "total_action_disagreements": sum(row["action_disagreement_count"] for row in per_seed),
                    "total_ranking_disagreements": sum(row["ranking_disagreement_count"] for row in per_seed),
                    "maximum_centred_q_difference_across_seeds": max(row["maximum_centred_q_difference"] for row in per_seed),
                    "median_of_seed_median_centred_q_differences": float(np.median([row["median_centred_q_difference"] for row in per_seed])),
                }
            for seed in SEEDS:
                task = load_json(Path(tasks[f"{arm}_{seed}"]["path"]))
                for grid in GRIDS:
                    records = task["runs"][prior][str(grid)]["records"]
                    grid_errors[str(grid)].extend(row["absolute_belief_mean_error"] for row in records)
                    grid_effective[str(grid)].extend(row["effective_branch_count"]["mean"] for row in records)
            disagreement_sequence = [pair_rows[f"{l}_to_{r}"]["total_action_disagreements"] for l, r in PAIR_SEQUENCE]
            ranking_sequence = [pair_rows[f"{l}_to_{r}"]["total_ranking_disagreements"] for l, r in PAIR_SEQUENCE]
            summaries[arm][prior] = {
                "successive_pairs": pair_rows,
                "action_disagreement_sequence_across_five_seeds": disagreement_sequence,
                "ranking_disagreement_sequence_across_five_seeds": ranking_sequence,
                "mean_truth_anchored_belief_error_by_grid": {
                    key: float(np.mean(value)) for key, value in grid_errors.items()
                },
                "mean_effective_branch_count_by_grid": {
                    key: float(np.mean(value)) for key, value in grid_effective.items()
                },
                "41_remains_outlier_by_successive_action_disagreement_count": bool(
                    disagreement_sequence[0] > max(disagreement_sequence[1:])
                ),
                "last_two_refinements_agree_at_every_root": bool(
                    disagreement_sequence[2] == 0 and disagreement_sequence[3] == 0
                ),
                "all_action_disagreements_near_tied_by_declared_rule": bool(
                    all(row["near_tied_by_declared_rule"] for row in all_disagreements
                        if row["arm"] == arm and row["prior"] == prior)
                ),
                "truth_anchored_belief_error_monotonically_nonincreasing": bool(
                    all(np.mean(grid_errors[str(GRIDS[i + 1])]) <= np.mean(grid_errors[str(GRIDS[i])])
                        for i in range(len(GRIDS) - 1))
                ),
                "representative_branch_effective_count_monotonically_nondecreasing": bool(
                    all(np.mean(grid_effective[str(GRIDS[i + 1])]) >= np.mean(grid_effective[str(GRIDS[i])])
                        for i in range(len(GRIDS) - 1))
                ),
            }
    primary_sequences = {arm: summaries[arm]["deployed"]["action_disagreement_sequence_across_five_seeds"] for arm in ARMS}
    labels = []
    if all(summaries[arm]["deployed"]["last_two_refinements_agree_at_every_root"] for arm in ARMS):
        labels.append("POLICY STABILITY OBSERVED ACROSS FINAL REFINEMENTS")
    if any(sequence[-1] > 0 for sequence in primary_sequences.values()):
        labels.append("INSTABILITY PERSISTS")
    if any(any(sequence[i + 1] > sequence[i] for i in range(len(sequence) - 1)) for sequence in primary_sequences.values()):
        labels.append("RESULTS OSCILLATE WITH GRID PHASE")
    labels.append("INSUFFICIENT TO SELECT A GRID")
    payload = {
        "schema": f"{SCHEMA}_aggregate_v1",
        "status": "PASS",
        "created_utc": now(),
        "environment": environment,
        "tasks": tasks,
        "summaries": summaries,
        "all_action_disagreements": all_disagreements,
        "interpretation": {
            "labels": labels,
            "near_tie_definition": (
                "at least one grid's top-two planner-Q margin is <=1% of the absolute maximum Q; "
                "this is not the 0.10 episode-return margin"
            ),
            "barycentric_control_materially_changes_pattern": {
                arm: summaries[arm]["deployed"]["action_disagreement_sequence_across_five_seeds"]
                != summaries[arm]["barycentric_control"]["action_disagreement_sequence_across_five_seeds"]
                for arm in ARMS
            },
            "A3_and_A4_same_primary_convergence_pattern": primary_sequences["A3"] == primary_sequences["A4"],
            "no_grid_selected_or_installed": True,
            "finite_horizon_planner_Q_not_compared_to_episode_return_margin": True,
            "post_failure_descriptive_not_preregistered_confirmation": True,
        },
        "no_return_boundary": {
            "scientific_evaluation_returns_read": False,
            "V7_or_V8_E1_comparison_run": False,
            "scientific_slurm_jobs_submitted": False,
        },
    }
    digest_payload = dict(payload)
    digest_payload.pop("created_utc")
    payload["characterisation_digest"] = canonical_digest(digest_payload)
    path = output / "PHASE5_GRID_CHARACTERISATION_RECEIPT.json"
    write_json_new(path, payload)
    print(json.dumps({
        "status": "PASS", "characterisation_digest": payload["characterisation_digest"],
        "labels": labels, "action_disagreements": len(all_disagreements),
        "output": str(path.resolve()),
    }, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--arm", choices=ARMS, required=True)
    task.add_argument("--seed", type=int, choices=SEEDS, required=True)
    subparsers.add_parser("aggregate")
    subparsers.add_parser("unit-test")
    args = parser.parse_args()
    if args.command == "task":
        run_task(args.arm, args.seed, args.output)
    elif args.command == "aggregate":
        aggregate(args.output)
    else:
        assert_environment()
        base = core.build_arms()[CELL_ID]
        print(json.dumps(control_prior_unit_tests(base), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
