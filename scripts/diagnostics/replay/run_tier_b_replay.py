#!/usr/bin/env python3
"""Parity-gated Tier-B diagnostic replay for the four general methods.

This imports the frozen general-run code, not the ricker-only PLUS/MOOR snapshot.
Logging copies method diagnostics and already-computed planning locals.  It never
re-evaluates a policy score, advances an environment, or consumes an RNG draw.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ACCEPTED_CSV,
    GENERAL_DATA_ROOT,
    GENERAL_PACKAGE,
    GENERAL_SCRIPTS,
    GENERAL_SRC,
    REPLAY_OUTPUT,
    require_scientific_tables,
)

REPLAY = REPLAY_OUTPUT / "general"
ACCEPTED_SHA = "7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c"

GENERAL_QUARANTINE = GENERAL_DATA_ROOT
GENERAL_CONFIG = GENERAL_PACKAGE / "configs" / "general_phase2e_full_sigma01_02.yaml"
GENERAL_MANIFEST = (
    GENERAL_PACKAGE / "manifests" / "full_general_sigma01_02_576_rows.csv"
)

sys.path[:0] = [str(GENERAL_SCRIPTS), str(GENERAL_SRC)]

from run_real_manifest_row import apply_row_config  # noqa: E402
from real_ecology_benchmark.backend import (  # noqa: E402
    resolve_backend_for_workload,
    set_active_backend,
)
from real_ecology_benchmark.beliefs import PublicBeliefCache  # noqa: E402
from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.dataset import dataset_sha256  # noqa: E402
import real_ecology_benchmark.evaluator as evaluator_mod  # noqa: E402
from real_ecology_benchmark.evaluator import ContinuousEvaluator  # noqa: E402
from real_ecology_benchmark.envs import make_env as real_make_env  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    _hidden_method_context,
    _load_or_fit_public_surrogate,
    build_method,
    ensure_dataset,
    make_filter_factory,
)
from real_ecology_benchmark.training_monitor import split_train_holdout  # noqa: E402


CELLS = {
    "B1": ("Crab-eating fox", "regime", 0.2),
    "B2": ("Amur tiger", "ricker", 0.1),
    "B3": ("Amur tiger", "allee", 0.2),
}
METHODS = (
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
)
METHOD_TAG = {
    "refplan": "refplan",
    "ogsrl": "ogsrl",
    "bamcts": "bamcts",
    "ensemble_value_disagreement_pessimism": "evd",
}
RECOVERY_ACTIONS = tuple(range(3, 11))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(value: str) -> str:
    return "_".join(value.lower().replace("-", " ").split())


def sigma_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_row(population: str, family: str, sigma: float, method: str) -> dict[str, str]:
    matches = [
        row for row in read_rows(GENERAL_MANIFEST)
        if row["reward_mode"] == "safe"
        and row["population"] == population
        and row["environment"] == family
        and float(row["sigma_obs"]) == sigma
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"general manifest match count {len(matches)}")
    return matches[0]


def accepted_fields(
    population: str, family: str, sigma: float, method: str
) -> dict[str, object]:
    matches = [
        row for row in read_rows(ACCEPTED_CSV)
        if row["population"] == population
        and row["environment"] == family
        and float(row["sigma_obs"]) == sigma
        and row["reward_mode"] == "safe"
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"accepted-row match count {len(matches)}")
    row = matches[0]
    return {
        "return_mean": float(row["operational_return_mean"]),
        "return_sd": float(row["operational_return_sd"]),
        "unsafe_fraction_mean": float(row["unsafe_fraction"]),
        "persistence_mean": float(row["persistence_mean"]),
        "collapse_entry_mean": float(row["collapse_rate"]),
        "min_population_mean": float(row["min_population_mean"]),
        "economic_cost_mean": float(row["economic_cost_mean"]),
        "dataset_sha256": row["dataset_sha256"],
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "refplan.py": GENERAL_SRC / "real_ecology_benchmark/methods/refplan.py",
        "ogsrl.py": GENERAL_SRC / "real_ecology_benchmark/methods/ogsrl.py",
        "bamcts.py": GENERAL_SRC / "real_ecology_benchmark/methods/bamcts.py",
        "ensemble_value_disagreement.py":
            GENERAL_SRC / "real_ecology_benchmark/methods/ensemble_value_disagreement.py",
        "public_models.py": GENERAL_SRC / "real_ecology_benchmark/public_models.py",
        "pipeline.py": GENERAL_SRC / "real_ecology_benchmark/pipeline.py",
        "config": GENERAL_CONFIG,
        "manifest": GENERAL_MANIFEST,
    }
    return {name: sha(path) for name, path in paths.items()}


def summarize7(rows: list[dict[str, object]]) -> dict[str, float]:
    def column(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    returns = column("operational_return")
    return {
        "return_mean": float(returns.mean()),
        "return_sd": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "unsafe_fraction_mean": float(column("unsafe_fraction").mean()),
        "persistence_mean": float(column("persistence").mean()),
        "collapse_entry_mean": float(column("collapse_entry").mean()),
        "min_population_mean": float(column("min_true_state").mean()),
        "economic_cost_mean": float(column("economic_cost").mean()),
    }


def parity(replayed: dict[str, float], accepted: dict[str, object]) -> dict[str, object]:
    fields = (
        "return_mean",
        "return_sd",
        "unsafe_fraction_mean",
        "persistence_mean",
        "collapse_entry_mean",
        "min_population_mean",
        "economic_cost_mean",
    )
    diffs = {field: abs(replayed[field] - float(accepted[field])) for field in fields}
    maximum = max(diffs.values())
    verdict = "PASS" if maximum <= 1e-9 else (
        "INVESTIGATE" if maximum <= 1e-6 else "FAIL"
    )
    return {
        "verdict": verdict,
        "max_abs_diff": maximum,
        "per_field": {
            field: {
                "replay": replayed[field],
                "accepted": accepted[field],
                "abs_diff": diffs[field],
            }
            for field in fields
        },
    }


def _copy_array(value) -> np.ndarray:
    return np.asarray(value).copy()


class TierBLogger:
    def __init__(self, cfg, method: str, policy):
        self.cfg = cfg
        self.method = method
        self.policy = policy
        self.rows: list[dict[str, object]] = []
        self.pending: dict[str, object] | None = None
        self.seed = -1
        self.step = 0
        self.trace_capture: dict[str, object] = {}

    def on_reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.step = 0
        self.pending = None

    def capture_refplan_locals(self, local: dict[str, object]) -> None:
        required = ("member_returns", "mean", "variance", "reflected", "sequences", "best")
        if not all(key in local for key in required):
            raise AssertionError("RefPlan trace did not expose expected computed locals")
        mean = _copy_array(local["mean"])
        variance = _copy_array(local["variance"])
        reflected = _copy_array(local["reflected"])
        sequences = _copy_array(local["sequences"]).astype(int)
        best = int(local["best"])
        action_mean = np.full(self.cfg.environment.num_actions, -np.inf)
        for action in range(self.cfg.environment.num_actions):
            matching = np.flatnonzero(sequences[:, 0] == action)
            if len(matching):
                action_mean[action] = mean[matching[np.argmax(mean[matching])]]
        self.trace_capture = {
            "member_returns": _copy_array(local["member_returns"]),
            "action_mean_lambda0": action_mean,
            "argmax_lambda0": int(sequences[int(np.argmax(mean)), 0]),
            "best_sequence_mean": float(mean[best]),
            "best_sequence_return_sd": float(np.sqrt(max(variance[best], 0.0))),
            "best_sequence_reflected": float(reflected[best]),
        }

    def capture_bamcts_locals(self, local: dict[str, object]) -> None:
        tree = local.get("tree")
        if not isinstance(tree, dict) or not tree:
            raise AssertionError("BA-MCTS trace did not expose a non-empty computed tree")
        roots = [node for key, node in tree.items() if int(key[0]) == 0]
        if not roots:
            raise AssertionError("BA-MCTS trace exposed no root nodes")
        visits = np.sum(
            [np.asarray(node.action_visits, dtype=np.int64) for node in roots], axis=0
        )
        self.trace_capture = {
            "root_visit_counts": visits,
            "max_depth": int(max(int(key[0]) for key in tree)),
            "tree_nodes_trace": int(len(tree)),
        }

    def on_act(
        self,
        belief,
        observation: float,
        action: int,
        diagnostics: dict[str, object],
        model_posterior: np.ndarray | None,
    ) -> None:
        row: dict[str, object] = {
            "seed": self.seed,
            "t": self.step,
            "obs_t": float(observation),
            "action_t": int(action),
        }
        if self.method == "refplan":
            weights = np.asarray(belief.weights, dtype=np.float64)
            row.update(
                {
                    "belief_mean": float(belief.mean_state()),
                    "belief_entropy": float(-np.sum(weights * np.log(weights + 1e-300))),
                    "model_posterior": _copy_array(model_posterior),
                    "model_entropy": float(diagnostics["model_entropy"]),
                    "posterior_max": float(diagnostics["posterior_max"]),
                    "action_scores": _copy_array(diagnostics["action_scores"]),
                    "public_extinction_risk": float(
                        diagnostics["public_extinction_risk"]
                    ),
                    **self.trace_capture,
                }
            )
        elif self.method == "ogsrl":
            predicted_cost = float(diagnostics["public_low_abundance_cost"])
            budget = float(diagnostics["safety_budget"])
            ood = float(diagnostics["ood_probability"])
            row.update(
                {
                    "action_probabilities": _copy_array(
                        diagnostics["action_probabilities"]
                    ),
                    "predicted_C25": predicted_cost,
                    "safety_budget": budget,
                    "safety_slack": budget - predicted_cost,
                    "lambda_safety": float(diagnostics["lambda_safety"]),
                    "lambda_ood": float(diagnostics["lambda_ood"]),
                    "ood_probability": ood,
                    "ood_guardian_flag": int(
                        ood > float(self.policy.deployment_ood_limit)
                    ),
                    "guardian_override": int(bool(diagnostics["guardian_override"])),
                    "hard_fallback": int(bool(diagnostics["hard_fallback"])),
                    "low_abundance_scale": float(diagnostics["low_abundance_scale"]),
                    "true_s_safe_evaluator_only": float(
                        self.cfg.environment.safety_threshold
                    ),
                    "cost_horizon_used": float(diagnostics["cost_horizon_used"]),
                    "behavior_normalized_cost": float(
                        self.policy.fit_diagnostics["behavior_normalized_cost"]
                    ),
                }
            )
        elif self.method == "bamcts":
            row.update(
                {
                    "root_q": _copy_array(diagnostics["root_q"]),
                    "tree_nodes": int(diagnostics["tree_nodes"]),
                    "root_model_belief": _copy_array(model_posterior),
                    **self.trace_capture,
                }
            )
            assert row["tree_nodes"] == row["tree_nodes_trace"]
        else:
            q_mean = _copy_array(diagnostics["q_ensemble_mean"])
            q_var = _copy_array(diagnostics["q_ensemble_disagreement_all"])
            score = _copy_array(diagnostics["pessimistic_q_score"])
            row.update(
                {
                    "q_ensemble_mean": q_mean,
                    "q_ensemble_variance": q_var,
                    "pessimistic_q_score": score,
                    "argmax_lambda0": int(np.argmax(q_mean)),
                    "argmax_pessimistic": int(np.argmax(score)),
                }
            )
        self.pending = row
        self.trace_capture = {}

    def on_step(self, env, result) -> None:
        if self.pending is None:
            raise AssertionError("environment step occurred without a logged decision")
        info = result.evaluator_info
        row = self.pending
        row.update(
            {
                "x_true_t": float(info["state_previous"]),
                "x_true_next": float(info["state"]),
                "reward_t": float(result.reward),
                "cost_t": float(env.actions[int(row["action_t"])].cost),
                "unsafe_next": int(
                    float(info["state"]) <= self.cfg.environment.safety_threshold
                ),
            }
        )
        self.rows.append(row)
        self.pending = None
        self.step += 1


_LOGGER: TierBLogger | None = None


def make_instrumented_class(base):
    class InstrumentedEnv(base):
        def reset(self, seed, state_override=None):
            result = base.reset(self, seed, state_override)
            _LOGGER.on_reset(seed)
            return result

        def step(self, action_id):
            result = base.step(self, action_id)
            _LOGGER.on_step(self, result)
            return result

    return InstrumentedEnv


def traced_call(target_code, callback, function, *args):
    previous = sys.gettrace()

    def tracer(frame, event, arg):
        if event == "return" and frame.f_code is target_code:
            callback(frame.f_locals)
        return tracer

    sys.settrace(tracer)
    try:
        return function(*args)
    finally:
        sys.settrace(previous)


def _rng_states(policy) -> dict[str, object]:
    found: dict[str, object] = {}
    for label, owner in (
        ("policy", policy),
        ("planner", getattr(policy, "planner", None)),
        ("dynamics", getattr(policy, "dynamics", None)),
    ):
        rng = getattr(owner, "rng", None)
        if isinstance(rng, np.random.Generator):
            found[label] = copy.deepcopy(rng.bit_generator.state)
    return found


def build_policy(cell_key: str, method: str):
    population, family, sigma = CELLS[cell_key]
    row = manifest_row(population, family, sigma, method)
    cfg = load_config(str(GENERAL_CONFIG))
    cfg, _cell, _eval_cell = apply_row_config(
        cfg, row, GENERAL_QUARANTINE, dataset_root=GENERAL_QUARANTINE
    )
    cfg.validate()
    set_active_backend(resolve_backend_for_workload(cfg.compute, f"method:{method}", cfg.environment))
    dataset = ensure_dataset(cfg, regenerate=False)
    surrogate_path = Path(cfg.dataset.output).with_name(
        Path(cfg.dataset.output).stem
        + ".regime_hidden.public_surrogate.v2.seed20116.npz"
    )
    if not surrogate_path.exists():
        raise FileNotFoundError(f"required frozen public surrogate missing: {surrogate_path}")
    surrogate, loaded_path, status = _load_or_fit_public_surrogate(cfg, dataset)
    if status != "loaded" or loaded_path != surrogate_path:
        raise AssertionError((status, loaded_path, surrogate_path))
    context = _hidden_method_context(cfg, dataset, surrogate)
    factory, _ = make_filter_factory(cfg, dataset, "learned", context)
    cache_path = Path(cfg.dataset.output).with_name(
        Path(cfg.dataset.output).stem + ".regime_hidden.learned.beliefs.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"required frozen learned-belief cache missing: {cache_path}")
    cache = PublicBeliefCache.load(cache_path)
    train, train_cache, holdout, holdout_cache, split_info = split_train_holdout(
        dataset, cache, cfg
    )
    policy, _ = build_method(
        method,
        cfg,
        train,
        factory,
        train_cache,
        timings={},
        holdout_dataset=holdout,
        holdout_cache=holdout_cache,
        split_info=split_info,
        method_context=context,
    )
    return cfg, policy, factory, dataset, surrogate_path, cache_path, split_info, row


def run_cell(cell_key: str, method: str) -> dict[str, object]:
    global _LOGGER
    population, family, sigma = CELLS[cell_key]
    accepted = accepted_fields(population, family, sigma, method)
    (
        cfg,
        policy,
        factory,
        dataset,
        surrogate_path,
        cache_path,
        split_info,
        manifest,
    ) = build_policy(cell_key, method)
    actual_hash = dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset)
    if actual_hash != accepted["dataset_sha256"]:
        raise AssertionError(
            f"dataset hash mismatch: {actual_hash} != {accepted['dataset_sha256']}"
        )

    _LOGGER = TierBLogger(cfg, method, policy)
    original_act = policy.act

    def logged_act(belief, observation):
        model_posterior = (
            _copy_array(policy.posterior) if hasattr(policy, "posterior") else None
        )
        if method == "refplan":
            bound = policy.planner.plan_marginalized
            action = traced_call(
                bound.__func__.__code__,
                _LOGGER.capture_refplan_locals,
                original_act,
                belief,
                observation,
            )
        elif method == "bamcts":
            action = traced_call(
                original_act.__func__.__code__,
                _LOGGER.capture_bamcts_locals,
                original_act,
                belief,
                observation,
            )
        else:
            action = original_act(belief, observation)
        post_act_rng = _rng_states(policy)
        diagnostics = copy.deepcopy(getattr(policy, "last_diagnostics", {}) or {})
        _LOGGER.on_act(
            belief, observation, int(action), diagnostics, model_posterior
        )
        if _rng_states(policy) != post_act_rng:
            raise AssertionError("logging changed an RNG state")
        return action

    policy.act = logged_act

    def patched_make_env(env_cfg):
        env = real_make_env(env_cfg)
        env.__class__ = make_instrumented_class(type(env))
        return env

    start = time.perf_counter()
    evaluator_mod.make_env = patched_make_env
    try:
        episode_rows = ContinuousEvaluator(cfg, factory, "learned").run(policy)
    finally:
        evaluator_mod.make_env = real_make_env
        policy.act = original_act
    elapsed = time.perf_counter() - start

    if len(episode_rows) != 20 or len(_LOGGER.rows) != 20 * 50:
        raise AssertionError((len(episode_rows), len(_LOGGER.rows)))
    if {int(row["n_steps"]) for row in episode_rows} != {50}:
        raise AssertionError("not every Tier-B episode has 50 steps")
    if any(not 0 <= int(row["action_t"]) < 11 for row in _LOGGER.rows):
        raise AssertionError("illegal action in Tier-B log")
    replayed = summarize7(episode_rows)
    return {
        "cell": cell_key,
        "method": method,
        "population": population,
        "family": family,
        "sigma": sigma,
        "cfg": cfg,
        "policy": policy,
        "logger": _LOGGER,
        "replayed": replayed,
        "accepted": accepted,
        "parity": parity(replayed, accepted),
        "elapsed_seconds": elapsed,
        "dataset_sha256": actual_hash,
        "surrogate_path": surrogate_path,
        "cache_path": cache_path,
        "split_info": split_info,
        "manifest_index": int(manifest["index"]),
    }


def derived_metrics(result: dict[str, object]) -> dict[str, object]:
    method = result["method"]
    rows = result["logger"].rows
    common = {
        "M14_belief_filtering_columns": {
            "status": "present_and_valid" if method == "refplan" else "absent",
            "interpretation": (
                "RefPlan uses the learned public observation filter and sigma_o."
                if method == "refplan"
                else "Method has no belief/filtering diagnostic columns; do not compare it beside PLUS/MOOR/RefPlan filtering diagnostics."
            ),
        }
    }
    if method == "refplan":
        common["RefPlan"] = {
            "mean_best_sequence_posterior_return": float(
                np.mean([row["best_sequence_mean"] for row in rows])
            ),
            "mean_best_sequence_return_sd": float(
                np.mean([row["best_sequence_return_sd"] for row in rows])
            ),
            "switch_fraction_lambda_ref_0": float(
                np.mean(
                    [
                        int(row["action_t"]) != int(row["argmax_lambda0"])
                        for row in rows
                    ]
                )
            ),
            "mean_model_entropy": float(np.mean([row["model_entropy"] for row in rows])),
            "mean_posterior_max": float(np.mean([row["posterior_max"] for row in rows])),
        }
    elif method == "ogsrl":
        slack = np.asarray([row["safety_slack"] for row in rows], dtype=float)
        common["M12_OGSRL_constraint"] = {
            "fraction_steps_constraint_binds": float(np.mean(slack <= 0.0)),
            "mean_slack": float(slack.mean()),
            "s_low": float(rows[0]["low_abundance_scale"]),
            "true_s_safe_evaluator_only": float(rows[0]["true_s_safe_evaluator_only"]),
            "behavior_normalized_cost": float(rows[0]["behavior_normalized_cost"]),
            "lambda_safety": float(rows[0]["lambda_safety"]),
            "lambda_ood": float(rows[0]["lambda_ood"]),
            "guardian_override_fraction": float(
                np.mean([row["guardian_override"] for row in rows])
            ),
            "ood_guardian_flag_fraction": float(
                np.mean([row["ood_guardian_flag"] for row in rows])
            ),
            "hard_fallback_fraction": float(
                np.mean([row["hard_fallback"] for row in rows])
            ),
        }
    elif method == "bamcts":
        common["BA_MCTS"] = {
            "mean_tree_nodes": float(np.mean([row["tree_nodes"] for row in rows])),
            "mean_max_depth": float(np.mean([row["max_depth"] for row in rows])),
            "mean_root_visits_total": float(
                np.mean([np.sum(row["root_visit_counts"]) for row in rows])
            ),
        }
    else:
        switches = [
            int(row["argmax_lambda0"]) != int(row["argmax_pessimistic"])
            for row in rows
        ]
        common["M11_EVD"] = {
            "switch_fraction_lambda_V_0": float(np.mean(switches)),
            "surrogate_reward_t": "absent: EVD fitted its Q ensemble to raw dataset.rewards",
        }
        if result["cell"] == "B2":
            mean_q = np.mean(
                np.stack([row["q_ensemble_mean"] for row in rows]), axis=0
            )
            ranked = sorted(RECOVERY_ACTIONS, key=lambda action: mean_q[action], reverse=True)
            common["M11_EVD"]["B2_recovery_actions_ranked_by_mean_Qbar"] = [
                {"action": f"a{action}", "mean_Qbar": float(mean_q[action])}
                for action in ranked
            ]
    return common


def write_outputs(result: dict[str, object]) -> str:
    cell = result["cell"]
    method = result["method"]
    population = result["population"]
    family = result["family"]
    sigma = result["sigma"]
    label = (
        f"{cell}_{slug(population)}_{family}_s{sigma_slug(sigma)}__"
        f"{method}__diagnostic_replay"
    )
    rows = result["logger"].rows
    scalar_columns = [
        "seed",
        "t",
        "obs_t",
        "action_t",
        "x_true_t",
        "x_true_next",
        "reward_t",
        "cost_t",
        "unsafe_next",
    ]
    array_columns: list[str] = []
    if method == "refplan":
        scalar_columns += [
            "belief_mean",
            "belief_entropy",
            "model_entropy",
            "posterior_max",
            "public_extinction_risk",
            "argmax_lambda0",
            "best_sequence_mean",
            "best_sequence_return_sd",
            "best_sequence_reflected",
        ]
        array_columns += [
            "model_posterior",
            "action_scores",
            "member_returns",
            "action_mean_lambda0",
        ]
    elif method == "ogsrl":
        scalar_columns += [
            "predicted_C25",
            "safety_budget",
            "safety_slack",
            "lambda_safety",
            "lambda_ood",
            "ood_probability",
            "ood_guardian_flag",
            "guardian_override",
            "hard_fallback",
            "low_abundance_scale",
            "true_s_safe_evaluator_only",
            "cost_horizon_used",
            "behavior_normalized_cost",
        ]
        array_columns.append("action_probabilities")
    elif method == "bamcts":
        scalar_columns += ["tree_nodes", "tree_nodes_trace", "max_depth"]
        array_columns += ["root_q", "root_visit_counts", "root_model_belief"]
    else:
        scalar_columns += ["argmax_lambda0", "argmax_pessimistic"]
        array_columns += [
            "q_ensemble_mean",
            "q_ensemble_variance",
            "pessimistic_q_score",
        ]

    # M14 schema invariant: filtering fields exist only for RefPlan.
    belief_names = {"belief_mean", "belief_entropy"}
    if method == "refplan":
        assert belief_names.issubset(scalar_columns)
    else:
        assert not belief_names.intersection(scalar_columns)
    if method == "ensemble_value_disagreement_pessimism":
        assert "surrogate_reward_t" not in scalar_columns

    (REPLAY / "logs").mkdir(parents=True, exist_ok=True)
    arrays = {
        column: np.asarray([row[column] for row in rows])
        for column in scalar_columns
    }
    arrays.update(
        {
            column: np.stack([np.asarray(row[column]) for row in rows])
            for column in array_columns
        }
    )
    np.savez_compressed(REPLAY / "logs" / f"{label}.npz", **arrays)
    with gzip.open(
        REPLAY / "logs" / f"{label}.scalars.csv.gz", "wt", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(scalar_columns)
        for row in rows:
            writer.writerow([row[column] for column in scalar_columns])

    (REPLAY / "derived").mkdir(parents=True, exist_ok=True)
    (REPLAY / "derived" / f"{label}.json").write_text(
        json.dumps(derived_metrics(result), indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "cell": cell,
        "side": "general",
        "method": method,
        "parity": result["parity"],
        "elapsed_seconds": result["elapsed_seconds"],
        "recomputed_fits": 0,
        "public_ensemble_fit_at_build": True,
        "dataset_sha256": result["dataset_sha256"],
        "dataset_hash_matches_accepted": True,
        "manifest_index": result["manifest_index"],
        "training_split": result["split_info"],
        "filter": "learned",
        "general_code_root": str(GENERAL_SRC),
        "general_config": str(GENERAL_CONFIG),
        "dataset_path": str(result["cfg"].dataset.output),
        "surrogate_path": str(result["surrogate_path"]),
        "belief_cache_path": str(result["cache_path"]),
        "source_hashes": source_hashes(),
        "logging": {
            "side_effect_free": True,
            "copies_already_computed_diagnostics": True,
            "rng_state_unchanged_after_log_copy": True,
            "belief_columns": "present" if method == "refplan" else "absent",
            "surrogate_reward_t": (
                "absent"
                if method == "ensemble_value_disagreement_pessimism"
                else "not_logged"
            ),
        },
    }
    (REPLAY / "parity").mkdir(parents=True, exist_ok=True)
    (REPLAY / "parity" / f"PARITY_{label}.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=tuple(CELLS))
    parser.add_argument("method", choices=METHODS)
    args = parser.parse_args()
    require_scientific_tables()
    if sha(ACCEPTED_CSV) != ACCEPTED_SHA:
        raise AssertionError("accepted controlling CSV hash changed")
    result = run_cell(args.cell, args.method)
    label = write_outputs(result)
    print(
        json.dumps(
            {
                "label": label,
                "cell": args.cell,
                "method": args.method,
                "parity": result["parity"]["verdict"],
                "max_abs_diff": result["parity"]["max_abs_diff"],
                "elapsed_seconds": round(result["elapsed_seconds"], 3),
                "recomputed_fits": 0,
                "dataset_hash_matches_accepted": True,
            },
            indent=2,
        )
    )
    print(json.dumps(result["parity"]["per_field"], indent=2))
    if result["parity"]["verdict"] != "PASS":
        raise SystemExit(
            "general accepted-cell parity gate failed: "
            f"{result['parity']['max_abs_diff']}"
        )


if __name__ == "__main__":
    main()
