#!/usr/bin/env python3
"""Run one conditionally authorized Arm T task through external exact-state seams.

This recovery runner is deliberately outside ``src/tracks/**``.  It opens only the
sealed derived offline view, never the original truth archive.  At deployment, a
private bridge retains only the evaluator's current abundance and supplies it to a
capability-gated filter/policy wrapper; no future state or other evaluator field is
accepted by either method-facing seam.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/i2b_stageb_arm_t_sigma02_20260809"
ARM_O_REG = DOC / "ARM_O_CANARY_REGISTRATION.json"
ARM_T_REG = DOC / "ARM_T_TASKS.json"
I2A = DOC.parent / "i2_increment_a_exact_state_adapters_20260808"
EXPECTED_SEEDS = [7001, 7051, 7101, 7151, 7201]
END_TO_END = "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY"
ECO = {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def tree_identity(root: Path) -> dict[str, Any]:
    manifest = tree_manifest(root)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(manifest),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": manifest,
    }


class ExactStateBridge:
    """Holds exactly one current raw abundance value and nothing else."""

    def __init__(self) -> None:
        self._current: np.float64 | None = None

    def set_current(self, value: float) -> None:
        current = np.float64(value)
        if not np.isfinite(current) or current < 0.0:
            raise RuntimeError("invalid current exact abundance")
        self._current = current

    def get_current(self) -> float:
        if self._current is None:
            raise RuntimeError("current exact abundance is unavailable")
        return float(self._current)


class RuntimeMetrics:
    def __init__(self) -> None:
        self.episodes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def reset(self, seed: int, state: float) -> None:
        self.current = {
            "seed": int(seed), "initial_state": float(state), "actions": [],
            "states": [float(state)], "safety_penalty_applied": [],
        }
        self.episodes.append(self.current)

    def step(self, action: int, result: Any) -> None:
        if self.current is None:
            raise RuntimeError("metrics step before reset")
        info = result.evaluator_info
        self.current["actions"].append(int(action))
        self.current["states"].append(float(info["state"]))
        self.current["safety_penalty_applied"].append(
            bool(info.get("safety_penalty_applied", False))
        )

    def summaries(self, discount: float, collapse_penalty: float) -> list[dict[str, Any]]:
        rows = []
        for episode in self.episodes:
            actions = episode["actions"]
            states = episode["states"]
            penalties = episode["safety_penalty_applied"]
            counts = np.bincount(actions, minlength=11).astype(int)
            rows.append({
                "seed": episode["seed"],
                "n_steps": len(actions),
                "action_counts": counts.tolist(),
                "distinct_actions": int(np.count_nonzero(counts)),
                "minimum_abundance": float(min(states)),
                "discounted_safety_penalty_contribution": float(sum(
                    (discount ** step) * collapse_penalty * int(flag)
                    for step, flag in enumerate(penalties)
                )),
                "unsafe_state_values_not_serialized": True,
            })
        return rows


class BridgeEnvironment:
    def __init__(self, wrapped: Any, bridge: ExactStateBridge, metrics: RuntimeMetrics):
        self._wrapped = wrapped
        self._bridge = bridge
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def reset(self, seed: int):
        result = self._wrapped.reset(seed)
        state = float(result.evaluator_info["state"])
        self._bridge.set_current(state)
        self._metrics.reset(seed, state)
        return result

    def step(self, action: int):
        result = self._wrapped.step(action)
        self._bridge.set_current(float(result.evaluator_info["state"]))
        self._metrics.step(action, result)
        return result


class ExactCurrentFilter:
    """Runs the registered public filter, then replaces only current particles."""

    def __init__(self, base: Any, bridge: ExactStateBridge, capability: Any,
                 exact_general_belief: Any, sigma: float, scale: float):
        self.base = base
        self.bridge = bridge
        self.capability = capability
        self.exact_general_belief = exact_general_belief
        self.sigma = sigma
        self.scale = scale

    def _adapt(self, belief: Any) -> Any:
        emitted, _receipt = self.exact_general_belief(
            belief,
            self.bridge.get_current(),
            capability=self.capability,
            observation_noise_sigma=self.sigma,
            observation_scale=self.scale,
            private_payload=None,
            oracle_filter=False,
        )
        emitted.diagnostics["exact_current_external_adapter"] = 1.0
        return emitted

    def reset(self, observation: float, seed: int):
        return self._adapt(self.base.reset(observation, seed))

    def update(self, belief: Any, action: int, observation: float):
        return self._adapt(self.base.update(belief, action, observation))


class ExactPolicy:
    """Method-facing wrapper; never accepts next state or evaluator payload."""

    def __init__(self, wrapped: Any, method: str, bridge: ExactStateBridge,
                 capability: Any, adapt_point_mass_belief: Any):
        self._wrapped = wrapped
        self.method = method
        self.bridge = bridge
        self.capability = capability
        self.adapt_point_mass_belief = adapt_point_mass_belief
        self.point_mass_receipts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def name(self) -> str:
        return self._wrapped.name

    def reset(self, seed: int) -> None:
        self._wrapped.reset(seed)

    def _assign_ecological(self, observation: float) -> None:
        raw = self.bridge.get_current()
        if self.method.startswith("plus_"):
            if self._wrapped.internal_beliefs is None:
                self._wrapped._initialize(observation)
            pairs = zip(self._wrapped.pomdps, self._wrapped.internal_beliefs)
            replaced = []
            for pomdp, previous in pairs:
                grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
                probs, receipt = self.adapt_point_mass_belief(
                    self.method, raw, float(pomdp.model.survey_scale), grid,
                    grid * np.float64(pomdp.model.survey_scale),
                )
                replaced.append(replace(previous, probabilities=probs))
                self.point_mass_receipts.append(receipt.__dict__)
            self._wrapped.internal_beliefs = replaced
        else:
            if self._wrapped.internal_belief is None:
                self._wrapped.internal_belief = self._wrapped.pomdp.initial_belief(observation)
            pomdp = self._wrapped.pomdp
            previous = self._wrapped.internal_belief
            grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
            probs, receipt = self.adapt_point_mass_belief(
                self.method, raw, float(pomdp.model.survey_scale), grid,
                grid * np.float64(pomdp.model.survey_scale),
            )
            self._wrapped.internal_belief = replace(previous, probabilities=probs)
            self.point_mass_receipts.append(receipt.__dict__)

    def act(self, belief: Any, observation: float) -> int:
        if self.method in ECO:
            self._assign_ecological(observation)
        return int(self._wrapped.act(belief, observation))

    def observe(self, belief: Any, action: int, result: Any) -> None:
        if self.method in ECO:
            # Preserve public history and capacity while suppressing the noisy
            # observation-likelihood update.  The point mass is reassigned before
            # every action; PLUS candidate weights therefore receive no noisy evidence.
            if self.method.startswith("plus_"):
                updated = []
                for pomdp, previous in zip(self._wrapped.pomdps, self._wrapped.internal_beliefs):
                    updated.append(replace(
                        previous,
                        capacity=pomdp.model.next_capacity(previous.capacity, int(action)),
                        previous_observation=previous.current_observation,
                        current_observation=float(result.observation),
                        timestep=previous.timestep + 1,
                    ))
                self._wrapped.internal_beliefs = updated
            else:
                previous = self._wrapped.internal_belief
                self._wrapped.internal_belief = replace(
                    previous,
                    capacity=self._wrapped.pomdp.model.next_capacity(previous.capacity, int(action)),
                    previous_observation=previous.current_observation,
                    current_observation=float(result.observation),
                    timestep=previous.timestep + 1,
                )
            return
        if self.method in {"refplan", "bamcts"}:
            result = replace(result, observation=self.bridge.get_current())
        self._wrapped.observe(belief, action, result)


def exact_cache(base: Any, states: np.ndarray, next_states: np.ndarray,
                dataset: Any, scale: float, sigma: float, adapter_cls: Any,
                cache_cls: Any) -> tuple[Any, dict[str, Any]]:
    adapter = adapter_cls()
    current, current_receipt = adapter.adapt(
        np.asarray(base.features, dtype=np.float64), states,
        observation_scale=scale, observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(),
        observation_history=dataset.observations.tolist(),
    )
    following, following_receipt = adapter.adapt(
        np.asarray(base.next_features, dtype=np.float64), next_states,
        observation_scale=scale, observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(),
        observation_history=dataset.next_observations.tolist(),
    )
    cache = cache_cls(
        current, following, states.copy(), next_states.copy(),
        {**base.metadata, "arm": "T", "construction": "allowlisted_exact_state"},
    )
    return cache, {
        "current": current_receipt.to_dict(),
        "next": following_receipt.to_dict(),
        "feature_sd_mean": float(np.mean(current[:, 1])),
        "feature_sd_std": float(np.std(current[:, 1])),
    }


def residual_sigmas(policy: Any) -> list[str]:
    values: list[float] = []
    dynamics = getattr(policy, "dynamics", None)
    for member in getattr(dynamics, "members", []):
        if hasattr(member, "residual_sigma"):
            values.append(float(member.residual_sigma))
    for fit in getattr(getattr(policy, "candidate_bank", None), "fits", []):
        values.append(float(fit.model.process_scale))
    if hasattr(policy, "fit_result"):
        values.append(float(policy.fit_result.model.process_scale))
    return [np.float64(value).hex() for value in values]


def run(task_index: int) -> None:
    arm_o = json.loads(ARM_O_REG.read_text(encoding="utf-8"))
    arm_t = json.loads(ARM_T_REG.read_text(encoding="utf-8"))
    if task_index not in range(12):
        raise RuntimeError("task index outside registered 0..11 range")
    task = arm_o["tasks"][task_index]
    task_t = arm_t["tasks"][task_index]
    if task_t != {"index": task_index, "cell": task["cell"], "method": task["method"]}:
        raise RuntimeError("Arm O/T task identity mismatch")
    task_root = OUT / "arm_t_tasks" / task["task_id"]
    if task_root.exists():
        raise RuntimeError(f"task namespace collision: {task_root}")
    task_root.mkdir(parents=True)
    started = time.time()
    receipt_path = task_root / "task_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "i2b_arm_t_task_receipt_v1",
        "task": task_t,
        "arm": "T",
        "status_label": "PROVISIONAL — NOT YET INDEPENDENTLY AUDITED",
        "started_unix": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpu_model": cpu_model(),
        "interpreter": str(Path(sys.executable).resolve()),
        "original_truth_archive_access": False,
        "runtime_next_states_access": False,
        "forbidden_runtime_fields_accessed": [],
    }
    try:
        if "8452Y" not in receipt["cpu_model"]:
            raise RuntimeError(f"pinned CPU unavailable: {receipt['cpu_model']}")
        method_spec = arm_t["method_specs"][task["method"]]
        if Path(sys.executable).resolve() != Path(method_spec["interpreter"]).resolve():
            raise RuntimeError("registered interpreter mismatch")
        shared = Path(task["shared_input_root"])
        public_path = shared / "public.npz"
        derived_path = Path(arm_t["cell_inputs"][task["cell"]]["derived_offline_path"])
        if sha(public_path) != task["public_file_sha256"]:
            raise RuntimeError("public input hash mismatch")
        if sha(derived_path) != arm_t["cell_inputs"][task["cell"]]["derived_offline_sha256"]:
            raise RuntimeError("derived input hash mismatch")

        sys.path.insert(0, str(ROOT / task["track_source"]))
        sys.path.insert(0, str(I2A))
        sys.path.insert(0, str(DOC))
        from adapter_interfaces import ExactStateFeatureAdapter, adapt_point_mass_belief
        from fasttrack_wrappers import exact_general_belief, issue_exact_state_capability
        from real_ecology_benchmark import evaluator as evaluator_module
        from real_ecology_benchmark import pipeline
        from real_ecology_benchmark.backend import resolve_backend_for_workload, set_active_backend
        from real_ecology_benchmark.beliefs import PublicBeliefCache, cache_public_beliefs
        from real_ecology_benchmark.config import load_config, real_environment_like
        from real_ecology_benchmark.dataset import dataset_sha256, load_public
        from real_ecology_benchmark.public_surrogate import fit_public_surrogate
        from real_ecology_benchmark.training_monitor import split_train_holdout, save_training_artifacts

        dataset = load_public(public_path)
        with np.load(derived_path, allow_pickle=False) as derived:
            if tuple(derived.files) != ("states", "next_states", "row_index", "episode_id", "timestep"):
                raise RuntimeError("derived schema mismatch")
            states = np.asarray(derived["states"], dtype=np.float64)
            next_states = np.asarray(derived["next_states"], dtype=np.float64)
            if not np.array_equal(derived["row_index"], np.arange(4000)):
                raise RuntimeError("derived row alignment mismatch")
            if not np.array_equal(derived["episode_id"], dataset.episode_id):
                raise RuntimeError("derived episode alignment mismatch")
            if not np.array_equal(derived["timestep"], dataset.timestep):
                raise RuntimeError("derived timestep alignment mismatch")
        if len(dataset) != 4000 or dataset.num_episodes != 160:
            raise RuntimeError("registered 4000-row/160-episode budget mismatch")

        cfg = load_config(task["config_path"])
        cfg.environment = real_environment_like(cfg.environment, task["population"], "allee")
        cfg.environment = replace(
            cfg.environment, observation_noise_sigma=0.2, reward_mode="safe", expose_rk="hidden"
        )
        cfg.dataset.output = str(public_path)
        cfg.dataset.private_output = str(shared / "private_archive_access_disabled.json")
        cfg.evaluation.output_dir = str(task_root / "evaluation")
        cfg.validate()
        if cfg.evaluation.seeds != EXPECTED_SEEDS or cfg.evaluation.episodes_per_seed != 4:
            raise RuntimeError("evaluation identity mismatch")
        if cfg.evaluation.horizon != 50 or cfg.evaluation.discount != 0.95:
            raise RuntimeError("evaluation convention mismatch")
        backend = set_active_backend(
            resolve_backend_for_workload(cfg.compute, f"arm_t:{task['method']}", cfg.environment)
        )

        # Public surrogate is either copied unchanged for the ecological frozen-fit
        # diagnostic route or rebuilt from the exact-state offline overlay.
        if task["method"] in ECO:
            copied_cache = task_root / "copied_fit_cache"
            shutil.copytree(Path(task["fit_cache_dir"]), copied_cache)
            source_cache_identity = tree_identity(Path(task["fit_cache_dir"]))
            copied_cache_identity = tree_identity(copied_cache)
            if source_cache_identity != copied_cache_identity:
                raise RuntimeError("copied ecological fit cache is not byte-identical")
            cfg.faithful.fit_cache_dir = str(copied_cache)
            surrogate_path = next(shared.glob("public.regime_hidden.public_surrogate.*.npz"))
            from real_ecology_benchmark.public_surrogate import PublicRewardRiskSurrogate
            surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
            exact_dataset = dataset
            preprocessing = {"public_zero_sd_preprocessing": "not consumed; branch is vacuous"}
        else:
            exact_metadata = dict(dataset.metadata)
            source_public_dataset_sha256 = exact_metadata.pop("dataset_sha256", None)
            exact_metadata.update({
                "arm": "T",
                "exact_state_overlay": True,
                "source_public_dataset_sha256": source_public_dataset_sha256,
            })
            exact_dataset = replace(
                dataset,
                observations=states.copy(),
                next_observations=next_states.copy(),
                metadata=exact_metadata,
            )
            exact_dataset.metadata["dataset_sha256"] = dataset_sha256(exact_dataset)
            exact_dataset.validate()
            surrogate = fit_public_surrogate(exact_dataset, seed=cfg.seed + 20_000)
            surrogate_path = task_root / "exact_state_public_surrogate.npz"
            surrogate.save(surrogate_path)
            source_cache_identity = None
            copied_cache_identity = None

        method_context = pipeline._hidden_method_context(cfg, exact_dataset, surrogate)
        factory, _proposal = pipeline.make_filter_factory(
            cfg, exact_dataset, task["filter"], method_context
        )

        if task["method"] in ECO:
            train_dataset = exact_dataset
            train_cache = holdout_dataset = holdout_cache = None
            split_info = {
                "training_enabled": bool(cfg.training.enabled),
                "holdout_fraction": 0.0,
                "fit_history_fraction": float(cfg.faithful.fit.history_fraction),
                "train_transitions": len(train_dataset),
                "train_episodes": train_dataset.num_episodes,
                "faithful_internal_beliefs": True,
            }
        else:
            base_cache = cache_public_beliefs(
                dataset, factory, cfg.seed + 30_000, method_context.observation_scale
            )
            full_cache, preprocessing = exact_cache(
                base_cache, states, next_states, dataset,
                method_context.observation_scale, method_context.observation_noise_sigma,
                ExactStateFeatureAdapter, PublicBeliefCache,
            )
            full_cache.save(task_root / "exact_state_offline_beliefs.npz")
            train_dataset, train_cache, holdout_dataset, holdout_cache, split_info = (
                split_train_holdout(exact_dataset, full_cache, cfg)
            )

        timings: dict[str, float] = {}
        policy, train_cache = pipeline.build_method(
            task["method"], cfg, train_dataset, factory, train_cache, timings,
            holdout_dataset, holdout_cache, split_info, method_context,
        )
        fit_root = task_root / "fit_artifacts"
        fit_root.mkdir()
        fit_summary = policy.save_fit_artifacts(fit_root)
        fit_summary.update(save_training_artifacts(
            getattr(policy, "training_history", None), fit_root, cfg.training
        ))
        fit_identity = tree_identity(fit_root)
        sigmas = residual_sigmas(policy)
        model_posterior = np.asarray(getattr(policy, "posterior", []), dtype=np.float64)
        posterior_entropy = (
            float(-np.sum(model_posterior * np.log(model_posterior + 1e-300)))
            if model_posterior.size else None
        )
        component_receipt = {
            "schema_version": "i2b_arm_t_pre_return_component_receipt_v1",
            "task": task_t,
            "written_before_evaluation": True,
            "offline_rows": 4000,
            "original_truth_archive_access": False,
            "runtime_next_states_access": False,
            "derived_offline_sha256": sha(derived_path),
            "public_file_sha256": sha(public_path),
            "preprocessing": preprocessing,
            "fit_artifact_identity": fit_identity,
            "residual_sigma_float64_hex": sigmas,
            "initial_model_posterior_entropy": posterior_entropy,
            "source_fit_cache_identity": source_cache_identity,
            "copied_fit_cache_identity": copied_cache_identity,
            "learned_artifact_change": bool(method_spec["learned_artifact_change"]),
            "interpretation_label": END_TO_END if method_spec["learned_artifact_change"] else method_spec["formal_frozen_fit_label"],
            "frozen_fit_eligible": False,
            "evd_raw_logged_reward_objective": task["method"] == "ensemble_value_disagreement_pessimism",
            "refplan_sd_disclosure": ({
                "arm_o_registered_standardized_offset": -2.7018 if task["cell"] == "tiger" else -2.7348,
                "arm_t_fit_time_sd_mean": preprocessing.get("feature_sd_mean"),
                "arm_t_fit_time_sd_std": preprocessing.get("feature_sd_std"),
                "arm_t_planner_root_sd": 0.0,
                "arm_t_removes_preexisting_mismatch": True,
            } if task["method"] == "refplan" else None),
            "ogsrl_safety_calibration": (
                "rebuilt from exact-state offline overlay" if task["method"] == "ogsrl" else None
            ),
        }
        component_path = task_root / "pre_return_component_receipt.json"
        strict_json(component_path, component_receipt)

        capability = issue_exact_state_capability(
            purpose="i2b_synthetic_or_future_authorized_runtime"
        )
        bridge = ExactStateBridge()
        runtime_metrics = RuntimeMetrics()
        exact_factory = lambda: ExactCurrentFilter(
            factory(), bridge, capability, exact_general_belief,
            method_context.observation_noise_sigma, method_context.observation_scale,
        )
        wrapped_policy = ExactPolicy(
            policy, task["method"], bridge, capability, adapt_point_mass_belief
        )
        original_make_env = evaluator_module.make_env
        evaluator_module.make_env = lambda config: BridgeEnvironment(
            original_make_env(config), bridge, runtime_metrics
        )
        try:
            evaluator = evaluator_module.ContinuousEvaluator(cfg, exact_factory, "exact_state_external")
            rows = evaluator.run(wrapped_policy)
        finally:
            evaluator_module.make_env = original_make_env

        evaluation_root = task_root / "evaluation_result"
        summary = evaluator.save(rows, evaluation_root, {
            "arm": "T",
            "status_label": "PROVISIONAL — NOT YET INDEPENDENTLY AUDITED",
            "method_bundle_interpretation": component_receipt["interpretation_label"],
            "dataset_rows": 4000,
            "training_split": split_info,
            "fit_diagnostics": policy.fit_diagnostics,
            "pre_return_component_receipt": str(component_path),
            **backend.to_dict(),
        })
        runtime_summary_path = task_root / "runtime_metric_receipt.json"
        strict_json(runtime_summary_path, {
            "schema_version": "i2b_arm_t_runtime_metric_receipt_v1",
            "episodes": runtime_metrics.summaries(
                cfg.evaluation.discount, cfg.environment.collapse_penalty
            ),
            "raw_state_trajectories_serialized": False,
            "runtime_next_states_serialized": False,
        })
        receipt.update({
            "ended_unix": time.time(),
            "duration_seconds": time.time() - started,
            "exit_status": 0,
            "episodes_path": str(evaluation_root / "episodes.csv"),
            "episodes_sha256": sha(evaluation_root / "episodes.csv"),
            "summary_path": str(evaluation_root / "summary.json"),
            "summary_sha256": sha(evaluation_root / "summary.json"),
            "runtime_metric_receipt": str(runtime_summary_path),
            "runtime_metric_receipt_sha256": sha(runtime_summary_path),
            "pre_return_component_receipt": str(component_path),
            "pre_return_component_receipt_sha256": sha(component_path),
            "fit_artifact_identity": fit_identity,
            "point_mass_assignment_count": len(wrapped_policy.point_mass_receipts),
            "result_summary": summary,
        })
        strict_json(receipt_path, receipt)
    except BaseException as exc:
        if not receipt_path.exists():
            receipt.update({
                "ended_unix": time.time(),
                "duration_seconds": time.time() - started,
                "exit_status": int(getattr(exc, "code", 1) or 1),
                "failure": str(exc),
                "traceback": traceback.format_exc(),
            })
            strict_json(receipt_path, receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_index", type=int)
    args = parser.parse_args()
    run(args.task_index)


if __name__ == "__main__":
    main()
