#!/usr/bin/env python3
"""Run one prospectively registered corrected Stage-B Arm O or Arm T task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from corrected_common import (
    ECOLOGICAL,
    END_TO_END,
    EVD,
    EXPECTED_BLOCK_SEEDS,
    LABEL,
    EvaluationEvidence,
    EvidenceEnvironment,
    ExactCurrentFilter,
    ExactPolicy,
    ExactStateBridge,
    PolicyEvidenceProxy,
    activity_from_actions,
    canonical_hash,
    compare_episodes,
    cpu_model,
    exact_cache,
    load_fitted_policy,
    refplan_dispersion,
    residual_diagnostics,
    serialize_fitted_policy,
    sha256_file,
    strict_json,
    tree_identity,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/stageb_sigma02_corrected_20260809"
OLD_DOC = DOC.parent / "i2b_fasttrack_integration_canary_20260808"
I2A = DOC.parent / "i2_increment_a_exact_state_adapters_20260808"
REGISTRATION = DOC / "CORRECTED_STAGEB_REGISTRATION.json"
TASK_MANIFEST = DOC / "CORRECTED_TASK_MANIFEST.json"
PREEXEC = DOC / "PRE_EXECUTION_HASHES.sha256"


def verify_preexecution_manifest() -> None:
    for line in PREEXEC.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"pre-execution identity mismatch: {relative}")


def load_inputs(task: dict[str, Any], registration: dict[str, Any]):
    shared = Path(task["shared_input_root"])
    public_path = shared / "public.npz"
    if sha256_file(public_path) != task["public_file_sha256"]:
        raise RuntimeError("public input hash mismatch")
    if sha256_file(Path(task["accepted_episodes_path"])) != task["accepted_episodes_sha256"]:
        raise RuntimeError("accepted result identity mismatch")
    if sha256_file(Path(task["config_path"])) != task["config_sha256"]:
        raise RuntimeError("configuration identity mismatch")
    surrogate = ROOT / registration["matched_surrogates"][task["cell"]]["path"]
    if sha256_file(surrogate) != registration["matched_surrogates"][task["cell"]]["sha256"]:
        raise RuntimeError("matched Arm O surrogate identity mismatch")
    return shared, public_path, surrogate


def build_config(task: dict[str, Any], task_root: Path, public_path: Path):
    from real_ecology_benchmark.config import load_config, real_environment_like

    cfg = load_config(task["config_path"])
    cfg.environment = real_environment_like(cfg.environment, task["population"], "allee")
    cfg.environment = replace(
        cfg.environment,
        observation_noise_sigma=0.2,
        reward_mode="safe",
        expose_rk="hidden",
    )
    cfg.dataset.output = str(public_path)
    cfg.dataset.private_output = str(
        Path(task["shared_input_root"]) / "private_archive_access_disabled.json"
    )
    cfg.evaluation.output_dir = str(task_root / "evaluation")
    if task["fit_cache_dir"]:
        cfg.faithful.fit_cache_dir = task["fit_cache_dir"]
    cfg.validate()
    if cfg.evaluation.seeds != EXPECTED_BLOCK_SEEDS:
        raise RuntimeError("evaluation block-seed identity mismatch")
    if cfg.evaluation.episodes_per_seed != 4:
        raise RuntimeError("episodes-per-seed mismatch")
    if cfg.evaluation.horizon != 50 or cfg.evaluation.discount != 0.95:
        raise RuntimeError("evaluation horizon/discount mismatch")
    if cfg.environment.num_actions != 11:
        raise RuntimeError("action-set mismatch")
    return cfg


def policy_probe_factory(
    factory: Any,
    observation: float,
    method: str,
    arm: str,
    raw_state: float | None,
    context: Any,
    exact_general_belief: Any | None,
    capability: Any | None,
    adapt_point_mass_belief: Any | None,
):
    def probe(policy: Any) -> dict[str, Any]:
        filt = factory()
        belief = filt.reset(float(observation), 910_001)
        policy.reset(920_001)
        wrapped: Any = policy
        if arm == "T":
            assert raw_state is not None
            if method in ECOLOGICAL:
                bridge = ExactStateBridge()
                bridge.set_current(raw_state)
                wrapped = ExactPolicy(policy, method, bridge, adapt_point_mass_belief)
            else:
                belief, _receipt = exact_general_belief(
                    belief,
                    raw_state,
                    capability=capability,
                    observation_noise_sigma=context.observation_noise_sigma,
                    observation_scale=context.observation_scale,
                    private_payload=None,
                    oracle_filter=False,
                )
        action = int(wrapped.act(belief, float(observation)))
        base = getattr(wrapped, "_wrapped", wrapped)
        predictions: list[list[float]] = []
        dynamics = getattr(base, "dynamics", None)
        for member in getattr(dynamics, "members", [])[:5]:
            predictions.append(
                np.asarray(member.mean_next(
                    np.asarray([belief.mean_state()], dtype=np.float64),
                    np.asarray([0], dtype=np.int64),
                ), dtype=np.float64).tolist()
            )
        if method == EVD:
            features = belief.public_features(context.observation_scale)[None, :]
            mean, variance, _members = base._statistics(features)
            predictions = [mean.tolist(), variance.tolist()]
        return {
            "probe_seed": 920001,
            "input_observation": float(observation),
            "input_exact_state": raw_state if arm == "T" else None,
            "action": action,
            "predictions": predictions,
            "last_diagnostics_sha256": canonical_hash(
                json.loads(json.dumps(
                    getattr(base, "last_diagnostics", {}),
                    default=lambda value: value.tolist()
                    if isinstance(value, np.ndarray) else value.item(),
                ))
            ),
        }
    return probe


def write_component_receipt(
    task_root: Path,
    task: dict[str, Any],
    arm: str,
    artifact: dict[str, Any],
    surrogate_path: Path,
    policy: Any,
    preprocessing: dict[str, Any],
    dispersion: dict[str, Any] | None,
    shared_ecological: bool,
) -> Path:
    residual = residual_diagnostics(policy)
    base = getattr(policy, "_wrapped", policy)
    candidate_process_scales: list[float] = []
    for fit in getattr(getattr(base, "candidate_bank", None), "fits", []):
        candidate_process_scales.append(float(fit.model.process_scale))
    if hasattr(base, "fit_result"):
        candidate_process_scales.append(float(base.fit_result.model.process_scale))
    initial = getattr(base, "posterior", None)
    initial_entropy = None
    if initial is not None:
        weights = np.asarray(initial, dtype=np.float64)
        initial_entropy = float(-np.sum(weights * np.log(weights + 1e-300)))
    label = (
        "FROZEN-FIT/STATE-INPUT-ONLY — IDENTICAL SERIALIZED ECOLOGICAL POLICY"
        if shared_ecological else END_TO_END
    )
    if task["method"] == EVD:
        label = END_TO_END
    receipt = {
        "schema_version": "corrected_pre_return_component_receipt_v1",
        "written_before_evaluation": True,
        "status_label": LABEL,
        "arm": arm,
        "cell": task["cell"],
        "method": task["method"],
        "offline_rows": 4000,
        "fitted_policy_pickle_path": artifact["canonical_pickle_path"],
        "fitted_policy_pickle_sha256": artifact["canonical_pickle_sha256"],
        "fitted_arrays_path": artifact["canonical_arrays_path"],
        "fitted_arrays_sha256": artifact["canonical_arrays_sha256"],
        "artifact_receipt_path": artifact["artifact_receipt_path"],
        "artifact_receipt_sha256": artifact["artifact_receipt_sha256"],
        "matched_reward_surrogate_path": str(surrogate_path),
        "matched_reward_surrogate_sha256": sha256_file(surrogate_path),
        "matched_reward_surrogate_label": "PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE",
        "arm_t_reward_surrogate_fitting_executed": False,
        "residual_sigma": residual,
        "candidate_process_scales": candidate_process_scales,
        "initial_model_or_candidate_posterior_entropy": initial_entropy,
        "preprocessing": preprocessing,
        "refplan_predictive_dispersion": dispersion,
        "interpretation_label": label,
        "shared_ecological_policy_across_arms": shared_ecological,
        "evd_raw_logged_reward_objective": task["method"] == EVD,
        "original_truth_archive_access": False,
        "runtime_next_states_access": False,
        "forbidden_method_fields_accessed": [],
        "method_input_boundary": {
            "public_observation_action_history_preserved": True,
            "arm_t_addition": "current exact raw abundance only" if arm == "T" else "none",
            "evaluator_reward_components": False,
            "safety_threshold": False,
            "allee_family_or_parameters": False,
            "innovations_or_rng_states": False,
        },
    }
    path = task_root / "pre_return_component_receipt.json"
    strict_json(path, receipt)
    return path


def run_arm_o(task: dict[str, Any], task_root: Path, cfg: Any,
              public_path: Path, surrogate_path: Path) -> dict[str, Any]:
    from real_ecology_benchmark import evaluator as evaluator_module
    from real_ecology_benchmark import pipeline
    from real_ecology_benchmark.public_surrogate import PublicRewardRiskSurrogate

    dataset = pipeline.ensure_dataset(cfg, regenerate=False)
    if len(dataset) != 4000 or dataset.num_episodes != 160:
        raise RuntimeError("registered 4000-row/160-episode budget mismatch")
    surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
    if surrogate.public_data_hash != task["public_dataset_sha256"]:
        raise RuntimeError("matched surrogate/public dataset binding mismatch")
    method_context = pipeline._hidden_method_context(cfg, dataset, surrogate)
    artifact_box: dict[str, Any] = {}
    policy_evidence_box: dict[str, PolicyEvidenceProxy] = {}
    evidence = EvaluationEvidence(cfg.evaluation.discount)
    original_build = pipeline.build_method
    original_surrogate = pipeline._load_or_fit_public_surrogate
    original_make_env = evaluator_module.make_env
    original_evaluator = pipeline.ContinuousEvaluator

    def forced_surrogate(_cfg: Any, _dataset: Any):
        return PublicRewardRiskSurrogate.load(surrogate_path), surrogate_path, (
            "prospectively_reconstructed_matched_arm_o"
        )

    def intercepted_build(*args: Any, **kwargs: Any):
        policy, cache = original_build(*args, **kwargs)
        probe = policy_probe_factory(
            args[3], float(dataset.observations[0]), task["method"], "O", None,
            method_context, None, None, None,
        )
        loaded, artifact = serialize_fitted_policy(
            policy, task_root / "fitted_artifact", task["method"], task["cell"],
            "O", surrogate_path, probe,
        )
        dispersion = refplan_dispersion(
            loaded, cache, np.asarray(args[2].actions),
            task_root / "refplan_predictive_dispersion.npz",
        ) if cache is not None else None
        component_path = write_component_receipt(
            task_root, task, "O", artifact, surrogate_path, loaded,
            {"route": "registered noisy public belief features"}, dispersion,
            shared_ecological=task["method"] in ECOLOGICAL,
        )
        proxy = PolicyEvidenceProxy(loaded)
        artifact_box.update({
            "artifact": artifact,
            "component_path": component_path,
            "dispersion": dispersion,
        })
        policy_evidence_box["proxy"] = proxy
        return proxy, cache

    class EvidenceEvaluator(evaluator_module.ContinuousEvaluator):
        def run(self, policy: Any):
            evaluator_module.make_env = lambda config: EvidenceEnvironment(
                original_make_env(config), evidence, None
            )
            try:
                return super().run(policy)
            finally:
                evaluator_module.make_env = original_make_env

    def deny_private(*_args: Any, **_kwargs: Any):
        raise FileNotFoundError("corrected Arm O prohibits private archive access")

    pipeline._load_or_fit_public_surrogate = forced_surrogate
    pipeline.build_method = intercepted_build
    pipeline.ContinuousEvaluator = EvidenceEvaluator
    pipeline.load_private = deny_private
    try:
        summary = pipeline.run_method(task["method"], cfg, task["filter"], regenerate=False)
    finally:
        pipeline._load_or_fit_public_surrogate = original_surrogate
        pipeline.build_method = original_build
        pipeline.ContinuousEvaluator = original_evaluator
        evaluator_module.make_env = original_make_env
    evidence_check = validate_evidence(evidence)
    evidence_path = task_root / "evaluator_timestep_evidence.json"
    strict_json(evidence_path, {
        "schema_version": "corrected_evaluator_timestep_evidence_v1",
        "status_label": LABEL,
        "arm": "O",
        "cell": task["cell"],
        "method": task["method"],
        "evaluator_only_not_method_visible": True,
        "validation": evidence_check,
        "episodes": evidence.episodes,
    })
    policy_evidence_path = task_root / "policy_posterior_and_diagnostics.json"
    strict_json(policy_evidence_path, {
        "schema_version": "corrected_policy_evidence_v1",
        "arm": "O",
        "episodes": policy_evidence_box["proxy"].episodes,
    })
    episodes_path = Path(summary["output_dir"]) / "episodes.csv"
    parity = compare_episodes(episodes_path, Path(task["accepted_episodes_path"]))
    parity_path = task_root / "accepted_arm_o_parity.json"
    strict_json(parity_path, parity)
    all_actions = [
        row["action"] for episode in evidence.episodes for row in episode["timesteps"]
    ]
    activity = activity_from_actions(all_actions)
    return {
        "summary": summary,
        "episodes_path": episodes_path,
        "artifact": artifact_box["artifact"],
        "component_path": artifact_box["component_path"],
        "evidence_path": evidence_path,
        "policy_evidence_path": policy_evidence_path,
        "parity_path": parity_path,
        "parity": parity,
        "activity": activity,
    }


def run_arm_t(task: dict[str, Any], task_root: Path, cfg: Any,
              public_path: Path, surrogate_path: Path,
              manifest: dict[str, Any]) -> dict[str, Any]:
    from adapter_interfaces import ExactStateFeatureAdapter, adapt_point_mass_belief
    from fasttrack_wrappers import exact_general_belief, issue_exact_state_capability
    from real_ecology_benchmark import evaluator as evaluator_module
    from real_ecology_benchmark import pipeline
    from real_ecology_benchmark.backend import resolve_backend_for_workload, set_active_backend
    from real_ecology_benchmark.beliefs import PublicBeliefCache, cache_public_beliefs
    from real_ecology_benchmark.dataset import dataset_sha256, load_public
    from real_ecology_benchmark.public_surrogate import PublicRewardRiskSurrogate
    from real_ecology_benchmark.training_monitor import split_train_holdout

    dataset = load_public(public_path)
    derived_spec = manifest["derived_inputs"][task["cell"]]
    derived_path = ROOT / derived_spec["path"]
    if sha256_file(derived_path) != derived_spec["sha256"]:
        raise RuntimeError("sealed derived input hash mismatch")
    with np.load(derived_path, allow_pickle=False) as derived:
        if tuple(derived.files) != (
            "states", "next_states", "row_index", "episode_id", "timestep"
        ):
            raise RuntimeError("derived allowlist schema mismatch")
        states = np.asarray(derived["states"], dtype=np.float64)
        next_states = np.asarray(derived["next_states"], dtype=np.float64)
        if not np.array_equal(derived["row_index"], np.arange(4000)):
            raise RuntimeError("derived row alignment mismatch")
        if not np.array_equal(derived["episode_id"], dataset.episode_id):
            raise RuntimeError("derived episode alignment mismatch")
        if not np.array_equal(derived["timestep"], dataset.timestep):
            raise RuntimeError("derived timestep alignment mismatch")
    if len(states) != 4000 or not np.isfinite(states).all() or not np.isfinite(next_states).all():
        raise RuntimeError("derived abundance content mismatch")
    surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
    if surrogate.public_data_hash != task["public_dataset_sha256"]:
        raise RuntimeError("matched surrogate/public data binding mismatch")
    metadata = dict(dataset.metadata)
    metadata["source_public_dataset_sha256"] = metadata.pop("dataset_sha256", None)
    metadata.update({"arm": "T", "exact_state_overlay": True})
    exact_dataset = replace(
        dataset,
        observations=states.copy(),
        next_observations=next_states.copy(),
        metadata=metadata,
    )
    exact_dataset.metadata["dataset_sha256"] = dataset_sha256(exact_dataset)
    exact_dataset.validate()
    method_context = pipeline._hidden_method_context(cfg, exact_dataset, surrogate)
    factory, _proposal = pipeline.make_filter_factory(
        cfg, exact_dataset, task["filter"], method_context
    )
    capability = issue_exact_state_capability(
        purpose="corrected_stageb_sigma02_registered_runtime"
    )
    preprocessing: dict[str, Any]
    split_info: dict[str, Any]
    dispersion = None
    if task["method"] in ECOLOGICAL:
        arm_o_root = OUT / "arm_o_tasks" / task["task_id"]
        o_receipt = json.loads((arm_o_root / "task_receipt.json").read_text(encoding="utf-8"))
        artifact = o_receipt["fitted_artifact"]
        policy = load_fitted_policy(
            Path(artifact["canonical_pickle_path"]), artifact["canonical_pickle_sha256"]
        )
        probe = policy_probe_factory(
            factory, float(dataset.observations[0]), task["method"], "T",
            float(states[0]), method_context, exact_general_belief, capability,
            adapt_point_mass_belief,
        )
        left = load_fitted_policy(
            Path(artifact["canonical_pickle_path"]), artifact["canonical_pickle_sha256"]
        )
        right = load_fitted_policy(
            Path(artifact["canonical_pickle_path"]), artifact["canonical_pickle_sha256"]
        )
        if canonical_hash(probe(left)) != canonical_hash(probe(right)):
            raise RuntimeError("shared ecological policy exact-state action reload mismatch")
        preprocessing = {
            "route": "identical serialized ecological policy; runtime point-mass only",
            "raw_to_latent": "raw_abundance / fitted survey_scale then nearest registered raw-state bin",
        }
        split_info = {
            "train_transitions": 4000,
            "train_episodes": 160,
            "policy_fitting_in_arm_t": False,
            "shared_arm_o_policy_sha256": artifact["canonical_pickle_sha256"],
        }
        shared_ecological = True
    else:
        base_cache = cache_public_beliefs(
            dataset, factory, cfg.seed + 30_000, method_context.observation_scale
        )
        full_cache, preprocessing = exact_cache(
            base_cache, states, next_states, dataset,
            method_context.observation_scale,
            method_context.observation_noise_sigma,
            ExactStateFeatureAdapter,
            PublicBeliefCache,
        )
        cache_path = task_root / "exact_state_offline_beliefs.npz"
        full_cache.save(cache_path)
        preprocessing["exact_cache_path"] = str(cache_path)
        preprocessing["exact_cache_sha256"] = sha256_file(cache_path)
        train_dataset, train_cache, holdout_dataset, holdout_cache, split_info = (
            split_train_holdout(exact_dataset, full_cache, cfg)
        )
        policy, _ = pipeline.build_method(
            task["method"], cfg, train_dataset, factory, train_cache, {},
            holdout_dataset, holdout_cache, split_info, method_context,
        )
        probe = policy_probe_factory(
            factory, float(dataset.observations[0]), task["method"], "T",
            float(states[0]), method_context, exact_general_belief, capability,
            adapt_point_mass_belief,
        )
        policy, artifact = serialize_fitted_policy(
            policy, task_root / "fitted_artifact", task["method"], task["cell"],
            "T", surrogate_path, probe,
        )
        dispersion = refplan_dispersion(
            policy, train_cache, np.asarray(train_dataset.actions),
            task_root / "refplan_predictive_dispersion.npz",
        )
        shared_ecological = False
    component_path = write_component_receipt(
        task_root, task, "T", artifact, surrogate_path, policy,
        preprocessing, dispersion, shared_ecological,
    )
    # Future-state arrays are offline fitting targets only and cease to exist in the
    # runner before evaluation.  The runtime bridge contains one current scalar.
    del next_states
    del exact_dataset
    bridge = ExactStateBridge()
    exact_factory = lambda: ExactCurrentFilter(
        factory(), bridge, capability, exact_general_belief,
        method_context.observation_noise_sigma, method_context.observation_scale,
    )
    exact_policy = ExactPolicy(policy, task["method"], bridge, adapt_point_mass_belief)
    policy_proxy = PolicyEvidenceProxy(exact_policy)
    evidence = EvaluationEvidence(cfg.evaluation.discount)
    original_make_env = evaluator_module.make_env
    evaluator_module.make_env = lambda config: EvidenceEnvironment(
        original_make_env(config), evidence, bridge
    )
    try:
        evaluator = evaluator_module.ContinuousEvaluator(
            cfg, exact_factory, "exact_state_context_preserving_external"
        )
        rows = evaluator.run(policy_proxy)
    finally:
        evaluator_module.make_env = original_make_env
    evaluation_root = task_root / "evaluation_result"
    summary = evaluator.save(rows, evaluation_root, {
        "arm": "T",
        "status_label": LABEL,
        "method_bundle_interpretation": END_TO_END
        if task["method"] not in ECOLOGICAL
        else "FROZEN-FIT/STATE-INPUT-ONLY — IDENTICAL SERIALIZED ECOLOGICAL POLICY",
        "dataset_rows": 4000,
        "training_split": split_info,
        "fit_diagnostics": getattr(policy, "fit_diagnostics", {}),
        "matched_reward_surrogate_sha256": sha256_file(surrogate_path),
        **set_active_backend(
            resolve_backend_for_workload(
                cfg.compute, f"corrected_arm_t:{task['method']}", cfg.environment
            )
        ).to_dict(),
    })
    evidence_check = validate_evidence(evidence)
    evidence_path = task_root / "evaluator_timestep_evidence.json"
    strict_json(evidence_path, {
        "schema_version": "corrected_evaluator_timestep_evidence_v1",
        "status_label": LABEL,
        "arm": "T",
        "cell": task["cell"],
        "method": task["method"],
        "evaluator_only_not_method_visible": True,
        "validation": evidence_check,
        "episodes": evidence.episodes,
    })
    policy_evidence_path = task_root / "policy_posterior_and_diagnostics.json"
    strict_json(policy_evidence_path, {
        "schema_version": "corrected_policy_evidence_v1",
        "arm": "T",
        "episodes": policy_proxy.episodes,
    })
    all_actions = [
        row["action"] for episode in evidence.episodes for row in episode["timesteps"]
    ]
    return {
        "summary": summary,
        "episodes_path": evaluation_root / "episodes.csv",
        "artifact": artifact,
        "component_path": component_path,
        "evidence_path": evidence_path,
        "policy_evidence_path": policy_evidence_path,
        "parity_path": None,
        "parity": None,
        "activity": activity_from_actions(all_actions),
        "bridge_receipt": {"current_scalar_writes": bridge.writes, "current_scalar_reads": bridge.reads},
        "point_mass_assignment_count": len(exact_policy.point_mass_receipts),
    }


def run(arm: str, task_index: int) -> None:
    verify_preexecution_manifest()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    if task_index not in range(12):
        raise RuntimeError("task index outside registered 0..11 range")
    task = manifest["tasks"][task_index]
    if task["index"] != task_index:
        raise RuntimeError("task-manifest order mismatch")
    task_root = OUT / f"arm_{arm.lower()}_tasks" / task["task_id"]
    if task_root.exists():
        raise RuntimeError(f"task namespace collision: {task_root}")
    task_root.mkdir(parents=True)
    started = time.time()
    receipt_path = task_root / "task_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "corrected_stageb_task_receipt_v1",
        "status_label": LABEL,
        "arm": arm,
        "task": task,
        "started_unix": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "cpu_model": cpu_model(),
        "interpreter": str(Path(sys.executable).resolve()),
        "original_truth_archive_access": False,
        "forbidden_truth_fields_accessed": [],
        "runtime_next_states_access": False,
    }
    try:
        if "8452Y" not in receipt["cpu_model"]:
            raise RuntimeError(f"pinned CPU unavailable: {receipt['cpu_model']}")
        if Path(sys.executable).resolve() != Path(task["interpreter"]).resolve():
            raise RuntimeError("registered interpreter mismatch")
        shared, public_path, surrogate_path = load_inputs(task, registration)
        sys.path.insert(0, str(ROOT / task["track_source"]))
        sys.path.insert(0, str(I2A))
        sys.path.insert(0, str(OLD_DOC))
        cfg = build_config(task, task_root, public_path)
        from real_ecology_benchmark.backend import resolve_backend_for_workload, set_active_backend
        set_active_backend(resolve_backend_for_workload(
            cfg.compute, f"corrected_arm_{arm.lower()}:{task['method']}", cfg.environment
        ))
        result = (
            run_arm_o(task, task_root, cfg, public_path, surrogate_path)
            if arm == "O"
            else run_arm_t(task, task_root, cfg, public_path, surrogate_path, manifest)
        )
        if arm == "O" and result["parity"]["result"] != "PASS":
            raise RuntimeError("corrected Arm O accepted-result parity failed")
        receipt.update({
            "ended_unix": time.time(),
            "duration_seconds": time.time() - started,
            "exit_status": 0,
            "episodes_path": str(result["episodes_path"]),
            "episodes_sha256": sha256_file(result["episodes_path"]),
            "summary_path": str(Path(result["summary"]["output_dir"]) / "summary.json")
            if "output_dir" in result["summary"] else str(task_root / "evaluation_result/summary.json"),
            "fitted_artifact": result["artifact"],
            "pre_return_component_receipt": str(result["component_path"]),
            "pre_return_component_receipt_sha256": sha256_file(result["component_path"]),
            "evaluator_timestep_evidence": str(result["evidence_path"]),
            "evaluator_timestep_evidence_sha256": sha256_file(result["evidence_path"]),
            "policy_posterior_and_diagnostics": str(result["policy_evidence_path"]),
            "policy_posterior_and_diagnostics_sha256": sha256_file(result["policy_evidence_path"]),
            "accepted_arm_o_parity": result["parity"],
            "accepted_arm_o_parity_path": str(result["parity_path"])
            if result["parity_path"] is not None else None,
            "activity": result["activity"],
            "matched_reward_surrogate_sha256": sha256_file(surrogate_path),
            "shared_fit_cache_identity": tree_identity(Path(task["fit_cache_dir"]))
            if task["fit_cache_dir"] else None,
            "bridge_receipt": result.get("bridge_receipt"),
            "point_mass_assignment_count": result.get("point_mass_assignment_count", 0),
            "no_retry_performed": True,
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
    parser.add_argument("arm", choices=("O", "T"))
    parser.add_argument("task_index", type=int)
    args = parser.parse_args()
    run(args.arm, args.task_index)


if __name__ == "__main__":
    main()
