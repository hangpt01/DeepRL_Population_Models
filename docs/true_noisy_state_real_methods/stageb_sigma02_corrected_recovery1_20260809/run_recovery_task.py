#!/usr/bin/env python3
"""One corrected recovery task; invoked only by the frozen Slurm arrays."""

from __future__ import annotations

import argparse
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

from recovery_common import (
    BLOCK_SEEDS, ECO, END_TO_END, EVD, LABEL, TRANSITION_METHODS,
    CurrentStateBridge, EvidenceEnv, EvidenceLedger, ExactFilter, ExactPolicy,
    PolicyLedger, activity, canonical_sha, compare_accepted, cpu_model,
    load_fitted, make_exact_cache, residual_receipt, save_fitted, sha,
    strict_json, tree_identity, verify_hash_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
OUT = ROOT / "outputs/stageb_sigma02_corrected_recovery1_20260809"
I2A = DOC.parent / "i2_increment_a_exact_state_adapters_20260808"
FAST = DOC.parent / "i2b_fasttrack_integration_canary_20260808"
REG = DOC / "CORRECTED_STAGEB_REGISTRATION.json"
TASKS = DOC / "CORRECTED_TASK_MANIFEST.json"
HASHES = DOC / "PRE_EXECUTION_HASHES.sha256"


def configure(task: dict[str, Any], task_root: Path, public_path: Path) -> Any:
    from real_ecology_benchmark.config import load_config, real_environment_like
    cfg = load_config(task["config_path"])
    cfg.environment = real_environment_like(cfg.environment, task["population"], "allee")
    cfg.environment = replace(cfg.environment, observation_noise_sigma=0.2,
                              reward_mode="safe", expose_rk="hidden")
    cfg.dataset.output = str(public_path)
    cfg.dataset.private_output = str(Path(task["shared_input_root"]) / "private_archive_access_disabled.json")
    cfg.evaluation.output_dir = str(task_root / "evaluation")
    if task["fit_cache_dir"]:
        cfg.faithful.fit_cache_dir = task["fit_cache_dir"]
    cfg.validate()
    if cfg.evaluation.seeds != BLOCK_SEEDS or cfg.evaluation.episodes_per_seed != 4:
        raise RuntimeError("evaluation identity mismatch")
    if (cfg.evaluation.horizon, cfg.evaluation.discount, cfg.environment.num_actions) != (50, 0.95, 11):
        raise RuntimeError("evaluator convention mismatch")
    return cfg


def probe_factory(factory: Any, observation: float, method: str, arm: str,
                  state: float | None, context: Any, exact_adapter: Any = None,
                  capability: Any = None, point_adapter: Any = None):
    def probe(policy: Any) -> dict[str, Any]:
        belief = factory().reset(observation, 910001)
        policy.reset(920001)
        acting: Any = policy
        if arm == "T":
            if state is None:
                raise RuntimeError("exact probe lacks current state")
            if method in ECO:
                bridge = CurrentStateBridge(); bridge.set(state)
                acting = ExactPolicy(policy, method, bridge, point_adapter)
            else:
                belief, _ = exact_adapter(
                    belief, state, capability=capability,
                    observation_noise_sigma=context.observation_noise_sigma,
                    observation_scale=context.observation_scale,
                    private_payload=None, oracle_filter=False,
                )
        action = int(acting.act(belief, observation))
        base = getattr(acting, "_policy", acting)
        predictions = []
        for member in getattr(getattr(base, "dynamics", None), "members", [])[:5]:
            predictions.append(np.asarray(member.mean_next(
                np.asarray([belief.mean_state()]), np.asarray([0])
            )).tolist())
        if method == EVD:
            features = belief.public_features(context.observation_scale)[None, :]
            mean, variance, _ = base._statistics(features)
            predictions = [mean.tolist(), variance.tolist()]
        diagnostics = json.loads(json.dumps(getattr(base, "last_diagnostics", {}),
                                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))
        return {"seed": 920001, "arm": arm, "action": action,
                "observation": observation, "state": state if arm == "T" else None,
                "predictions": predictions, "diagnostics_sha256": canonical_sha(diagnostics)}
    return probe


def ref_dispersion(policy: Any, cache: Any, actions: np.ndarray, path: Path) -> dict[str, Any] | None:
    if policy.name != "refplan":
        return None
    values = np.vstack([member.mean_next(cache.mean_states, actions)
                        for member in policy.dynamics.members])
    variance = np.var(values, axis=0)
    np.savez(path, per_member_predictions=values, ensemble_variance=variance,
             input_mean_states=cache.mean_states, actions=actions)
    return {"path": str(path), "sha256": sha(path), "members": int(values.shape[0]),
            "rows": int(values.shape[1]), "variance_mean": float(variance.mean()),
            "variance_median": float(np.median(variance)),
            "variance_q95": float(np.quantile(variance, 0.95)),
            "confound": "fitted predictive dispersion is bundled with the information intervention"}


def component_receipt(task_root: Path, task: dict[str, Any], arm: str,
                      artifact: dict[str, Any], surrogate: Path, policy: Any,
                      preprocessing: dict[str, Any], dispersion: dict[str, Any] | None,
                      ecological_shared: bool) -> Path:
    weights = getattr(policy, "posterior", None)
    initial_entropy = None if weights is None else float(-np.sum(
        np.asarray(weights) * np.log(np.asarray(weights) + 1e-300)))
    process_scales = [float(fit.model.process_scale)
                      for fit in getattr(getattr(policy, "candidate_bank", None), "fits", [])]
    if hasattr(policy, "fit_result"):
        process_scales.append(float(policy.fit_result.model.process_scale))
    path = task_root / "pre_return_component_receipt.json"
    strict_json(path, {
        "schema_version": "corrected_pre_return_component_v1", "written_before_evaluation": True,
        "status_label": LABEL, "cell": task["cell"], "method": task["method"], "arm": arm,
        "offline_rows": 4000, "fitted_policy_pickle": artifact["pickle_path"],
        "fitted_policy_sha256": artifact["pickle_sha256"],
        "fitted_parameter_path": artifact["parameter_path"],
        "fitted_parameter_sha256": artifact["parameter_sha256"],
        "artifact_receipt": artifact["receipt_path"], "artifact_receipt_sha256": artifact["receipt_sha256"],
        "matched_surrogate": str(surrogate), "matched_surrogate_sha256": sha(surrogate),
        "matched_surrogate_label": "PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE",
        "arm_t_surrogate_fit_executed": False, "residual_sigma": residual_receipt(policy),
        "candidate_process_scales": process_scales, "initial_posterior_entropy": initial_entropy,
        "preprocessing": preprocessing, "refplan_predictive_dispersion": dispersion,
        "interpretation_label": (
            "FROZEN-FIT/STATE-INPUT-ONLY — IDENTICAL SERIALIZED ECOLOGICAL POLICY"
            if ecological_shared else END_TO_END),
        "shared_ecological_policy": ecological_shared, "evd_raw_logged_rewards": task["method"] == EVD,
        "original_truth_archive_access": False, "runtime_next_states_access": False,
        "forbidden_method_fields_accessed": [],
        "method_boundary": {"public_history_preserved": True,
                            "exact_current_abundance": arm == "T",
                            "reward_components": False, "safety_threshold": False,
                            "allee_family_parameters": False, "innovations_rng": False},
    })
    return path


def save_ledgers(task_root: Path, arm: str, task: dict[str, Any],
                 evidence: EvidenceLedger, policy: PolicyLedger) -> tuple[Path, Path, dict[str, Any]]:
    if len(evidence.episodes) != 20 or len(policy.episodes) != 20:
        raise RuntimeError("incomplete evaluation evidence")
    errors = []
    for episode in evidence.episodes:
        for row in episode["timesteps"]:
            errors.append(abs(row["benefit_reward_term"] + row["action_cost_term"]
                              + row["safety_penalty_term"] - row["total_true_reward"]))
    validation = {"episodes": 20, "timesteps": sum(len(x["timesteps"]) for x in evidence.episodes),
                  "maximum_reward_reconstruction_error": max(errors, default=0.0),
                  "method_received_evaluator_only_fields": False,
                  "rng_state_hashes_and_call_counts_complete": True}
    epath = task_root / "evaluator_timestep_evidence.json"
    ppath = task_root / "policy_posterior_and_diagnostics.json"
    strict_json(epath, {"schema_version": "corrected_timestep_evidence_v1", "status_label": LABEL,
                        "cell": task["cell"], "method": task["method"], "arm": arm,
                        "evaluator_only_not_method_visible": True, "validation": validation,
                        "episodes": evidence.episodes})
    strict_json(ppath, {"schema_version": "corrected_policy_evidence_v1", "cell": task["cell"],
                        "method": task["method"], "arm": arm, "episodes": policy.episodes})
    return epath, ppath, validation


def arm_o(task: dict[str, Any], task_root: Path, cfg: Any,
          surrogate_path: Path) -> dict[str, Any]:
    from real_ecology_benchmark import evaluator as evaluator_module, pipeline
    from real_ecology_benchmark.public_surrogate import PublicRewardRiskSurrogate
    dataset = pipeline.ensure_dataset(cfg, False)
    if len(dataset) != 4000 or dataset.num_episodes != 160:
        raise RuntimeError("offline budget mismatch")
    surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
    if surrogate.public_data_hash != task["public_dataset_sha256"]:
        raise RuntimeError("surrogate public binding mismatch")
    context = pipeline._hidden_method_context(cfg, dataset, surrogate)
    evidence = EvidenceLedger(cfg.evaluation.discount)
    box: dict[str, Any] = {}
    original_build, original_surrogate = pipeline.build_method, pipeline._load_or_fit_public_surrogate
    original_evaluator, original_make_env = pipeline.ContinuousEvaluator, evaluator_module.make_env

    def fixed_surrogate(_cfg: Any, _dataset: Any):
        return PublicRewardRiskSurrogate.load(surrogate_path), surrogate_path, "prospectively_reconstructed_matched_arm_o"

    def build(*args: Any, **kwargs: Any):
        fitted, cache = original_build(*args, **kwargs)
        probe = probe_factory(args[3], float(dataset.observations[0]), task["method"],
                              "O", None, context)
        fitted, artifact = save_fitted(fitted, task_root / "fitted_artifact", task["method"],
                                       task["cell"], "O", surrogate_path, probe)
        dispersion = ref_dispersion(fitted, cache, np.asarray(args[2].actions),
                                    task_root / "refplan_predictive_dispersion.npz") if cache is not None else None
        cpath = component_receipt(task_root, task, "O", artifact, surrogate_path, fitted,
                                  {"route": "registered noisy public features"}, dispersion,
                                  task["method"] in ECO)
        ledger = PolicyLedger(fitted)
        box.update(artifact=artifact, component=cpath, policy_ledger=ledger)
        return ledger, cache

    class Evaluator(evaluator_module.ContinuousEvaluator):
        def run(self, policy: Any):
            evaluator_module.make_env = lambda config: EvidenceEnv(original_make_env(config), evidence)
            try:
                return super().run(policy)
            finally:
                evaluator_module.make_env = original_make_env

    def deny(*_a: Any, **_k: Any):
        raise FileNotFoundError("private archive denied")

    pipeline.build_method, pipeline._load_or_fit_public_surrogate = build, fixed_surrogate
    pipeline.ContinuousEvaluator, pipeline.load_private = Evaluator, deny
    try:
        summary = pipeline.run_method(task["method"], cfg, task["filter"], False)
    finally:
        pipeline.build_method, pipeline._load_or_fit_public_surrogate = original_build, original_surrogate
        pipeline.ContinuousEvaluator, evaluator_module.make_env = original_evaluator, original_make_env
    epath, ppath, validation = save_ledgers(task_root, "O", task, evidence, box["policy_ledger"])
    episodes = Path(summary["output_dir"]) / "episodes.csv"
    parity = compare_accepted(episodes, Path(task["accepted_episodes_path"]))
    parity_path = task_root / "accepted_arm_o_parity.json"; strict_json(parity_path, parity)
    return {"summary": summary, "episodes": episodes, "artifact": box["artifact"],
            "component": box["component"], "evidence": epath, "policy_evidence": ppath,
            "parity": parity, "parity_path": parity_path,
            "activity": activity([row["action"] for ep in evidence.episodes for row in ep["timesteps"]]),
            "validation": validation}


def arm_t(task: dict[str, Any], task_root: Path, cfg: Any, public_path: Path,
          surrogate_path: Path, registration: dict[str, Any]) -> dict[str, Any]:
    from adapter_interfaces import ExactStateFeatureAdapter, adapt_point_mass_belief
    from fasttrack_wrappers import exact_general_belief, issue_exact_state_capability
    from real_ecology_benchmark import evaluator as evaluator_module, pipeline
    from real_ecology_benchmark.beliefs import PublicBeliefCache, cache_public_beliefs
    from real_ecology_benchmark.dataset import dataset_sha256, load_public
    from real_ecology_benchmark.public_surrogate import PublicRewardRiskSurrogate
    from real_ecology_benchmark.training_monitor import split_train_holdout
    dataset = load_public(public_path)
    spec = registration["derived_inputs"][task["cell"]]
    derived_path = ROOT / spec["path"]
    if sha(derived_path) != spec["sha256"]:
        raise RuntimeError("derived input hash mismatch")
    with np.load(derived_path, allow_pickle=False) as data:
        if tuple(data.files) != ("states", "next_states", "row_index", "episode_id", "timestep"):
            raise RuntimeError("derived schema mismatch")
        states, next_states = np.asarray(data["states"]), np.asarray(data["next_states"])
        if not np.array_equal(data["row_index"], np.arange(4000)) or not np.array_equal(data["episode_id"], dataset.episode_id) or not np.array_equal(data["timestep"], dataset.timestep):
            raise RuntimeError("derived/public alignment mismatch")
    surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
    if surrogate.public_data_hash != task["public_dataset_sha256"]:
        raise RuntimeError("matched surrogate binding mismatch")
    metadata = dict(dataset.metadata); metadata["source_public_dataset_sha256"] = metadata.pop("dataset_sha256", None)
    metadata.update({"arm": "T", "allowlisted_exact_state": True})
    exact_dataset = replace(dataset, observations=states.copy(), next_observations=next_states.copy(), metadata=metadata)
    exact_dataset.metadata["dataset_sha256"] = dataset_sha256(exact_dataset); exact_dataset.validate()
    context = pipeline._hidden_method_context(cfg, exact_dataset, surrogate)
    factory, _ = pipeline.make_filter_factory(cfg, exact_dataset, task["filter"], context)
    capability = issue_exact_state_capability(purpose="corrected_stageb_recovery1_sigma02")
    if task["method"] in ECO:
        oreceipt = json.loads((OUT / "arm_o_tasks" / task["task_id"] / "task_receipt.json").read_text())
        artifact = oreceipt["fitted_artifact"]
        policy = load_fitted(Path(artifact["pickle_path"]), artifact["pickle_sha256"])
        probe = probe_factory(factory, float(dataset.observations[0]), task["method"], "T",
                              float(states[0]), context, exact_general_belief, capability,
                              adapt_point_mass_belief)
        if canonical_sha(probe(load_fitted(Path(artifact["pickle_path"]), artifact["pickle_sha256"]))) != canonical_sha(probe(load_fitted(Path(artifact["pickle_path"]), artifact["pickle_sha256"]))):
            raise RuntimeError("shared ecological exact-action reload mismatch")
        preprocessing = {"route": "identical Arm O serialized PBVI policy; runtime point mass only",
                         "conversion": "raw abundance / fitted survey_scale, checked in raw-grid units"}
        split = {"offline_rows": 4000, "arm_t_policy_fit": False,
                 "shared_policy_sha256": artifact["pickle_sha256"]}
        shared = True
    else:
        base = cache_public_beliefs(dataset, factory, cfg.seed + 30000, context.observation_scale)
        full, preprocessing = make_exact_cache(base, states, next_states, dataset,
                                                context.observation_scale,
                                                context.observation_noise_sigma,
                                                ExactStateFeatureAdapter, PublicBeliefCache)
        cache_path = task_root / "exact_offline_beliefs.npz"; full.save(cache_path)
        preprocessing.update(cache_path=str(cache_path), cache_sha256=sha(cache_path))
        train, train_cache, holdout, holdout_cache, split = split_train_holdout(exact_dataset, full, cfg)
        policy, _ = pipeline.build_method(task["method"], cfg, train, factory, train_cache, {},
                                          holdout, holdout_cache, split, context)
        probe = probe_factory(factory, float(dataset.observations[0]), task["method"], "T",
                              float(states[0]), context, exact_general_belief, capability,
                              adapt_point_mass_belief)
        policy, artifact = save_fitted(policy, task_root / "fitted_artifact", task["method"],
                                       task["cell"], "T", surrogate_path, probe)
        dispersion = ref_dispersion(policy, train_cache, np.asarray(train.actions),
                                    task_root / "refplan_predictive_dispersion.npz")
        shared = False
    dispersion = None if task["method"] != "refplan" else locals().get("dispersion")
    cpath = component_receipt(task_root, task, "T", artifact, surrogate_path, policy,
                              preprocessing, dispersion, shared)
    del next_states, exact_dataset
    bridge = CurrentStateBridge()
    exact_factory = lambda: ExactFilter(factory(), bridge, capability, exact_general_belief,
                                        context.observation_noise_sigma, context.observation_scale)
    acting = ExactPolicy(policy, task["method"], bridge, adapt_point_mass_belief)
    policy_ledger = PolicyLedger(acting); evidence = EvidenceLedger(cfg.evaluation.discount)
    original_make_env = evaluator_module.make_env
    evaluator_module.make_env = lambda config: EvidenceEnv(original_make_env(config), evidence, bridge)
    try:
        evaluator = evaluator_module.ContinuousEvaluator(cfg, exact_factory, "exact_current_context_preserving")
        rows = evaluator.run(policy_ledger)
    finally:
        evaluator_module.make_env = original_make_env
    evaluation_root = task_root / "evaluation_result"
    summary = evaluator.save(rows, evaluation_root, {"arm": "T", "status_label": LABEL,
        "output_dir": str(evaluation_root), "training_split": split,
        "fit_diagnostics": getattr(policy, "fit_diagnostics", {}),
        "interpretation_label": "FROZEN-FIT/STATE-INPUT-ONLY — IDENTICAL SERIALIZED ECOLOGICAL POLICY" if shared else END_TO_END,
        "matched_surrogate_sha256": sha(surrogate_path)})
    epath, ppath, validation = save_ledgers(task_root, "T", task, evidence, policy_ledger)
    return {"summary": summary, "episodes": evaluation_root / "episodes.csv", "artifact": artifact,
            "component": cpath, "evidence": epath, "policy_evidence": ppath,
            "parity": None, "parity_path": None,
            "activity": activity([row["action"] for ep in evidence.episodes for row in ep["timesteps"]]),
            "validation": validation, "bridge": {"reads": bridge.reads, "writes": bridge.writes},
            "point_mass_assignments": len(acting.point_mass_receipts)}


def run(arm: str, index: int) -> None:
    verify_hash_manifest(ROOT, HASHES)
    registration = json.loads(REG.read_text())
    tasks = json.loads(TASKS.read_text())
    if index not in range(12) or tasks["tasks"][index]["index"] != index:
        raise RuntimeError("task index mismatch")
    task = tasks["tasks"][index]
    task_root = OUT / f"arm_{arm.lower()}_tasks" / task["task_id"]
    if task_root.exists():
        raise RuntimeError(f"task collision: {task_root}")
    task_root.mkdir(parents=True)
    started = time.time(); receipt_path = task_root / "task_receipt.json"
    receipt: dict[str, Any] = {"schema_version": "corrected_recovery_task_v1", "status_label": LABEL,
        "arm": arm, "task": task, "started_unix": started, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"), "hostname": platform.node(),
        "cpu_model": cpu_model(), "interpreter": str(Path(sys.executable).resolve()),
        "original_truth_archive_access": False, "forbidden_truth_fields_accessed": [],
        "runtime_next_states_access": False, "failed_partial_namespace_access": False}
    try:
        if "8452Y" not in receipt["cpu_model"]:
            raise RuntimeError("pinned Intel Xeon Platinum 8452Y unavailable")
        if Path(sys.executable).resolve() != Path(task["interpreter"]).resolve():
            raise RuntimeError("registered interpreter mismatch")
        public_path = Path(task["shared_input_root"]) / "public.npz"
        if sha(public_path) != task["public_file_sha256"] or sha(Path(task["config_path"])) != task["config_sha256"] or sha(Path(task["accepted_episodes_path"])) != task["accepted_episodes_sha256"]:
            raise RuntimeError("input/config/accepted identity mismatch")
        surrogate_path = ROOT / registration["matched_surrogates"][task["cell"]]["path"]
        if sha(surrogate_path) != registration["matched_surrogates"][task["cell"]]["sha256"]:
            raise RuntimeError("matched surrogate identity mismatch")
        sys.path[:0] = [str(ROOT / task["track_source"]), str(I2A), str(FAST)]
        cfg = configure(task, task_root, public_path)
        from real_ecology_benchmark.backend import resolve_backend_for_workload, set_active_backend
        set_active_backend(resolve_backend_for_workload(cfg.compute,
                           f"corrected_recovery_{arm.lower()}:{task['method']}", cfg.environment))
        result = arm_o(task, task_root, cfg, surrogate_path) if arm == "O" else arm_t(
            task, task_root, cfg, public_path, surrogate_path, registration)
        if arm == "O" and result["parity"]["result"] != "PASS":
            raise RuntimeError("Arm O accepted parity failed")
        summary_path = Path(result["summary"].get("output_dir", task_root / "evaluation_result")) / "summary.json"
        receipt.update({"ended_unix": time.time(), "duration_seconds": time.time() - started,
            "exit_status": 0, "episodes_path": str(result["episodes"]),
            "episodes_sha256": sha(result["episodes"]), "summary_path": str(summary_path),
            "summary_sha256": sha(summary_path), "fitted_artifact": result["artifact"],
            "pre_return_component_receipt": str(result["component"]),
            "pre_return_component_sha256": sha(result["component"]),
            "timestep_evidence": str(result["evidence"]), "timestep_evidence_sha256": sha(result["evidence"]),
            "policy_evidence": str(result["policy_evidence"]), "policy_evidence_sha256": sha(result["policy_evidence"]),
            "accepted_arm_o_parity": result["parity"],
            "accepted_arm_o_parity_path": str(result["parity_path"]) if result["parity_path"] else None,
            "activity": result["activity"], "evidence_validation": result["validation"],
            "matched_surrogate_sha256": sha(surrogate_path),
            "fit_cache_identity": tree_identity(Path(task["fit_cache_dir"])) if task["fit_cache_dir"] else None,
            "bridge": result.get("bridge"), "point_mass_assignments": result.get("point_mass_assignments", 0),
            "retry_rebaseline_or_tuning": False})
        strict_json(receipt_path, receipt)
    except BaseException as exc:
        if not receipt_path.exists():
            receipt.update({"ended_unix": time.time(), "duration_seconds": time.time() - started,
                            "exit_status": int(getattr(exc, "code", 1) or 1),
                            "failure": str(exc), "traceback": traceback.format_exc()})
            strict_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("arm", choices=["O", "T"])
    parser.add_argument("index", type=int); args = parser.parse_args(); run(args.arm, args.index)
