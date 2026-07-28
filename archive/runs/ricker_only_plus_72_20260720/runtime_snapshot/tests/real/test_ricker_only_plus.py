from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from real_ecology_benchmark.config import (
    FaithfulConfig,
    FaithfulFitConfig,
    FaithfulModelConfig,
    FaithfulPlannerConfig,
    ModelConfig,
    PlannerConfig,
)
from real_ecology_benchmark.faithful_ecology import MechanisticModel
from real_ecology_benchmark.faithful_fit import FitResult, build_candidate_bank
from real_ecology_benchmark.faithful_fit import ordered_trajectory_sse
from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP
from real_ecology_benchmark.methods import METHODS
from real_ecology_benchmark.methods.plus_faithful import (
    PLUSRickerOnlyFaithfulPBVIPolicy,
)
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner

from .faithful_fixtures import public_context, public_dataset, ricker_model


def _config() -> FaithfulConfig:
    return FaithfulConfig(
        model=FaithfulModelConfig(forms=("ricker",), candidates_per_form=8, prior="uniform"),
        fit=FaithfulFitConfig(starts=1, iterations=2, mc_paths=1),
        planner=FaithfulPlannerConfig(
            state_bins=9, capacity_bins=3, observation_bins=9,
            transition_samples=4, observation_samples=4, belief_points=2,
            observation_branches=2, horizon=1,
        ),
    )


def _result(candidate_id: str, fit_ids: tuple[int, ...], holdout: tuple[int, ...]) -> FitResult:
    index = int(candidate_id.rsplit("_", 1)[1])
    base = ricker_model()
    growth = base.growth.copy()
    growth[0] += index * 0.001
    model = replace(base, growth=growth, candidate_id=candidate_id)
    return FitResult(
        model=model, objective=float(index), holdout_normalized_survey_sse=1.0,
        selected_gradient_norm=0.1, curvature_condition_estimate=2.0,
        start_gradient_norms=(0.1,), start_objective_traces=((1.0,),),
        finite_start_parameter_std_mean=0.0, selected_start=0,
        start_objectives=(1.0,), action_rows=(10, 10, 10),
        action_episodes=(2, 2, 2), sparse_actions=(), fit_episode_ids=fit_ids,
        holdout_episode_ids=holdout, public_data_hash="a" * 64,
        random_bank_hash="b" * 64, optimizer="torch_lbfgs", iterations=2,
        fit_cache_key=f"key-{candidate_id}",
    )


def test_ricker_only_method_is_distinct_and_strictly_routed():
    assert METHODS["plus_adapted_ricker_only_pbvi"] is PLUSRickerOnlyFaithfulPBVIPolicy
    policy = PLUSRickerOnlyFaithfulPBVIPolicy(
        public_context(), ModelConfig(), PlannerConfig(horizon=1), seed=1,
        faithful_cfg=_config(),
    )
    assert policy.name == "plus_adapted_ricker_only_pbvi"
    with pytest.raises(ValueError, match="forms"):
        PLUSRickerOnlyFaithfulPBVIPolicy(
            public_context(), ModelConfig(), PlannerConfig(horizon=1), seed=1,
            faithful_cfg=replace(
                _config(), model=FaithfulModelConfig(candidates_per_form=2)
            ),
        )


def test_ricker_bank_is_one_full_history_plus_seven_deterministic_bootstraps(monkeypatch):
    calls = []

    def fake_fit(_dataset, _context, form, _model_cfg, _fit_cfg, seed,
                 candidate_id, fit_episode_ids, holdout_episode_ids,
                 regime_persistence, cache_dir):
        del regime_persistence, cache_dir
        assert form == "ricker"
        calls.append((seed, candidate_id, tuple(fit_episode_ids), tuple(holdout_episode_ids)))
        return _result(candidate_id, tuple(fit_episode_ids), tuple(holdout_episode_ids)), "hit"

    monkeypatch.setattr(
        "real_ecology_benchmark.faithful_fit.load_or_fit_mechanistic_model", fake_fit
    )
    dataset = public_dataset(episodes=8, length=3)
    bank, statuses = build_candidate_bank(
        dataset, public_context(), _config().model, _config().fit, 51116
    )
    first_calls = list(calls)
    calls.clear()
    bank_again, _ = build_candidate_bank(
        dataset, public_context(), _config().model, _config().fit, 51116
    )
    assert calls == first_calls
    assert len(bank.fits) == len(bank_again.fits) == len(statuses) == 8
    assert calls[0][2] == tuple(sorted(set(calls[0][2])))
    assert all(len(call[2]) == len(calls[0][2]) for call in calls[1:])
    assert np.array_equal(bank.initial_weights, np.full(8, 1.0 / 8.0))
    assert bank.construction_type == "ricker_only_episode_bootstrap_map_1full_7bootstrap_v1"


def _inactive_variant(base: MechanisticModel) -> MechanisticModel:
    return replace(
        base, depensation_thresholds=np.asarray([0.15, 0.30]),
        theta_exponent=3.5, regime_multipliers=np.asarray([0.5, 1.5]),
        regime_matrix=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )


def test_ricker_candidate_kernels_and_pbvi_ignore_other_family_parameters():
    base, changed = ricker_model(), _inactive_variant(ricker_model())
    cfg = FaithfulPlannerConfig(name="pbvi", state_bins=17, horizon=3)
    left = CandidatePOMDP(base, public_context(), cfg, seed=3)
    right = CandidatePOMDP(changed, public_context(), cfg, seed=3)
    dataset = public_dataset()
    assert ordered_trajectory_sse(base, dataset) == ordered_trajectory_sse(changed, dataset)
    for action in range(3):
        assert np.array_equal(
            left.transition_matrix(1.0, action), right.transition_matrix(1.0, action)
        )
    belief = left.initial_belief(65.0)
    qa = PointBasedPlanner(left, cfg, 0.95, seed=1).action_values(belief)
    qb = PointBasedPlanner(right, cfg, 0.95, seed=1).action_values(belief)
    assert np.array_equal(qa, qb)
    updated_left, evidence_left = left.update(belief, 1, 60.0)
    updated_right, evidence_right = right.update(belief, 1, 60.0)
    assert evidence_left == evidence_right
    assert np.array_equal(updated_left.probabilities, updated_right.probabilities)


def test_ricker_only_policy_has_no_hidden_family_input():
    context = public_context()
    assert not hasattr(context, "environment") and not hasattr(context, "family")
    source = inspect.getsource(PLUSRickerOnlyFaithfulPBVIPolicy)
    assert "environment" not in source and "hidden family" not in source.lower()


def test_acceptance_rejects_return_sentinels_and_summary(tmp_path):
    from scripts.run_ricker_only_plus_acceptance import guarded_json

    sentinel = tmp_path / "artifact.json"
    sentinel.write_text(json.dumps({"operational_return": "SENTINEL"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden return field"):
        guarded_json(sentinel)
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot open"):
        guarded_json(summary)


def test_frozen_manifests_and_all_dry_run_routes_validate():
    from pathlib import Path
    from scripts.validate_ricker_only_plus_launch import cell_workers, commands, validate

    root = Path(__file__).resolve().parents[2]
    manifest_root = root / "experiments/ricker_only_plus_72/manifests"
    fit = manifest_root / "ricker_only_plus_fit_72.csv"
    plan = manifest_root / "ricker_only_plus_plan_72.csv"
    result = validate(fit, plan)
    resolved = commands(fit, plan, root, Path("/sealed/run"), Path("/python"))
    assert result["fit_rows"] == result["plan_rows"] == 72
    assert len(resolved) == 144
    assert len({tuple(item["command"]) for item in resolved}) == 144
    assert all(item["threads"] == 1 for item in resolved)
    assert not any(item["return_fields_opened"] for item in resolved)
    workers = cell_workers(resolved)
    assert len(workers) == 72
    assert all(worker["allocated_tasks"] == 1 for worker in workers)
    assert all(
        "run_ricker_only_fit_locked.py" in item["command"][2]
        for item in resolved if item["stage"] == "fit"
    )
    assert all(
        "run_ricker_only_plan_with_receipt.py" in item["command"][2]
        for item in resolved if item["stage"] == "plan"
    )
    canary = commands(
        manifest_root / "ricker_only_plus_canary_fit_1.csv",
        manifest_root / "ricker_only_plus_canary_plan_1.csv",
        root, Path("/sealed/canary"), Path("/python"),
        root / "configs/paper_faithful_hidden_ricker_only_plus_canary_v1.yaml",
    )
    assert len(canary) == 2
    assert all("canary_v1.yaml" in item["command"][6] for item in canary)
    assert len(cell_workers(canary)) == 1


def test_cache_lock_identity_covers_every_registered_identity_component():
    from scripts.run_ricker_only_fit_locked import full_lock_identity

    values = {
        "runtime_digest": "a" * 64,
        "config_sha256": "b" * 64,
        "public_data_hash": "c" * 64,
        "transition_data_hash": "d" * 64,
        "candidate_index": 3,
        "candidate_seed": 81116,
        "cache_key": "e" * 64,
    }
    baseline = full_lock_identity(**values)
    replacements = {
        "runtime_digest": "f" * 64,
        "config_sha256": "1" * 64,
        "public_data_hash": "2" * 64,
        "transition_data_hash": "3" * 64,
        "candidate_index": 4,
        "candidate_seed": 91116,
        "cache_key": "4" * 64,
    }
    for field, replacement in replacements.items():
        changed = {**values, field: replacement}
        assert full_lock_identity(**changed) != baseline


def test_cell_worker_failure_suppresses_only_its_plan(tmp_path):
    root = Path(__file__).resolve().parents[2]
    fake = tmp_path / "fake-python"
    trace = tmp_path / "trace"
    fake.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$TRACE\"\n"
        "case \"$*\" in *run_ricker_only_fit_locked.py*) exit 7;; esac\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {
        **os.environ,
        "CODE_ROOT": str(root),
        "FIT_MANIFEST": "/fit.csv",
        "PLAN_MANIFEST": "/plan.csv",
        "OUTPUT_ROOT": "/sealed/output",
        "PYTHON_BIN": str(fake),
        "SLURM_ARRAY_TASK_ID": "11",
        "TRACE": str(trace),
    }
    result = subprocess.run(
        [str(root / "scripts/slurm/run_ricker_only_plus_cell.sh")],
        env=env, text=True, capture_output=True, check=False,
    )
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 7
    assert len(lines) == 1 and "run_ricker_only_fit_locked.py" in lines[0]
    assert "run_real_manifest_row.py" not in lines[0]


def test_plan_completion_requires_exact_hashes_and_rejects_temporary_files(tmp_path):
    from scripts.run_ricker_only_plan_with_receipt import verified_artifact_hashes

    def write(name, content=b"arrays"):
        path = tmp_path / name
        path.write_bytes(content)
        return path

    write("candidate_bank.npz")
    write("candidate_bank.json", json.dumps({
        "array_hash": __import__("hashlib").sha256(b"arrays").hexdigest()
    }).encode())
    for name in (
        "faithful_fit.npz", "planner_provenance.json",
        "pbvi_policy_diagnostics.npz",
    ):
        write(name, b"{}" if name.endswith(".json") else b"arrays")
    write("faithful_fit.json", b"{}")
    write("privacy_audit.json", json.dumps({
        "status": "passed", "forbidden_name_hits": []
    }).encode())
    array_hash = __import__("hashlib").sha256(b"arrays").hexdigest()
    for prefix in ("candidate", "pomdp_model"):
        for index in range(8):
            write(f"{prefix}_{index:03d}.npz")
            write(
                f"{prefix}_{index:03d}.json",
                json.dumps({"array_hash": array_hash}).encode(),
            )
    hashes = verified_artifact_hashes(tmp_path)
    assert len(hashes) == 39
    write(".partial.123.tmp")
    with pytest.raises(RuntimeError, match="temporary artifacts remain"):
        verified_artifact_hashes(tmp_path)
