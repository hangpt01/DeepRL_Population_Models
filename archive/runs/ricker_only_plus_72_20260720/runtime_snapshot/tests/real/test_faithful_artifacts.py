import json

import numpy as np

from real_ecology_benchmark.faithful_artifacts import (
    load_model,
    save_fit_artifacts,
    save_model,
)
from real_ecology_benchmark.faithful_fit import CANDIDATE_CONSTRUCTION, FitResult

from .faithful_fixtures import ricker_model


def test_model_artifact_round_trip_without_pickle(tmp_path):
    model = ricker_model()
    save_model(tmp_path / "candidate.npz", model)
    loaded = load_model(tmp_path / "candidate.npz")
    assert loaded.parameter_hash() == model.parameter_hash()
    with np.load(tmp_path / "candidate.npz", allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)


def test_plus_artifacts_record_bootstrap_map_candidate_construction(tmp_path):
    fit = FitResult(
        model=ricker_model(),
        objective=1.0,
        holdout_normalized_survey_sse=1.5,
        selected_gradient_norm=0.1,
        curvature_condition_estimate=2.0,
        start_gradient_norms=(0.1,),
        start_objective_traces=((2.0, 1.0),),
        finite_start_parameter_std_mean=0.0,
        selected_start=0,
        start_objectives=(1.0,),
        action_rows=(10, 10, 10),
        action_episodes=(2, 2, 2),
        sparse_actions=(),
        fit_episode_ids=(0, 1),
        holdout_episode_ids=(2,),
        public_data_hash="a" * 64,
        random_bank_hash="b" * 64,
        optimizer="torch_lbfgs",
        iterations=2,
    )
    result = save_fit_artifacts(
        tmp_path,
        [fit],
        [],
        "plus_adapted_mechanistic_fixed_pi_pbvi_v2",
        np.asarray([1.0]),
        "uniform",
        CANDIDATE_CONSTRUCTION,
    )
    root = tmp_path / "faithful_artifacts"
    bank = json.loads((root / "candidate_bank.json").read_text(encoding="utf-8"))
    fit_payload = json.loads((root / "faithful_fit.json").read_text(encoding="utf-8"))

    assert bank["candidate_construction"] == CANDIDATE_CONSTRUCTION
    assert fit_payload["candidate_construction"] == CANDIDATE_CONSTRUCTION
    assert result["candidate_construction"] == CANDIDATE_CONSTRUCTION
