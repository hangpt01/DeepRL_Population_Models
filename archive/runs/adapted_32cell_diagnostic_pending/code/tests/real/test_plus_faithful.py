from dataclasses import replace

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
from real_ecology_benchmark.faithful_fit import CandidateBank, FitResult
from real_ecology_benchmark.methods.plus_faithful import PLUSFaithfulPBVIPolicy

from .faithful_fixtures import public_context, public_dataset, ricker_model


def _fit(model):
    return FitResult(
        model=model,
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


def test_uniform_prior_is_exact_and_duplicate_models_fail():
    first = ricker_model()
    growth = first.growth.copy()
    growth[0] += 0.01
    second = type(first)(**{**first.__dict__, "growth": growth, "candidate_id": "second"})
    bank = CandidateBank((_fit(first), _fit(second)), np.asarray([0.5, 0.5]), "uniform")
    assert np.array_equal(bank.initial_weights, np.asarray([0.5, 0.5]))
    try:
        CandidateBank((_fit(first), _fit(first)), np.asarray([0.5, 0.5]), "uniform")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate candidate bank did not fail")


def test_adapted_plus_policy_fits_one_candidate_per_mechanistic_form(tmp_path):
    pytest.importorskip("torch")
    faithful = FaithfulConfig(
        model=FaithfulModelConfig(candidates_per_form=1),
        fit=FaithfulFitConfig(starts=1, iterations=2, mc_paths=1, regime_mc_paths=8),
        planner=FaithfulPlannerConfig(
            state_bins=9,
            capacity_bins=3,
            observation_bins=9,
            transition_samples=4,
            observation_samples=4,
            belief_points=2,
            observation_branches=2,
            horizon=1,
        ),
        fit_cache_dir=str(tmp_path / "fit_cache"),
    )
    policy = PLUSFaithfulPBVIPolicy(
        public_context(),
        ModelConfig(),
        PlannerConfig(horizon=1),
        seed=19,
        faithful_cfg=faithful,
    )
    diagnostics = policy.fit(public_dataset(episodes=4, length=5))
    assert policy.name == "plus_adapted_mechanistic_pbvi"
    assert diagnostics["candidate_count"] == 4.0
    assert {fit.model.form for fit in policy.candidate_bank.fits} == {
        "ricker",
        "allee",
        "theta",
        "regime",
    }
    assert all(fit.fit_cache_key for fit in policy.candidate_bank.fits)
    yield_policy = PLUSFaithfulPBVIPolicy(
        replace(public_context(), reward_mode="yield"),
        ModelConfig(),
        PlannerConfig(horizon=1),
        seed=19,
        faithful_cfg=faithful,
    )
    yield_diagnostics = yield_policy.fit(public_dataset(episodes=4, length=5))
    assert yield_diagnostics["fit_cache_hits"] == 4.0
    assert all(left is not right for left, right in zip(policy.planners, yield_policy.planners))
