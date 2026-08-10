"""Tiny real frozen-object fixtures used by the orchestration gate tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..common import canonical_json_bytes
from ..real_artifacts import _registered_track_imports, validate_frozen_fitted_object_roundtrip


@lru_cache(maxsize=16)
def frozen_object_evidence(
    method: str,
    cell: str,
    public_view_sha256: str,
) -> tuple[bytes, bytes]:
    repository = Path(__file__).resolve().parents[4]
    if method.startswith(("plus_", "moor_")):
        return _ecological_fixture(
            method,
            cell,
            public_view_sha256,
            repository,
        )
    return _general_fixture(
        method,
        cell,
        public_view_sha256,
        repository,
    )


def _general_fixture(method: str, cell: str, public_view_sha256: str, repository: Path):
    with _registered_track_imports("general", repository):
        from real_ecology_benchmark.behavior_model import CalibratedBehaviorModel
        from real_ecology_benchmark.config import MethodContext, ModelConfig, PlannerConfig
        from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy
        from real_ecology_benchmark.methods.ensemble_value_disagreement import (
            BootstrapQMember,
            EnsembleValueDisagreementPolicy,
        )
        from real_ecology_benchmark.methods.ogsrl import OGSRLPolicy, PublicKNNGuardian
        from real_ecology_benchmark.methods.refplan import RefPlanPolicy
        from real_ecology_benchmark.public_models import (
            PublicDynamicsEnsemble,
            PublicDynamicsMember,
            PublicParticlePlanner,
        )
        from real_ecology_benchmark.public_surrogate import (
            PublicFeatureSpec,
            PublicRewardRiskSurrogate,
        )
        from real_ecology_benchmark.realdata import PUBLIC_ACTION_CHANNELS
        from real_ecology_benchmark.types import BeliefState

        action_count = 11
        pop_id = "pop_tiger" if cell.startswith("amur_tiger") else "pop_fox"
        feature_spec = PublicFeatureSpec(
            observation_scale=100.0,
            continuous_mean=np.zeros(7, dtype=np.float64),
            continuous_std=np.ones(7, dtype=np.float64),
            pop_vocabulary=(pop_id,),
            num_actions=action_count,
            public_horizon=50,
        )
        surrogate = PublicRewardRiskSurrogate(
            feature_spec=feature_spec,
            reward_coefficients=np.zeros(20, dtype=np.float64),
            risk_coefficients=None,
            constant_risk=0.1,
            action_costs=np.zeros(action_count, dtype=np.float64),
            diagnostics={"synthetic_fixture": 1.0},
            public_data_hash=public_view_sha256,
            split_seed=17,
        )
        context = MethodContext(
            num_actions=action_count,
            action_costs=(0.0,) * action_count,
            action_channels=(next(iter(sorted(PUBLIC_ACTION_CHANNELS))),) * action_count,
            observation_noise_sigma=0.2,
            horizon=50,
            observation_scale=100.0,
            pop_id=pop_id,
            surrogate=surrogate,
        )
        model_cfg = ModelConfig(ensemble_size=5)
        planner_cfg = PlannerConfig(
            horizon=1,
            sequences=11,
            particles=2,
            bamcts_depth=1,
            bamcts_simulations=2,
            ogsrl_cost_horizon=1,
            ogsrl_deployment_rollouts=2,
        )
        members = [
            PublicDynamicsMember(
                coefficients=np.concatenate(
                    (
                        np.array([0.18 + index * 0.001, 0.4], dtype=np.float64),
                        np.zeros(34, dtype=np.float64),
                    )
                ),
                residual_sigma=0.02,
                num_actions=action_count,
                observation_scale=100.0,
            )
            for index in range(5)
        ]
        belief = BeliefState(
            states=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
            contexts=np.column_stack(
                (
                    np.full(4, 15.0, dtype=np.float64),
                    np.full(4, 20.0, dtype=np.float64),
                    np.arange(4, dtype=np.float64),
                )
            ),
            regimes=np.zeros(4, dtype=np.int8),
            log_weights=np.log(np.full(4, 0.25, dtype=np.float64)),
            observation=25.0,
        )
        feature_dim = belief.public_features(context.observation_scale).shape[0]
        prior = CalibratedBehaviorModel(
            weights=np.full((action_count, feature_dim + 1), 0.001, dtype=np.float64),
            mean=np.zeros(feature_dim + 1, dtype=np.float64),
            scale=np.ones(feature_dim + 1, dtype=np.float64),
        )

        if method == "refplan":
            fitted = RefPlanPolicy(context, model_cfg, planner_cfg, seed=31)
            fitted.dynamics = PublicDynamicsEnsemble(members, seed=23)
            fitted.posterior = np.full(5, 0.2, dtype=np.float64)
            fitted.policy_prior = prior
            fitted.prior_weights = prior.weights
            fitted.prior_scaler = (prior.mean, prior.scale)
            fitted.planner = PublicParticlePlanner(context, planner_cfg, seed=31)
            component = "refplan_fitted_policy"
        elif method == "ogsrl":
            fitted = OGSRLPolicy(context, model_cfg, planner_cfg, seed=41)
            fitted.dynamics = PublicDynamicsEnsemble(members, seed=23)
            guardian_features = 2 + action_count
            fitted.guardian = PublicKNNGuardian(
                anchors=np.zeros((2, guardian_features), dtype=np.float64),
                mean=np.zeros(guardian_features, dtype=np.float64),
                scale=np.ones(guardian_features, dtype=np.float64),
                threshold=1e6,
                num_actions=action_count,
                observation_scale=100.0,
            )
            fitted.surrogate = surrogate
            fitted.s_low = 50.0
            fitted.actor_weights = np.zeros((action_count, 3), dtype=np.float64)
            fitted.lambda_safety = 1.0
            fitted.lambda_ood = 1.0
            component = "ogsrl_fitted_policy"
        elif method == "bamcts":
            fitted = BAMCTSPolicy(
                context,
                model_cfg,
                planner_cfg,
                simulations=2,
                depth=1,
                seed=37,
            )
            fitted.dynamics = PublicDynamicsEnsemble(members, seed=23)
            fitted.posterior = np.full(5, 0.2, dtype=np.float64)
            component = "bamcts_fitted_policy"
        elif method == "ensemble_value_disagreement_pessimism":
            fitted = EnsembleValueDisagreementPolicy(context, model_cfg, planner_cfg, seed=9)
            fitted.behavior_model = prior
            fitted.q_members = [
                BootstrapQMember(
                    q_weights=np.full(
                        (action_count, feature_dim + 1),
                        index / 100.0,
                        dtype=np.float64,
                    ),
                    bootstrap_episode_ids=np.arange(4, dtype=np.int32),
                    seed=100 + index,
                    bellman_mse=0.01,
                )
                for index in range(20)
            ]
            fitted.feature_dim = feature_dim
            fitted.q_weights = np.mean(
                np.stack([member.q_weights for member in fitted.q_members]), axis=0
            )
            component = "evd_fitted_policy"
        else:
            raise AssertionError(f"unsupported general method: {method}")
        payload, receipt = validate_frozen_fitted_object_roundtrip(
            component,
            fitted,
            operation="act",
            args=(belief, 25.0),
            repository_root=repository,
        )
        return payload, canonical_json_bytes(receipt)


def _ecological_fixture(method: str, cell: str, public_view_sha256: str, repository: Path):
    with _registered_track_imports("ecological", repository):
        from real_ecology_benchmark.config import (
            FaithfulConfig,
            FaithfulFitConfig,
            FaithfulModelConfig,
            FaithfulPlannerConfig,
            MethodContext,
            ModelConfig,
            PlannerConfig,
        )
        from real_ecology_benchmark.faithful_ecology import MechanisticModel
        from real_ecology_benchmark.faithful_fit import (
            RICKER_ONLY_CANDIDATE_CONSTRUCTION,
            CandidateBank,
            FitResult,
        )
        from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP
        from real_ecology_benchmark.methods.moor_faithful import MOORFaithfulRickerPBVIPolicy
        from real_ecology_benchmark.methods.plus_faithful import (
            PLUSRickerOnlyFaithfulPBVIPolicy,
        )
        from real_ecology_benchmark.planners.pbvi import PointBasedPlanner
        from real_ecology_benchmark.types import BeliefState

        action_count = 11
        pop_id = "pop_tiger" if cell.startswith("amur_tiger") else "pop_fox"
        channels = ("none",) * action_count
        context = MethodContext(
            num_actions=action_count,
            action_costs=(0.0,) * action_count,
            action_channels=channels,
            observation_noise_sigma=0.2,
            horizon=50,
            observation_scale=100.0,
            pop_id=pop_id,
        )
        faithful = FaithfulConfig(
            model=FaithfulModelConfig(forms=("ricker",), candidates_per_form=8, prior="uniform"),
            fit=FaithfulFitConfig(starts=1, iterations=1, mc_paths=1, regime_mc_paths=8),
            planner=FaithfulPlannerConfig(
                state_bins=5,
                capacity_bins=2,
                observation_bins=5,
                transition_samples=1,
                observation_samples=1,
                belief_points=1,
                observation_branches=2,
                horizon=1,
            ),
        )

        def fit_result(index: int) -> FitResult:
            growth = np.zeros(action_count, dtype=np.float64)
            growth[0] = 0.20 + index * 0.001
            model = MechanisticModel(
                form="ricker",
                growth=growth,
                mortality=np.zeros(action_count, dtype=np.float64),
                capacity_increment=np.zeros(action_count, dtype=np.float64),
                stocking=np.zeros(action_count, dtype=np.float64),
                reset_log_mean=np.log(0.65),
                reset_log_scale=0.08,
                initial_capacity=1.0,
                capacity_ceiling=1.8,
                process_scale=0.03,
                observation_scale=0.05,
                survey_scale=100.0,
                depensation_thresholds=np.array([0.2, 0.35], dtype=np.float64),
                theta_exponent=2.0,
                regime_multipliers=np.array([0.7, 1.2], dtype=np.float64),
                regime_matrix=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
                action_channels=channels,
                candidate_id=f"synthetic_ricker_{index:02d}",
            )
            return FitResult(
                model=model,
                objective=float(index),
                holdout_normalized_survey_sse=1.0,
                selected_gradient_norm=0.1,
                curvature_condition_estimate=2.0,
                start_gradient_norms=(0.1,),
                start_objective_traces=((1.0,),),
                finite_start_parameter_std_mean=0.0,
                selected_start=0,
                start_objectives=(1.0,),
                action_rows=(1,) * action_count,
                action_episodes=(1,) * action_count,
                sparse_actions=(),
                fit_episode_ids=(0,),
                holdout_episode_ids=(1,),
                public_data_hash=public_view_sha256,
                random_bank_hash="b" * 64,
                optimizer="torch_lbfgs",
                iterations=1,
                fit_cache_key=f"synthetic-cache-{index:02d}",
            )

        fits = tuple(fit_result(index) for index in range(8))
        pomdps = [
            CandidatePOMDP(item.model, context, faithful.planner, 100 + index)
            for index, item in enumerate(fits)
        ]
        planners = [
            PointBasedPlanner(pomdp, faithful.planner, 0.95, 200 + index)
            for index, pomdp in enumerate(pomdps)
        ]
        if method == "plus_adapted_ricker_only_pbvi":
            fitted = PLUSRickerOnlyFaithfulPBVIPolicy(
                context,
                ModelConfig(),
                PlannerConfig(horizon=1),
                seed=43,
                faithful_cfg=faithful,
            )
            fitted.candidate_bank = CandidateBank(
                fits,
                np.full(8, 1.0 / 8.0, dtype=np.float64),
                "uniform",
                RICKER_ONLY_CANDIDATE_CONSTRUCTION,
            )
            fitted.fit_cache_statuses = ["synthetic"] * 8
            fitted.pomdps = pomdps
            fitted.planners = planners
            fitted.posterior = fitted.candidate_bank.initial_weights.copy()
            component = "plus_fitted_policy"
        elif method == "moor_adapted_ricker_misspec_pbvi":
            fitted = MOORFaithfulRickerPBVIPolicy(
                context,
                ModelConfig(),
                PlannerConfig(horizon=1),
                seed=47,
                faithful_cfg=faithful,
            )
            fitted.fit_result = fits[0]
            fitted.fit_cache_status = "synthetic"
            fitted.pomdp = CandidatePOMDP(fits[0].model, context, faithful.planner, 300)
            fitted.planner = PointBasedPlanner(fitted.pomdp, faithful.planner, 0.95, 301)
            component = "moor_fitted_policy"
        else:
            raise AssertionError(f"unsupported ecological method: {method}")
        belief = BeliefState(
            states=np.array([10.0, 20.0], dtype=np.float64),
            contexts=np.ones((2, 3), dtype=np.float64),
            regimes=np.zeros(2, dtype=np.int8),
            log_weights=np.log(np.full(2, 0.5, dtype=np.float64)),
            observation=25.0,
            diagnostics={"public_observation_filter": 1.0},
        )
        payload, receipt = validate_frozen_fitted_object_roundtrip(
            component,
            fitted,
            operation="act",
            args=(belief, 25.0),
            repository_root=repository,
        )
        return payload, canonical_json_bytes(receipt)
