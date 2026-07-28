import numpy as np

from real_ecology_benchmark.config import FaithfulPlannerConfig
from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner

from .faithful_fixtures import public_context, ricker_model


def tiny_config():
    return FaithfulPlannerConfig(
        state_bins=9, capacity_bins=3, observation_bins=9,
        transition_samples=8, observation_samples=8, belief_points=4,
        observation_branches=3, horizon=2,
    )


def test_transition_and_bayes_update_normalize():
    pomdp = CandidatePOMDP(ricker_model(), public_context(), tiny_config(), 11)
    transition = pomdp.transition_matrix(1.0, 0)
    assert np.allclose(transition.sum(axis=1), 1.0)
    belief = pomdp.initial_belief(60.0)
    updated, evidence = pomdp.update(belief, 0, 65.0)
    assert np.isclose(updated.probabilities.sum(), 1.0)
    assert np.isfinite(evidence)


def test_pbvi_is_deterministic_and_records_provenance():
    pomdp = CandidatePOMDP(ricker_model(), public_context(), tiny_config(), 11)
    planner = PointBasedPlanner(pomdp, tiny_config(), 0.95, 17)
    belief = pomdp.initial_belief(60.0)
    first = planner.action_values(belief)
    second = PointBasedPlanner(pomdp, tiny_config(), 0.95, 17).action_values(belief)
    assert np.array_equal(first, second)
    assert planner.provenance().invocation_count == 1
    assert planner.provenance().external_invocation is False
