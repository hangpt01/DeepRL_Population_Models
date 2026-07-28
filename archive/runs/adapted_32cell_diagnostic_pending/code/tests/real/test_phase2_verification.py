"""Phase-2 verification tests for the corrected paper-aligned PLUS/MOOR baselines.

Covers, per the completion protocol:
  - r_a asymmetric transform, endpoints, q=0 centre, gradients, max-split, structural zeros;
  - PBVI vs an exact finite-horizon POMDP, with a QMDP-divergent information-gathering case,
    exercising the same ``PointBasedPlanner`` backup used by corrected PLUS and MOOR;
  - MOOR (single Ricker) cannot be affected by the inactive Allee/theta/regime parameters;
  - the regime switch-count convention (24 switch opportunities over a 25-transition episode).
"""

from __future__ import annotations

import types
import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch only in the paper-faithful venv
    _HAS_TORCH = False

from real_ecology_benchmark.config import (
    FaithfulModelConfig,
    FaithfulPlannerConfig,
)
from real_ecology_benchmark.faithful_ecology import MechanisticModel, fixed_regime_matrix
from real_ecology_benchmark.faithful_pomdp import CandidateBelief, CandidatePOMDP
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner

from .faithful_fixtures import public_context, ricker_model


# --------------------------------------------------------------------------------------
# Task 1 - r_a asymmetric transform, max-split, gradients, structural zeros
# --------------------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "paper-faithful fitter requires torch")
class SignedRateTransformTests(unittest.TestCase):
    """The approved asymmetric bounded optimizer parameterization (Option A)."""

    def _decode(self, raw_list, form="ricker"):
        from real_ecology_benchmark.faithful_fit import _decode

        context = public_context(3)  # channels: none, rate+capacity, state
        model_cfg = FaithfulModelConfig()
        raw = torch.tensor(raw_list, dtype=torch.float64)
        return _decode(raw, context, form, model_cfg, 0.9, torch)

    def test_transform_centre_and_asymptotic_bounds(self):
        # public_context(3): rate_actions=[1], so rate_count = 1 baseline + 1 = 2.
        # Decode order: 5 scalars, 2 signed rates, 1 capacity, 1 stocking = 9 raw entries.
        # q = 0 -> r = 0.25 + 1.75*tanh(0) = 0.25 (raw-coordinate centre, NOT a prior).
        decoded = self._decode([0.0] * 9)
        self.assertTrue(torch.allclose(decoded["signed_rates"], torch.tensor(0.25, dtype=torch.float64)))
        # Asymptotic upper bound: q -> +inf gives r -> 0.25 + 1.75 = 2.0.
        upper = self._decode([0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0, 0.0, 0.0])
        self.assertTrue(torch.all(upper["signed_rates"] <= 2.0 + 1e-9))
        self.assertGreater(float(upper["signed_rates"].max()), 2.0 - 1e-6)
        # Asymptotic lower bound: q -> -inf gives r -> 0.25 - 1.75 = -1.5.
        lower = self._decode([0.0, 0.0, 0.0, 0.0, 0.0, -40.0, -40.0, 0.0, 0.0])
        self.assertTrue(torch.all(lower["signed_rates"] >= -1.5 - 1e-9))
        self.assertLess(float(lower["signed_rates"].min()), -1.5 + 1e-6)

    def test_transform_matches_documented_formula(self):
        for q in (-2.0, -0.5, 0.0, 0.3, 1.7):
            decoded = self._decode([0.0, 0.0, 0.0, 0.0, 0.0, q, q, 0.0, 0.0])
            expected = 0.25 + 1.75 * np.tanh(q)
            self.assertAlmostEqual(float(decoded["signed_rates"][0]), expected, places=10)

    def test_max_split_negative_zero_positive(self):
        from real_ecology_benchmark.faithful_fit import _symmetric_positive

        for r in (-0.8, 0.0, 1.4):
            value = torch.tensor([r], dtype=torch.float64)
            g = _symmetric_positive(value, torch)
            h = _symmetric_positive(-value, torch)
            self.assertAlmostEqual(float(g), max(r, 0.0), places=12)
            self.assertAlmostEqual(float(h), max(-r, 0.0), places=12)
            # Exact structural exclusivity: never both positive.
            self.assertAlmostEqual(float(g) * float(h), 0.0, places=12)

    def test_clarke_subgradient_signs(self):
        from real_ecology_benchmark.faithful_fit import _symmetric_positive

        for r, expected_slope in ((-0.7, 0.0), (0.0, 0.5), (0.9, 1.0)):
            value = torch.tensor([r], dtype=torch.float64, requires_grad=True)
            _symmetric_positive(value, torch).sum().backward()
            self.assertAlmostEqual(float(value.grad), expected_slope, places=12)

    def test_structural_zeros_from_public_channels(self):
        from real_ecology_benchmark.faithful_fit import _action_layout, action_parameter_count

        channels = (
            "none", "rate", "rate", "rate", "rate",
            "capacity", "capacity", "rate+capacity", "rate+capacity", "rate+capacity",
            "state",
        )
        rate_actions, capacity_actions = _action_layout(channels)
        # rate params only where the public channel includes a rate mechanism.
        self.assertEqual(rate_actions, (1, 2, 3, 4, 7, 8, 9))
        # capacity params only where the channel includes a capacity mechanism.
        self.assertEqual(capacity_actions, (5, 6, 7, 8, 9))
        # 1 shared baseline + 7 rate + 5 capacity + 1 stocking = 14 free action-effect params.
        self.assertEqual(action_parameter_count(channels), 14)


# --------------------------------------------------------------------------------------
# Task 2 - PBVI vs exact finite-horizon POMDP, with a QMDP-divergent case
# --------------------------------------------------------------------------------------


class _ProbeRevealPOMDP:
    """A tiny 4-state POMDP where active information gathering strictly helps.

    States encode (type in {0,1}) x (mode in {normal, revealed}):
        0 = (T0, normal)  1 = (T1, normal)  2 = (T0, revealed)  3 = (T1, revealed)
    Observations are action-independent p(o|s') (as in the real CandidatePOMDP):
    revealed states emit their type sharply; normal states are uninformative.
    Actions: 0 = probe (normal->revealed), 1 = commit-0, 2 = commit-1.
    A blind commit at an uncertain belief scores ~0; probing first reveals the type and
    enables a correct commit. QMDP, assuming full observability after one step, refuses to
    pay the probe cost and commits blindly - so its action differs from the exact optimum.
    This class implements exactly the CandidatePOMDP interface the planner calls.
    """

    OBS_VALUES = np.array([0.0, 1.0])
    _OBS = np.array(
        [[0.5, 0.5], [0.5, 0.5], [0.95, 0.05], [0.05, 0.95]], dtype=np.float64
    )
    _T = np.array(
        [
            # probe: normal<->revealed, type preserved
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            # commit-0: back to normal, type preserved
            [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            # commit-1: identical transition to commit-0
            [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
        ],
        dtype=np.float64,
    )
    _COST = 0.05
    # reward on the *next* state (matches expected_public_reward semantics)
    _R = np.array(
        [
            [-_COST, -_COST, -_COST, -_COST],  # probe
            [1.0, -1.0, 1.0, -1.0],            # commit-0: +1 iff type 0
            [-1.0, 1.0, -1.0, 1.0],            # commit-1: +1 iff type 1
        ],
        dtype=np.float64,
    )

    def __init__(self):
        self.context = types.SimpleNamespace(num_actions=3)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        out = belief.probabilities @ self._T[action]
        return out / out.sum()

    def representative_observations(self, predicted, branch_count):
        weights = predicted @ self._OBS
        return self.OBS_VALUES.copy(), weights

    def update(self, belief: CandidateBelief, action: int, observation: float):
        predicted = self.predict(belief, action)
        obs_index = int(np.argmin(np.abs(self.OBS_VALUES - observation)))
        likelihood = self._OBS[:, obs_index]
        joint = predicted * likelihood
        evidence = float(joint.sum())
        posterior = joint / evidence if evidence > 0 else predicted
        return (
            CandidateBelief(posterior, 1.0, observation, observation, belief.timestep + 1),
            float(np.log(max(evidence, 1e-300))),
        )

    def expected_public_reward(self, belief, action, predicted=None):
        predicted = self.predict(belief, action) if predicted is None else predicted
        return float(predicted @ self._R[action])

    def model_hash(self) -> str:
        return "probe_reveal_pomdp"

    # --- reference solvers (pure, using the same T/O/R the planner sees) ---
    def exact_value(self, belief, depth, horizon, gamma):
        if depth >= horizon:
            return 0.0
        return max(
            self.exact_q(belief, a, depth, horizon, gamma)
            for a in range(self.context.num_actions)
        )

    def exact_q(self, belief, action, depth, horizon, gamma):
        predicted = self.predict(belief, action)
        immediate = self.expected_public_reward(belief, action, predicted)
        obs, weights = self.representative_observations(predicted, 2)
        cont = 0.0
        for o, w in zip(obs, weights):
            if w <= 0.0:
                continue
            child, _ = self.update(belief, action, float(o))
            cont += w * self.exact_value(child, depth + 1, horizon, gamma)
        return immediate + gamma * cont

    def qmdp_action(self, belief, horizon, gamma):
        # Fully observable value V_MDP(s, depth) assuming the state is known.
        v = np.zeros((horizon + 1, 4))
        for depth in range(horizon - 1, -1, -1):
            for s in range(4):
                v[depth, s] = max(
                    self._R[a, :] @ self._T[a][s] + gamma * (self._T[a][s] @ v[depth + 1])
                    for a in range(3)
                )
        b = belief.probabilities
        q = [
            sum(
                b[s] * (self._R[a, :] @ self._T[a][s] + gamma * (self._T[a][s] @ v[1]))
                for s in range(4)
            )
            for a in range(3)
        ]
        return int(np.argmax(q)), np.asarray(q)


class PBVIExactPOMDPTests(unittest.TestCase):
    HORIZON = 3
    GAMMA = 0.95
    VALUE_TOL = 1e-6  # PBVI's reachable graph covers this tiny belief set exactly

    def _planner(self, model):
        config = FaithfulPlannerConfig(
            name="pbvi", horizon=self.HORIZON, belief_points=64, observation_branches=2
        )
        return PointBasedPlanner(model, config, self.GAMMA, seed=7)

    def test_pbvi_matches_exact_and_diverges_from_qmdp_when_information_helps(self):
        model = _ProbeRevealPOMDP()
        uncertain = CandidateBelief(np.array([0.5, 0.5, 0.0, 0.0]), 1.0, 0.5, 0.5, 0)

        exact_q = np.array(
            [model.exact_q(uncertain, a, 0, self.HORIZON, self.GAMMA) for a in range(3)]
        )
        exact_action = int(np.argmax(exact_q))
        qmdp_action, _qmdp_q = model.qmdp_action(uncertain, self.HORIZON, self.GAMMA)
        pbvi_q = self._planner(model).action_values(uncertain)
        pbvi_action = int(np.argmax(pbvi_q))

        # The exact optimum gathers information (probe) at an uncertain belief.
        self.assertEqual(exact_action, 0, "exact policy should probe first")
        # QMDP commits blindly: it must choose a different action than the exact optimum.
        self.assertNotEqual(qmdp_action, exact_action, "QMDP should diverge here")
        self.assertIn(qmdp_action, (1, 2))
        # The real PBVI backup recovers the exact action - it is NOT QMDP.
        self.assertEqual(pbvi_action, exact_action, "PBVI must match exact, not QMDP")
        # The optimal action's value (the one PBVI acts on) matches exact tightly.
        self.assertAlmostEqual(
            pbvi_q[exact_action], exact_q[exact_action], places=5,
            msg=f"PBVI optimal value {pbvi_q[exact_action]} vs exact {exact_q[exact_action]}",
        )
        # In this registered test PBVI selected the exact optimal information-gathering action and
        # matched that action's exact value within 1e-5; suboptimal-action values remain approximate,
        # as expected for point-based planning (no general value exactness is claimed).
        self.assertGreater(pbvi_q[0] - max(pbvi_q[1], pbvi_q[2]), 0.5)
        self.assertGreater(exact_q[0] - max(exact_q[1], exact_q[2]), 0.5)

    def test_pbvi_commits_correctly_when_type_is_known(self):
        model = _ProbeRevealPOMDP()
        known_type0 = CandidateBelief(np.array([0.98, 0.02, 0.0, 0.0]), 1.0, 0.5, 0.5, 0)
        exact_action = int(
            np.argmax(
                [model.exact_q(known_type0, a, 0, self.HORIZON, self.GAMMA) for a in range(3)]
            )
        )
        pbvi_action = int(np.argmax(self._planner(model).action_values(known_type0)))
        self.assertEqual(exact_action, 1, "known type-0 should commit-0")
        self.assertEqual(pbvi_action, exact_action)


# --------------------------------------------------------------------------------------
# Task 4 - MOOR (single Ricker) cannot be affected by inactive Allee/theta/regime params
# --------------------------------------------------------------------------------------


class MOORRickerMaskTests(unittest.TestCase):
    def _ricker(self, thresholds, theta, multipliers, matrix):
        base = ricker_model(3)
        return MechanisticModel(
            form="ricker",
            growth=base.growth,
            mortality=base.mortality,
            capacity_increment=base.capacity_increment,
            stocking=base.stocking,
            reset_log_mean=base.reset_log_mean,
            reset_log_scale=base.reset_log_scale,
            initial_capacity=base.initial_capacity,
            capacity_ceiling=base.capacity_ceiling,
            process_scale=base.process_scale,
            observation_scale=base.observation_scale,
            survey_scale=base.survey_scale,
            depensation_thresholds=np.asarray(thresholds),
            theta_exponent=float(theta),
            regime_multipliers=np.asarray(multipliers),
            regime_matrix=np.asarray(matrix),
            action_channels=base.action_channels,
            candidate_id="mask",
        )

    def test_inactive_form_parameters_cannot_change_ricker_outputs(self):
        default = self._ricker([0.2, 0.35], 2.0, [0.7, 1.2], [[0.9, 0.1], [0.1, 0.9]])
        # A completely different (still valid) set of Allee/theta/regime parameters.
        perturbed = self._ricker([0.15, 0.30], 3.5, [0.5, 1.5], [[0.8, 0.2], [0.2, 0.8]])

        # 1) one-step dynamics identical for every action and abundance.
        for action in range(3):
            for x in (0.0, 0.4, 0.9, 1.6):
                self.assertEqual(
                    float(default.noiseless_next(x, 1.0, action)),
                    float(perturbed.noiseless_next(x, 1.0, action)),
                )

        # 2) the discretized transition kernels are byte-identical.
        context = public_context(3)
        planner_cfg = FaithfulPlannerConfig(name="pbvi", state_bins=17, horizon=3)
        pomdp_a = CandidatePOMDP(default, context, planner_cfg, seed=3)
        pomdp_b = CandidatePOMDP(perturbed, context, planner_cfg, seed=3)
        for action in range(3):
            self.assertTrue(
                np.array_equal(
                    pomdp_a.transition_matrix(1.0, action),
                    pomdp_b.transition_matrix(1.0, action),
                )
            )

        # 3) ricker uses a single regime (Pi never sampled).
        self.assertEqual(default.num_regimes, 1)
        self.assertEqual(perturbed.num_regimes, 1)

        # 4) planner action values identical.
        belief = pomdp_a.initial_belief(65.0)
        qa = PointBasedPlanner(pomdp_a, planner_cfg, 0.95, seed=1).action_values(belief)
        qb = PointBasedPlanner(pomdp_b, planner_cfg, 0.95, seed=1).action_values(belief)
        self.assertTrue(np.array_equal(qa, qb))


# --------------------------------------------------------------------------------------
# Task 3 - regime switch-count convention (25-transition episode -> 24 opportunities)
# --------------------------------------------------------------------------------------


class RegimeSwitchCountTests(unittest.TestCase):
    def test_expected_switches_over_25_step_episode_is_24_times_one_minus_p(self):
        # Reproduce the objective's exact indexing: z_0 from draw[0]; for each of the L
        # transitions use z_t then draw z_{t+1} from draw[t+1]. The regimes actually USED
        # in transitions are z_0..z_{L-1} (L states) -> L-1 switch opportunities.
        length = 25  # complete-episode length (config.dataset.episode_length)
        paths = 60_000
        for p in (0.80, 0.90, 0.97):
            matrix = fixed_regime_matrix(p)
            rng = np.random.default_rng(int(p * 1000))
            draws = rng.random((paths, length + 1))
            regimes = np.empty((paths, length), dtype=np.int64)
            z = (draws[:, 0] >= 0.5).astype(np.int64)
            for step in range(length):
                regimes[:, step] = z
                prob_zero = matrix[z, 0]
                z = (draws[:, step + 1] > prob_zero).astype(np.int64)
            switches = np.sum(regimes[:, 1:] != regimes[:, :-1], axis=1).mean()
            self.assertAlmostEqual(switches, 24.0 * (1.0 - p), delta=0.05)
            # The realised count is strictly closer to 24*(1-p) than to the previously
            # reported 25*(1-p) (the extra draw z_L is generated but never used).
            self.assertLess(
                abs(switches - 24.0 * (1.0 - p)), abs(switches - 25.0 * (1.0 - p))
            )


if __name__ == "__main__":
    unittest.main()
