"""Constructed paper-mechanism gates for the corrected hidden general-RL baselines.

These tests exercise the *defining* mechanism of each method on small, deterministic
hidden-r/K problems, without reading any performance return:

* RefPlan  -- the deployment model posterior materially changes planning, and the
              method is behaviorally distinct from the uniform-ensemble MOPO planner.
* BA-MCTS  -- the categorical ensemble belief is updated *inside* simulated histories
              (Eq.-4 style), concentrates on the data-consistent member, yields
              history-dependent actions, and is distinct from root-sampling BAMCP.
* Ensemble value-disagreement pessimism -- independently bootstrapped conservative
              fitted-Q members disagree without direct output perturbations.

All fixtures use the public hidden pipeline only (MethodContext + public surrogate +
PublicObservationFilter); no private r/K/family/safety quantity is accessed.
"""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from real_ecology_benchmark import realdata
from real_ecology_benchmark.beliefs import PublicObservationFilter, cache_public_beliefs
from real_ecology_benchmark.collector import collect_dataset
from real_ecology_benchmark.config import (
    FilterConfig,
    MethodContext,
    ModelConfig,
    PlannerConfig,
    real_environment,
)
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy
from real_ecology_benchmark.methods.mopo import MOPOPolicy
from real_ecology_benchmark.methods.refplan import RefPlanPolicy
from real_ecology_benchmark.public_surrogate import fit_public_surrogate


def _hidden_fixture(sigma_obs: float = 0.2, transitions: int = 240, episode: int = 12):
    env_cfg = real_environment(
        "Amur tiger", "ricker", expose_rk="hidden", observation_noise_sigma=sigma_obs
    )
    dataset, private = collect_dataset(make_env(env_cfg), transitions, episode, seed=116)
    surrogate = fit_public_surrogate(dataset, seed=20_116)
    context = MethodContext(
        num_actions=env_cfg.num_actions,
        action_costs=tuple(float(v) for v in dataset.action_costs),
        action_channels=realdata.public_action_channels(),
        observation_noise_sigma=env_cfg.observation_noise_sigma,
        horizon=env_cfg.horizon,
        observation_scale=surrogate.observation_scale,
        pop_id=str(dataset.pop_ids[0]),
        reward_mode=env_cfg.reward_mode,
        surrogate=surrogate,
    )
    filter_cfg = FilterConfig(particles=48, proposal="learned")
    factory = lambda: PublicObservationFilter(context, filter_cfg)
    cache = cache_public_beliefs(
        dataset, factory, seed=30_116, observation_scale=context.observation_scale
    )
    return env_cfg, dataset, private, context, filter_cfg, factory, cache


class RefPlanMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.env_cfg, cls.dataset, cls.private, cls.context,
         cls.filter_cfg, factory, cls.cache) = _hidden_fixture()
        cls.factory = staticmethod(factory)
        cls.model_cfg = ModelConfig(ensemble_size=6)
        cls.planner_cfg = PlannerConfig(horizon=4, sequences=48, particles=16, pessimism=0.5)

    def _fit_refplan(self):
        policy = RefPlanPolicy(self.context, self.model_cfg, self.planner_cfg, seed=7)
        policy.fit(self.dataset, self.cache)
        return policy

    def _belief(self):
        return self.factory().reset(float(self.dataset.observations[0]), 55)

    def test_hidden_posterior_materially_changes_plan_scores(self):
        """Same belief, same models, different model posterior => different scores."""
        policy = self._fit_refplan()
        belief = self._belief()
        n = len(policy.dynamics.members)

        def scores_for(posterior):
            # Reset the planner RNG so particle starts / per-member streams are
            # identical across calls; any difference is caused only by posterior.
            policy.planner.rng = np.random.default_rng(999)
            _a, diag = policy.planner.plan_marginalized(
                belief, policy.dynamics, posterior, pessimism=0.5
            )
            return np.asarray(diag.action_scores)

        uniform = np.full(n, 1.0 / n)
        conc0 = np.zeros(n); conc0[0] = 1.0
        conc1 = np.zeros(n); conc1[min(1, n - 1)] = 1.0
        s_uniform, s0, s1 = scores_for(uniform), scores_for(conc0), scores_for(conc1)

        # The posterior must move the finite action-score vector.
        self.assertGreater(float(np.nanmax(np.abs(s0 - s_uniform))), 1e-6)
        self.assertGreater(float(np.nanmax(np.abs(s0 - s1))), 1e-6)

    def test_at_least_two_members_prefer_different_actions(self):
        """Ensemble genuinely disagrees, so marginalization is not vacuous."""
        policy = self._fit_refplan()
        belief = self._belief()
        n = len(policy.dynamics.members)
        argmaxes = set()
        for i in range(n):
            posterior = np.zeros(n); posterior[i] = 1.0
            policy.planner.rng = np.random.default_rng(999)
            action, _ = policy.planner.plan_marginalized(
                belief, policy.dynamics, posterior, pessimism=0.5
            )
            argmaxes.add(int(action))
        self.assertGreaterEqual(
            len(argmaxes), 2,
            "single-member posteriors all select the same action; marginalization inert",
        )

    def test_refplan_distinct_from_mopo_under_concentrated_posterior(self):
        """RefPlan with a non-uniform deployment posterior differs from MOPO.

        MOPO plans over the uniform ensemble; RefPlan marginalizes b(theta).  We
        drive the RefPlan posterior toward whichever single member most disagrees
        with the uniform-ensemble choice and require a different selected action
        for at least one such member (otherwise the two are behaviorally equal).
        """
        refplan = self._fit_refplan()
        mopo = MOPOPolicy(self.context, self.model_cfg, self.planner_cfg, seed=7)
        mopo.fit(self.dataset, self.cache)
        belief = self._belief()
        mopo.planner.rng = np.random.default_rng(999)
        mopo_action, _ = mopo.planner.plan(belief, mopo.dynamics, pessimism=0.5)

        n = len(refplan.dynamics.members)
        differing = []
        for i in range(n):
            posterior = np.zeros(n); posterior[i] = 1.0
            refplan.planner.rng = np.random.default_rng(999)
            action, _ = refplan.planner.plan_marginalized(
                belief, refplan.dynamics, posterior, pessimism=0.5
            )
            differing.append(int(action) != int(mopo_action))
        self.assertTrue(
            any(differing),
            "no concentrated posterior changes the action relative to MOPO",
        )

    def test_observe_updates_posterior_away_from_uniform(self):
        """Deployment belief update actually moves the posterior."""
        policy = self._fit_refplan()
        belief = self._belief()
        before = policy.posterior.copy()
        # Feed a public transition; the residual-likelihood update must reweight.
        from real_ecology_benchmark.types import PublicTransition

        result = PublicTransition(
            observation=float(self.dataset.next_observations[0]),
            done=False, truncated=False, terminated=False,
        )
        policy.observe(belief, int(self.dataset.actions[0]), result)
        after = policy.posterior
        self.assertAlmostEqual(float(after.sum()), 1.0, places=6)
        self.assertGreater(float(np.max(np.abs(after - before))), 1e-9)

    def test_public_policy_prior_is_live_and_distinct(self):
        """Swapping only pi_prior changes a candidate plan/action."""
        policy = self._fit_refplan()
        belief = self._belief()

        class FixedPrior:
            def __init__(self, action, count):
                self.action = action
                self.count = count

            def probabilities(self, features):
                probs = np.zeros((len(features), self.count))
                probs[:, self.action] = 1.0
                return probs

        changed = False
        for seed in range(12):
            policy.planner.rng = np.random.default_rng(seed)
            a0, _ = policy.planner.plan_marginalized(
                belief, policy.dynamics, policy.posterior,
                policy_prior=FixedPrior(0, policy.num_actions), prior_epsilon=0.10,
            )
            policy.planner.rng = np.random.default_rng(seed)
            a1, _ = policy.planner.plan_marginalized(
                belief, policy.dynamics, policy.posterior,
                policy_prior=FixedPrior(policy.num_actions - 1, policy.num_actions),
                prior_epsilon=0.10,
            )
            changed |= a0 != a1
        self.assertTrue(changed, "policy-prior swap never changed plan selection")
        self.assertIsNot(policy.policy_prior, policy.posterior)
        floor = policy.prior_epsilon / policy.num_actions
        mixed = (1.0 - policy.prior_epsilon) * np.array([1.0] + [0.0] * 10) + floor
        self.assertGreater(float(np.min(mixed)), 0.0)

    def test_prior_is_reevaluated_on_simulated_continuations(self):
        """Identical root priors branch at later proposal states."""
        policy = self._fit_refplan()
        belief = self._belief()

        class StateDependentPrior:
            def __init__(self, count, threshold):
                self.count = count
                self.threshold = threshold
                self.seen = []

            def probabilities(self, features):
                self.seen.append(features.copy())
                probs = np.zeros((len(features), self.count))
                # Root rows have the same public-feature mean; later predictive
                # histories split and therefore propose different actions.
                action = (features[:, 0] > self.threshold).astype(int)
                probs[np.arange(len(features)), action] = 1.0
                return probs

        root = belief.public_features(self.context.observation_scale)[0]
        prior = StateDependentPrior(policy.num_actions, root + 1e-6)
        policy.planner.rng = np.random.default_rng(123)
        policy.planner.plan_marginalized(
            belief, policy.dynamics, policy.posterior,
            policy_prior=prior, prior_epsilon=0.0,
        )
        sequences = policy.planner.last_proposal_sequences
        self.assertEqual(len(prior.seen), self.planner_cfg.horizon)
        self.assertTrue(np.allclose(prior.seen[0][:, 0], prior.seen[0][0, 0]))
        self.assertGreater(np.ptp(prior.seen[1][:, 0]), 0.0)
        self.assertGreater(len(np.unique(sequences[:, 1])), 1)


class BAMCTSMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.env_cfg, cls.dataset, cls.private, cls.context,
         cls.filter_cfg, factory, cls.cache) = _hidden_fixture()
        cls.factory = staticmethod(factory)
        cls.model_cfg = ModelConfig(ensemble_size=6)
        cls.planner_cfg = PlannerConfig(horizon=4, sequences=32, particles=8)

    def _fit(self, simulations=64, depth=6):
        policy = BAMCTSPolicy(
            self.context, self.model_cfg, self.planner_cfg,
            simulations=simulations, depth=depth, seed=7,
        )
        policy.fit(self.dataset, self.cache)
        return policy

    def test_belief_from_logweights_is_numerically_stable(self):
        policy = self._fit(simulations=8, depth=2)
        n = len(policy.dynamics.members)
        # Uniform log-weights -> uniform belief.
        b = policy._belief_from_logweights(np.zeros(n))
        self.assertTrue(np.allclose(b, 1.0 / n))
        # Extreme / degenerate inputs never produce NaN or all-zero.
        for lw in (np.full(n, -1e9), np.array([-np.inf] * n), np.linspace(-800, 800, n)):
            out = policy._belief_from_logweights(lw)
            self.assertAlmostEqual(float(out.sum()), 1.0, places=6)
            self.assertTrue(np.all(np.isfinite(out)))

    def test_intree_belief_concentrates_on_generating_member(self):
        """Eq.-4 update: transitions generated by member k concentrate belief on k."""
        policy = self._fit()
        n = len(policy.dynamics.members)
        k = min(2, n - 1)
        member = policy.dynamics.members[k]
        log_belief = np.log(np.full(n, 1.0 / n))
        rng = np.random.default_rng(0)
        # Feed several *noise-free* member-k transitions (its predicted mean).
        current = float(np.median(self.dataset.observations[self.dataset.observations > 0]))
        for _ in range(12):
            action = int(rng.integers(0, policy.num_actions))
            following = float(member.mean_next(np.asarray([current]), np.asarray([action]))[0])
            log_belief = log_belief + policy._public_member_loglik(current, action, following)
            log_belief -= log_belief.max()
            current = following if following > 0 else current
        belief = policy._belief_from_logweights(log_belief)
        self.assertEqual(int(np.argmax(belief)), k)
        self.assertGreater(belief[k], 1.0 / n)

    def test_belief_moves_from_root_so_not_root_sampling(self):
        """A single consistent path must move the belief away from the root prior."""
        policy = self._fit()
        n = len(policy.dynamics.members)
        member = policy.dynamics.members[0]
        root = np.full(n, 1.0 / n)
        log_belief = np.log(root)
        current = float(np.median(self.dataset.observations[self.dataset.observations > 0]))
        following = float(member.mean_next(np.asarray([current]), np.asarray([0]))[0])
        log_belief = log_belief + policy._public_member_loglik(current, 0, following)
        moved = policy._belief_from_logweights(log_belief)
        self.assertGreater(float(np.max(np.abs(moved - root))), 1e-6,
                           "belief did not update in-tree (would be root sampling)")

    def test_root_posterior_seed_changes_action_choice(self):
        """History/belief dependence: different root beliefs can change the action."""
        policy = self._fit(simulations=96, depth=6)
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        n = len(policy.dynamics.members)

        def action_for(posterior):
            policy.posterior = np.asarray(posterior, dtype=float)
            policy.rng = np.random.default_rng(123)
            return policy.act(belief, float(self.dataset.observations[0]))

        conc0 = np.zeros(n); conc0[0] = 1.0
        conc1 = np.zeros(n); conc1[min(3, n - 1)] = 1.0
        a0 = action_for(conc0)
        q0 = np.asarray(policy.last_diagnostics["root_q"])
        a1 = action_for(conc1)
        q1 = np.asarray(policy.last_diagnostics["root_q"])
        self.assertGreater(float(np.max(np.abs(q0 - q1))), 1e-9,
                           "root belief seed does not affect the search")


class OGSRLMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.env_cfg, cls.dataset, cls.private, cls.context,
         cls.filter_cfg, factory, cls.cache) = _hidden_fixture()
        cls.factory = staticmethod(factory)
        cls.model_cfg = ModelConfig(ensemble_size=5)
        cls.planner_cfg = PlannerConfig(
            horizon=4, sequences=32, particles=8, ogsrl_cost_horizon=6,
            ogsrl_deployment_rollouts=32,
        )

    def _fit(self):
        from real_ecology_benchmark.methods.ogsrl import OGSRLPolicy

        policy = OGSRLPolicy(self.context, self.model_cfg, self.planner_cfg, seed=7)
        diag = policy.fit(self.dataset, self.cache)
        return policy, diag

    def test_hidden_conopt_actor_trains(self):
        policy, diag = self._fit()
        self.assertEqual(diag.get("hidden_actor_trained"), 1.0)
        self.assertTrue(hasattr(policy, "actor_weights"))
        self.assertGreater(
            abs(diag["actor_norm_end"] - diag["actor_norm_start"]), 1e-4,
            "actor weights did not move -- ConOpt did not train",
        )

    def test_ood_dual_binds_and_guardian_restricts(self):
        """The OOD constraint drives the dual and the guardian filters actions."""
        policy, diag = self._fit()
        # lambda_ood should move off its init of 1.0 because guardian OOD cost is
        # non-degenerate (the safety-cost channel may not, and that is expected).
        self.assertNotAlmostEqual(diag["lambda_ood"], 1.0, places=3)
        # Construct an out-of-support belief so at least one action is flagged OOD.
        belief = self.factory().reset(float(np.max(self.dataset.observations)) * 2.0, 55)
        ood, risks = policy._public_belief_action_risks(belief)
        self.assertGreater(float(np.max(ood)), 0.0, "guardian never flags OOD")

    def test_public_low_abundance_cost_and_safety_dual_bind(self):
        """Public shortfall is non-degenerate and moves the safety dual."""
        policy, diag = self._fit()
        values = policy._public_low_abundance_cost(
            np.array([0.0, 0.5 * policy.s_low, policy.s_low, 2.0 * policy.s_low])
        )
        self.assertTrue(np.allclose(values, [1.0, 0.5, 0.0, 0.0]))
        self.assertGreaterEqual(diag["low_abundance_cost_prevalence"], 0.05)
        self.assertLessEqual(diag["low_abundance_cost_prevalence"], 0.30)
        self.assertGreaterEqual(diag["safety_budget"], 0.0)
        self.assertLessEqual(diag["safety_budget"], 1.0)
        self.assertEqual(diag["cost_horizon"], 6.0)
        self.assertEqual(diag["behavior_cost_episode_count"], 20.0)
        self.assertEqual(diag["safety_dual_moved"], 1.0)
        self.assertNotAlmostEqual(diag["lambda_safety"], 1.0, places=5)

    def test_normalized_discounted_occupancy_invariants(self):
        from real_ecology_benchmark.methods.ogsrl import normalized_discounted_occupancy

        for horizon in (1, 6, 25):
            self.assertEqual(float(normalized_discounted_occupancy(np.zeros(horizon), 0.95)), 0.0)
            self.assertAlmostEqual(
                float(normalized_discounted_occupancy(np.ones(horizon), 0.95)), 1.0
            )
        hand = normalized_discounted_occupancy(np.array([1.0, 0.0, 0.5]), 0.5)
        self.assertAlmostEqual(float(hand), (1.0 + 0.25 * 0.5) / 1.75)
        self.assertAlmostEqual(
            float(normalized_discounted_occupancy(np.array([0.0, 1.0, 0.5]), 1.0)),
            0.5,
        )

    def test_behavior_budget_is_exact_unclipped_episode_mean(self):
        from real_ecology_benchmark.methods.ogsrl import normalized_discounted_occupancy

        policy, diag = self._fit()
        step_cost = policy._public_low_abundance_cost(self.dataset.next_observations)
        episode_values = []
        for episode in np.unique(self.dataset.episode_id):
            values = step_cost[self.dataset.episode_id == episode][:6]
            episode_values.append(normalized_discounted_occupancy(values, 0.95))
        expected = float(np.mean(episode_values))
        self.assertAlmostEqual(diag["safety_budget"], expected, places=12)
        self.assertEqual(policy.safety_budget, expected)
        self.assertEqual(policy.deployment_safety_limit, expected)

    def test_actor_and_deployment_share_pathwise_cost_helper(self):
        policy, _ = self._fit()
        starts = np.asarray(self.dataset.observations[:8], dtype=float)
        with patch.object(
            policy,
            "_public_pathwise_cost_rollout",
            wraps=policy._public_pathwise_cost_rollout,
        ) as shared:
            policy._public_rollouts(starts, horizon=2)
            self.assertEqual(shared.call_count, 1)
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        with patch.object(
            policy,
            "_public_pathwise_cost_rollout",
            wraps=policy._public_pathwise_cost_rollout,
        ) as shared:
            policy._public_belief_action_risks(belief)
            self.assertEqual(shared.call_count, policy.num_actions)

    def test_pathwise_cost_precedes_average_and_strict_jensen_gap(self):
        policy, _ = self._fit()

        class CrossingMember:
            def sample_next_from_standard_normal(self, observations, actions, standard_normal):
                del observations, actions
                return np.where(np.asarray(standard_normal) < 0.0, 0.0, 2.0 * policy.s_low)

        original = policy.dynamics
        try:
            policy.dynamics = SimpleNamespace(members=[CrossingMember()])
            steps = policy._public_pathwise_cost_rollout(
                np.full(2, policy.s_low),
                rng=np.random.default_rng(1),
                horizon=1,
                member_schedule=np.zeros((1, 2), dtype=int),
                residual_draws=np.array([[-1.0, 1.0]]),
                action_uniforms=np.array([[0.25, 0.25]]),
            )
            expected_pathwise = float(np.mean(steps[0].cost))
            cost_of_mean = float(policy._public_low_abundance_cost(
                np.array([np.mean(steps[0].following)])
            )[0])
            self.assertEqual(expected_pathwise, 0.5)
            self.assertEqual(cost_of_mean, 0.0)
            self.assertGreater(expected_pathwise, cost_of_mean)
        finally:
            policy.dynamics = original

    def test_deployment_is_repeatable_and_does_not_mutate_policy_rng(self):
        policy, _ = self._fit()
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        before = deepcopy(policy.rng.bit_generator.state)
        first = policy._public_belief_action_risks(belief)
        middle = deepcopy(policy.rng.bit_generator.state)
        second = policy._public_belief_action_risks(belief)
        after = deepcopy(policy.rng.bit_generator.state)
        self.assertEqual(before, middle)
        self.assertEqual(before, after)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(policy.last_risk_horizon, 6)
        self.assertEqual(policy.last_risk_rollouts, 32)

    def test_deployment_reuses_common_random_numbers_across_actions(self):
        policy, _ = self._fit()
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        captured = []
        original = policy._public_pathwise_cost_rollout

        def recording(*args, **kwargs):
            captured.append(tuple(np.asarray(kwargs[key]).copy() for key in (
                "member_schedule", "residual_draws", "action_uniforms"
            )))
            return original(*args, **kwargs)

        with patch.object(policy, "_public_pathwise_cost_rollout", side_effect=recording):
            policy._public_belief_action_risks(belief)
        self.assertEqual(len(captured), policy.num_actions)
        for candidate in captured[1:]:
            for left, right in zip(captured[0], candidate):
                self.assertTrue(np.array_equal(left, right))

    def test_deployment_member_schedule_is_balanced(self):
        policy, _ = self._fit()
        schedule, _residuals, _uniforms = policy._public_random_plan(
            256, 25, np.random.default_rng(123), balanced_members=True
        )
        for row in schedule:
            counts = np.bincount(row, minlength=len(policy.dynamics.members))
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_hidden_action_is_actor_driven_with_feasibility_mask(self):
        policy, _ = self._fit()
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        action = policy.act(belief, float(self.dataset.observations[0]))
        self.assertIn(action, range(self.env_cfg.num_actions))
        self.assertIn("action_probabilities", policy.last_diagnostics)
        self.assertIn("guardian_override", policy.last_diagnostics)

    def test_action_choice_responds_to_normalized_low_abundance_risk(self):
        from unittest.mock import patch

        policy, _ = self._fit()
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        policy.actor_weights[:] = 0.0
        policy.deployment_safety_limit = 0.5
        policy.deployment_ood_limit = 1.0
        policy.last_risk_horizon = self.planner_cfg.ogsrl_cost_horizon
        first_risk = np.zeros(policy.num_actions); first_risk[0] = 1.0
        second_risk = np.zeros(policy.num_actions); second_risk[1:] = 1.0
        with patch.object(
            policy, "_public_belief_action_risks",
            return_value=(np.zeros(policy.num_actions), first_risk),
        ):
            first = policy.act(belief, belief.observation)
        with patch.object(
            policy, "_public_belief_action_risks",
            return_value=(np.zeros(policy.num_actions), second_risk),
        ):
            second = policy.act(belief, belief.observation)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)


class EnsembleValueDisagreementMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.env_cfg, cls.dataset, cls.private, cls.context,
         cls.filter_cfg, factory, cls.cache) = _hidden_fixture(sigma_obs=0.2)
        cls.factory = staticmethod(factory)
        cls.model_cfg = ModelConfig(ensemble_size=5)
        cls.planner_cfg = PlannerConfig()

    def _fit(self, ensemble_size=8, seed=7):
        from real_ecology_benchmark.methods.ensemble_value_disagreement import (
            EnsembleValueDisagreementPolicy,
        )

        policy = EnsembleValueDisagreementPolicy(
            self.context, self.model_cfg, self.planner_cfg,
            ensemble_size=ensemble_size, seed=seed,
        )
        diag = policy.fit(self.dataset, self.cache)
        return policy, diag

    def test_members_differ_only_through_episode_bootstrap_fit(self):
        policy, diag = self._fit()
        bootstraps = [tuple(member.bootstrap_episode_ids) for member in policy.q_members]
        self.assertGreater(len(set(bootstraps)), 1)
        weights = np.stack([member.q_weights for member in policy.q_members])
        self.assertGreater(float(np.max(np.var(weights, axis=0))), 0.0)
        self.assertFalse(any(hasattr(member, "perturbation") for member in policy.q_members))
        self.assertFalse(hasattr(policy, "world_support"))
        self.assertGreater(diag["q_ensemble_disagreement_unobserved"], 0.0)

    def test_disagreement_is_exact_empirical_q_variance(self):
        from real_ecology_benchmark.behavior_model import augment_features

        policy, _ = self._fit()
        features = self.cache.features[:9]
        fitted = np.stack([
            augment_features(features) @ member.q_weights.T for member in policy.q_members
        ])
        mean_q, disagreement, values = policy._statistics(features)
        self.assertTrue(np.array_equal(values, fitted))
        self.assertTrue(np.allclose(mean_q, fitted.mean(axis=0)))
        self.assertTrue(np.allclose(disagreement, fitted.var(axis=0)))

    def test_same_seed_reproduces_complete_fit(self):
        first, _ = self._fit(seed=17)
        second, _ = self._fit(seed=17)
        for left, right in zip(first.q_members, second.q_members):
            self.assertTrue(np.array_equal(left.bootstrap_episode_ids, right.bootstrap_episode_ids))
            self.assertTrue(np.array_equal(left.q_weights, right.q_weights))

    def test_registered_disagreement_penalty_can_change_selected_action(self):
        policy, _ = self._fit(ensemble_size=2)
        belief = self.factory().reset(float(self.dataset.observations[0]), 55)
        mean_q = np.zeros((1, policy.num_actions))
        variance = np.zeros_like(mean_q)
        mean_q[0, :2] = [1.00, 0.99]
        variance[0, :2] = [0.20, 0.00]
        values = np.stack([mean_q, mean_q])
        self.assertEqual(int(np.argmax(mean_q[0])), 0)
        with patch.object(policy, "_statistics", return_value=(mean_q, variance, values)):
            action = policy.act(belief, belief.observation)
        self.assertEqual(policy.disagreement_penalty, 0.1)
        self.assertEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
