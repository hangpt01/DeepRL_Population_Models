"""OGSRL: model-based categorical constrained policy with OOD guardian."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..beliefs import BeliefCache
from ..controls import advance_public_controls, control_fields_enabled
from ..dataset import TrajectoryDataset
from ..dynamics import ContinuousDynamicsEnsemble
from ..public_models import PublicDynamicsEnsemble
from ..reward import build_reward, safety_penalty_indicator
from ..types import BeliefState
from .base import BasePolicy


def _softmax1(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / exp.sum()


@dataclass
class KNNGuardian:
    anchors: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    num_actions: int
    K_ref: float
    safety_threshold: float
    k: int = 5

    @staticmethod
    def features(
        states: np.ndarray,
        actions: np.ndarray,
        num_actions: int,
        K_ref: float,
        safety_threshold: float,
        rho: np.ndarray | float | None = None,
        kappa: np.ndarray | float | None = None,
        K_eff: np.ndarray | float | None = None,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=int)
        states, actions = np.broadcast_arrays(states, actions)
        n = len(states.reshape(-1))
        flat_states = states.reshape(-1)
        flat_actions = actions.reshape(-1)
        safe_K = max(float(K_ref), 1.0)

        def control_column(value, default):
            if value is None:
                return np.full(n, float(default), dtype=np.float64)
            return np.broadcast_to(np.asarray(value, dtype=np.float64), states.shape).reshape(-1)

        rho_arr = control_column(rho, 0.0)
        kappa_arr = control_column(kappa, 0.0)
        if K_eff is None:
            K_eff_arr = np.maximum(float(K_ref) + kappa_arr, float(K_ref))
        else:
            K_eff_arr = control_column(K_eff, float(K_ref))

        action_one_hot = np.eye(num_actions, dtype=np.float64)[flat_actions]
        abundance = np.log1p(np.maximum(flat_states, 0.0) / safe_K)
        unsafe = (flat_states <= safety_threshold).astype(np.float64)
        safety_distance = (flat_states - safety_threshold) / safe_K
        return np.column_stack([
            abundance,
            unsafe,
            safety_distance,
            action_one_hot,
            rho_arr,
            kappa_arr / safe_K,
            K_eff_arr / safe_K,
        ])

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        num_actions: int,
        seed: int,
        K_ref: float,
        safety_threshold: float,
        rho: np.ndarray | float | None = None,
        kappa: np.ndarray | float | None = None,
        K_eff: np.ndarray | float | None = None,
        alpha: float = 0.05,
        max_anchors: int = 2048,
    ) -> "KNNGuardian":
        points = cls.features(
            states, actions, num_actions, K_ref, safety_threshold, rho, kappa, K_eff
        )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(points))
        split = min(max_anchors, max(32, int(0.8 * len(points))))
        anchors = points[order[:split]]
        validation = points[order[split:]]
        if len(validation) < 8:
            validation = anchors
        mean = anchors.mean(axis=0)
        scale = np.maximum(anchors.std(axis=0), 1e-6)
        anchors_z = (anchors - mean) / scale
        temp = cls(anchors_z, mean, scale, 0.0, num_actions, K_ref, safety_threshold)
        distances = temp.score(validation)
        threshold = float(np.quantile(distances, 1.0 - alpha))
        return cls(anchors_z, mean, scale, threshold, num_actions, K_ref, safety_threshold)

    def score(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        z = (points - self.mean) / self.scale
        result = np.empty(len(z))
        for start in range(0, len(z), 512):
            chunk = z[start : start + 512]
            distance = np.sum((chunk[:, None, :] - self.anchors[None, :, :]) ** 2, axis=2)
            kth = min(self.k - 1, distance.shape[1] - 1)
            result[start : start + len(chunk)] = np.sqrt(
                np.partition(distance, kth, axis=1)[:, kth]
            )
        return result

    def ood(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rho: np.ndarray | float | None = None,
        kappa: np.ndarray | float | None = None,
        K_eff: np.ndarray | float | None = None,
    ) -> np.ndarray:
        points = self.features(
            states, actions, self.num_actions, self.K_ref,
            self.safety_threshold, rho, kappa, K_eff,
        )
        return (self.score(points) > self.threshold).astype(np.float64)


@dataclass
class PublicKNNGuardian:
    anchors: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    num_actions: int
    observation_scale: float
    k: int = 5

    @staticmethod
    def features(observations, actions, num_actions, observation_scale):
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=int)
        observations, actions = np.broadcast_arrays(observations, actions)
        x = np.log1p(
            np.maximum(observations.reshape(-1), 0.0) / observation_scale
        )
        onehot = np.eye(num_actions, dtype=np.float64)[actions.reshape(-1)]
        return np.column_stack([x, x * x, onehot])

    @classmethod
    def fit(cls, observations, actions, num_actions, observation_scale, seed, alpha=0.05):
        points = cls.features(observations, actions, num_actions, observation_scale)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(points))
        split = min(2048, max(32, int(0.8 * len(points))))
        anchors = points[order[:split]]
        validation = points[order[split:]]
        if len(validation) < 8:
            validation = anchors
        mean = anchors.mean(axis=0)
        scale = np.maximum(anchors.std(axis=0), 1e-6)
        normalized = (anchors - mean) / scale
        temporary = cls(
            normalized, mean, scale, 0.0, num_actions, observation_scale
        )
        threshold = float(np.quantile(temporary.score(validation), 1.0 - alpha))
        return cls(
            normalized, mean, scale, threshold, num_actions, observation_scale
        )

    def score(self, points):
        normalized = (np.asarray(points, dtype=np.float64) - self.mean) / self.scale
        result = np.empty(len(normalized))
        for start in range(0, len(normalized), 512):
            chunk = normalized[start : start + 512]
            distance = np.sum(
                (chunk[:, None, :] - self.anchors[None, :, :]) ** 2, axis=2
            )
            kth = min(self.k - 1, distance.shape[1] - 1)
            result[start : start + len(chunk)] = np.sqrt(
                np.partition(distance, kth, axis=1)[:, kth]
            )
        return result

    def ood_probability(self, observations, actions):
        points = self.features(
            observations, actions, self.num_actions, self.observation_scale
        )
        return (self.score(points) > self.threshold).astype(np.float64)


class OGSRLPolicy(BasePolicy):
    name = "ogsrl"

    def __init__(
        self,
        *args,
        safety_budget: float = 0.02,
        ood_budget: float = 0.05,
        train_iterations: int = 30,
        deployment_safety_limit: float | None = None,
        deployment_ood_limit: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.safety_budget = safety_budget
        self.ood_budget = ood_budget
        self.train_iterations = train_iterations
        self.deployment_safety_limit = (
            safety_budget if deployment_safety_limit is None else deployment_safety_limit
        )
        self.deployment_ood_limit = (
            ood_budget if deployment_ood_limit is None else deployment_ood_limit
        )

    def _state_features(self, states: np.ndarray, rho=0.0, kappa=0.0) -> np.ndarray:
        x = np.log1p(np.maximum(states, 0.0) / self.env_cfg.K_ref)
        unsafe = (states <= self.env_cfg.safety_threshold).astype(float)
        columns = [np.ones(len(states)), x, x * x, unsafe]
        if control_fields_enabled(self.env_cfg):
            rho_arr = np.broadcast_to(np.asarray(rho, dtype=np.float64), states.shape)
            kappa_arr = np.broadcast_to(np.asarray(kappa, dtype=np.float64), states.shape)
            _rho_next, _kappa_next, K_eff = advance_public_controls(
                self.env_cfg, np.zeros(len(states), dtype=int), rho_arr, kappa_arr
            )
            columns.extend([rho_arr, kappa_arr / self.env_cfg.K_ref, K_eff / self.env_cfg.K_ref])
        return np.column_stack(columns)

    def _policy(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.actor_weights.T
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / exp.sum(axis=1, keepdims=True)

    def _sample_cached_beliefs(self, features: np.ndarray) -> np.ndarray:
        """Draw training starts from cached posterior log-state moments."""

        mean_log_state = features[:, 0]
        standard_deviation = np.maximum(features[:, 1], 0.0)
        draw = self.rng.normal(mean_log_state, standard_deviation)
        return self.env_cfg.K_ref * np.expm1(np.maximum(draw, 0.0))

    def _rollouts(
        self,
        start_states: np.ndarray,
        start_rho=None,
        start_kappa=None,
        horizon: int | None = None,
    ):
        # Explicit arg wins; otherwise take the rollout horizon from the planner config,
        # whose default (6) is the old literal -- so this is inert by construction.
        horizon = self.planner_cfg.ogsrl_rollout_horizon if horizon is None else horizon
        batch = len(start_states)
        states = start_states.copy()
        rho = np.zeros(batch) if start_rho is None else np.asarray(start_rho, dtype=np.float64).copy()
        kappa = np.zeros(batch) if start_kappa is None else np.asarray(start_kappa, dtype=np.float64).copy()
        alive = np.ones(batch, dtype=bool)
        records = []
        for _ in range(horizon):
            features = self._state_features(states, rho, kappa)
            probs = self._policy(features)
            actions = np.asarray([
                self.rng.choice(self.num_actions, p=p) for p in probs
            ], dtype=int)
            next_states, _ = self.dynamics.sample_next(
                states, actions, np.zeros(batch), np.zeros(batch, dtype=np.int8), self.rng,
                rho, kappa,
            )
            if control_fields_enabled(self.env_cfg):
                next_rho, next_kappa, _next_K_eff = advance_public_controls(
                    self.env_cfg, actions, rho, kappa
                )
            else:
                next_rho, next_kappa = rho, kappa
            crossing = alive & (states > self.env_cfg.safety_threshold) & (
                next_states <= self.env_cfg.safety_threshold
            )
            penalty_indicator = safety_penalty_indicator(
                self.env_cfg, states, next_states, crossing
            )
            costs = np.asarray([self.reward.actions[a].cost for a in actions])
            # Benefit on the TRUE next state s_{t+1} (spec E6); penalty mode-aware.
            reward = (
                self.reward.utility(next_states)
                - costs
                - self.reward.collapse_penalty * penalty_indicator
            )
            if control_fields_enabled(self.env_cfg):
                current_K_eff = np.clip(
                    self.env_cfg.K_base + kappa, self.env_cfg.K_min, self.env_cfg.K_max
                )
                ood = self.guardian.ood(states, actions, rho, kappa, current_K_eff)
            else:
                ood = self.guardian.ood(states, actions)
            records.append((features, probs, actions, reward, penalty_indicator.astype(float), ood))
            states = next_states
            rho = np.asarray(next_rho, dtype=np.float64)
            kappa = np.asarray(next_kappa, dtype=np.float64)
            alive &= ~crossing
        return records

    def _discounted_returns(self, records, index):
        running = np.zeros(len(records[0][0]))
        returns = []
        for record in reversed(records):
            running = record[index] + self.planner_cfg.discount * running
            returns.append(running.copy())
        return list(reversed(returns))

    def _rollout_training_metrics(self, records):
        reward_returns = self._discounted_returns(records, 3)
        safety_returns = self._discounted_returns(records, 4)
        ood_returns = self._discounted_returns(records, 5)
        objective = (
            reward_returns[0]
            - self.lambda_safety * safety_returns[0]
            - self.lambda_ood * ood_returns[0]
        )
        return reward_returns, safety_returns, ood_returns, {
            "surrogate_loss": float(-np.mean(objective)),
            "objective_mean": float(np.mean(objective)),
            "reward_return": float(np.mean(reward_returns[0])),
            "safety_cost": float(np.mean(safety_returns[0])),
            "ood_cost": float(np.mean(ood_returns[0])),
            "lambda_safety": float(self.lambda_safety),
            "lambda_ood": float(self.lambda_ood),
        }

    def _validation_metrics(self, beliefs: BeliefCache, batch_size: int):
        if len(beliefs.mean_states) == 0:
            return {}
        state = self.rng.bit_generator.state
        try:
            idx = self.rng.choice(
                len(beliefs.mean_states),
                size=min(batch_size, len(beliefs.mean_states)),
                replace=False,
            )
            records = self._rollouts(
                self._sample_cached_beliefs(beliefs.features[idx]),
                None if beliefs.rho is None else beliefs.rho[idx],
                None if beliefs.kappa is None else beliefs.kappa[idx],
            )
            _rr, _sr, _oo, metrics = self._rollout_training_metrics(records)
            return metrics
        finally:
            self.rng.bit_generator.state = state

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("OGSRL requires cached shared beliefs")
        if self.hidden:
            self.dynamics = PublicDynamicsEnsemble.fit(
                dataset,
                beliefs,
                self.public_context,
                self.model_cfg,
                self.seed,
            )
            self.guardian = PublicKNNGuardian.fit(
                beliefs.mean_states,
                dataset.actions,
                self.num_actions,
                self.public_context.observation_scale,
                self.seed,
            )
            self.surrogate = self.public_context.surrogate
            diagnostics = dict(self.surrogate.diagnostics)
            diagnostics.update(
                {
                    "guardian_threshold": self.guardian.threshold,
                    "guardian_feature_dim": float(self.guardian.anchors.shape[1]),
                    "expose_rk_hidden": 1.0,
                }
            )
            return diagnostics
        self.dynamics = ContinuousDynamicsEnsemble.fit(
            dataset, beliefs, self.num_actions, self.model_cfg.ensemble_size,
            self.model_cfg.ridge, self.env_cfg.K_ref, self.seed, self.env_cfg,
        )
        self.guardian = KNNGuardian.fit(
            beliefs.mean_states,
            dataset.actions,
            self.num_actions,
            self.seed,
            self.env_cfg.K_ref,
            self.env_cfg.safety_threshold,
            beliefs.rho,
            beliefs.kappa,
            beliefs.K_eff,
        )
        self.reward = build_reward(self.env_cfg)
        feature_dim = self._state_features(
            np.asarray([max(self.env_cfg.N0, 1.0)]),
            0.0,
            0.0,
        ).shape[1]
        self.actor_weights = self.rng.normal(0.0, 0.02, (self.num_actions, feature_dim))
        self.lambda_safety = 1.0
        self.lambda_ood = 1.0
        batch_size = min(128, len(dataset))
        last_safety = last_ood = 0.0
        holdout_beliefs = self.training_holdout_beliefs
        for iteration in range(self.train_iterations):
            idx = self.rng.integers(0, len(dataset), batch_size)
            records = self._rollouts(
                self._sample_cached_beliefs(beliefs.features[idx]),
                None if beliefs.rho is None else beliefs.rho[idx],
                None if beliefs.kappa is None else beliefs.kappa[idx],
            )
            reward_returns, safety_returns, ood_returns, train_metrics = (
                self._rollout_training_metrics(records)
            )
            gradient = np.zeros_like(self.actor_weights)
            samples = 0
            critic_rows = []
            for record, rr, sr, oo in zip(records, reward_returns, safety_returns, ood_returns):
                features, probs, actions = record[:3]
                objective = rr - self.lambda_safety * sr - self.lambda_ood * oo
                objective -= objective.mean()
                for i in range(len(features)):
                    grad_logits = -probs[i]
                    grad_logits[actions[i]] += 1.0
                    gradient += objective[i] * grad_logits[:, None] * features[i][None, :]
                critic_rows.append((features, actions, rr, sr, oo))
                samples += len(features)
            self.actor_weights += 0.03 * gradient / max(samples, 1)
            last_safety = float(np.mean(safety_returns[0]))
            last_ood = float(np.mean(ood_returns[0]))
            self.lambda_safety = max(
                0.0, self.lambda_safety + 0.1 * (last_safety - self.safety_budget)
            )
            self.lambda_ood = max(
                0.0, self.lambda_ood + 0.1 * (last_ood - self.ood_budget)
            )
            self.log_training(iteration, "train", train_metrics, phase="actor")
            if holdout_beliefs is not None:
                self.log_training(
                    iteration,
                    "holdout",
                    self._validation_metrics(holdout_beliefs, batch_size),
                    phase="actor",
                )
        # Linear reward/safety/OOD critics for diagnostics and constrained evaluation.
        X_parts = []
        targets = [[], [], []]
        for features, actions, rr, sr, oo in critic_rows:
            X_parts.append(np.column_stack([features, np.eye(self.num_actions)[actions]]))
            targets[0].append(rr)
            targets[1].append(sr)
            targets[2].append(oo)
        X = np.vstack(X_parts)
        self.critics = []
        for chunks in targets:
            y = np.concatenate(chunks)
            self.critics.append(
                np.linalg.solve(
                    X.T @ X + self.model_cfg.ridge * np.eye(X.shape[1]), X.T @ y
                )
            )
        return {
            "guardian_threshold": self.guardian.threshold,
            "guardian_feature_dim": float(self.guardian.anchors.shape[1]),
            "lambda_safety": self.lambda_safety,
            "lambda_ood": self.lambda_ood,
            "modeled_safety_cost": last_safety,
            "modeled_ood_cost": last_ood,
        }

    def _belief_action_risks(self, belief: BeliefState) -> tuple[np.ndarray, np.ndarray]:
        """Posterior OOD and next-step unsafe-occupancy probabilities."""

        states = np.asarray(belief.states, dtype=np.float64)
        weights = belief.weights
        ood_probability = np.zeros(self.num_actions)
        unsafe_probability = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            actions = np.full(len(states), action, dtype=int)
            ood_probability[action] = float(
                np.sum(
                    weights
                    * self.guardian.ood(
                        states,
                        actions,
                        None if belief.rho is None else float(belief.rho),
                        None if belief.kappa is None else float(belief.kappa),
                        None if belief.K_eff is None else float(belief.K_eff),
                    )
                )
            )
            _mean, _variance, member_predictions = self.dynamics.predict(
                states,
                actions,
                0.0 if belief.rho is None else belief.rho,
                0.0 if belief.kappa is None else belief.kappa,
            )
            member_predictions[:, states == 0.0] = 0.0
            particle_unsafe = np.mean(
                member_predictions <= self.env_cfg.safety_threshold, axis=0
            )
            unsafe_probability[action] = float(np.sum(weights * particle_unsafe))
        return ood_probability, unsafe_probability

    def act(self, belief: BeliefState, observation: float) -> int:
        if self.hidden:
            count = len(belief.states)
            previous = (
                belief.contexts[:, 0]
                if belief.contexts.ndim > 1
                else np.full(count, observation)
            )
            values = np.zeros(self.num_actions, dtype=np.float64)
            risks = np.zeros(self.num_actions, dtype=np.float64)
            ood = np.zeros(self.num_actions, dtype=np.float64)
            for action in range(self.num_actions):
                actions = np.full(count, action, dtype=int)
                mean, variance, _ = self.dynamics.predict(belief.states, actions)
                reward, risk = self.surrogate.predict(
                    previous,
                    belief.states,
                    mean,
                    actions,
                    np.full(count, belief.timestep),
                    np.full(count, self.public_context.pop_id),
                )
                values[action] = float(
                    np.sum(belief.weights * reward)
                    - self.planner_cfg.pessimism * np.mean(np.sqrt(variance))
                )
                risks[action] = float(np.sum(belief.weights * risk))
                ood[action] = float(
                    np.sum(
                        belief.weights
                        * self.guardian.ood_probability(belief.states, actions)
                    )
                )
            feasible = (risks <= self.deployment_safety_limit) & (
                ood <= self.deployment_ood_limit
            )
            hard_fallback = not bool(np.any(feasible))
            if hard_fallback:
                violation = (
                    np.maximum(risks - self.deployment_safety_limit, 0.0)
                    / max(self.deployment_safety_limit, 1e-6)
                    + np.maximum(ood - self.deployment_ood_limit, 0.0)
                    / max(self.deployment_ood_limit, 1e-6)
                )
                action = int(np.argmin(violation - 1e-6 * values))
            else:
                action = int(np.argmax(np.where(feasible, values, -np.inf)))
            self.last_diagnostics = {
                "action_scores": values.tolist(),
                "public_extinction_risk": float(risks[action]),
                "ood_probability": float(ood[action]),
                "hard_fallback": hard_fallback,
                "risk_fallback": self.surrogate.diagnostics.get(
                    "risk_fallback", "none"
                ),
                "terminated_prevalence": self.surrogate.diagnostics.get(
                    "terminated_prevalence", 0.0
                ),
            }
            return action
        particle_probabilities = self._policy(
            self._state_features(
                belief.states,
                0.0 if belief.rho is None else belief.rho,
                0.0 if belief.kappa is None else belief.kappa,
            )
        )
        probs = belief.weights @ particle_probabilities
        unconstrained_action = int(np.argmax(probs))
        ood_probability, unsafe_probability = self._belief_action_risks(belief)
        feasible = (
            (ood_probability <= self.deployment_ood_limit)
            & (unsafe_probability <= self.deployment_safety_limit)
        )
        hard_fallback = not bool(np.any(feasible))
        if hard_fallback:
            # If no action lies in the estimated safe/support set, take the
            # least-violating action.  Actor probability only breaks ties.
            safety_scale = max(self.deployment_safety_limit, 1e-6)
            ood_scale = max(self.deployment_ood_limit, 1e-6)
            violation = (
                np.maximum(unsafe_probability - self.deployment_safety_limit, 0.0)
                / safety_scale
                + np.maximum(ood_probability - self.deployment_ood_limit, 0.0)
                / ood_scale
            )
            action = int(np.argmin(violation - 1e-6 * probs))
        else:
            guarded_probabilities = np.where(feasible, probs, -np.inf)
            action = int(np.argmax(guarded_probabilities))
        self.last_diagnostics = {
            "action_probabilities": probs.tolist(),
            "ood_probability": float(ood_probability[action]),
            "ood_probability_all": ood_probability.tolist(),
            "unsafe_probability": float(unsafe_probability[action]),
            "unsafe_probability_all": unsafe_probability.tolist(),
            "guardian_override": float(action != unconstrained_action),
            "hard_fallback": float(hard_fallback),
            "lambda_safety": self.lambda_safety,
            "lambda_ood": self.lambda_ood,
        }
        return action
