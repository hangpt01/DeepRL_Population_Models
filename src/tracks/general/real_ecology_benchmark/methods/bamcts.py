"""Belief-rooted continuous BA-MCTS with model root sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from ..beliefs import BeliefCache
from ..controls import advance_public_controls, control_fields_enabled
from ..dataset import TrajectoryDataset
from ..dynamics import ContinuousDynamicsEnsemble
from ..reward import build_reward, safety_penalty_indicator
from ..public_models import PublicDynamicsEnsemble
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


@dataclass
class _Node:
    visits: int = 0
    action_visits: np.ndarray | None = None
    action_value: np.ndarray | None = None


class BAMCTSPolicy(BasePolicy):
    name = "bamcts"

    def __init__(
        self,
        *args,
        simulations: int | None = None,
        depth: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Explicit kwarg wins; otherwise take the search budget from the planner config.
        # The config defaults are the old literals (128/5), so this is inert unless a
        # caller deliberately overrides them.
        self.simulations = (
            self.planner_cfg.bamcts_simulations if simulations is None else simulations
        )
        self.depth = self.planner_cfg.bamcts_depth if depth is None else depth
        self.exploration = 1.25

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("BA-MCTS requires cached shared beliefs")
        size = max(self.model_cfg.ensemble_size, 5)
        if self.hidden:
            self.dynamics = PublicDynamicsEnsemble.fit(
                dataset,
                beliefs,
                self.public_context,
                self.model_cfg,
                self.seed,
                minimum_members=size,
            )
            self.posterior = np.full(len(self.dynamics.members), 1.0 / len(self.dynamics.members))
            return {"members": float(size), "simulations": float(self.simulations), "expose_rk_hidden": 1.0}
        self.dynamics = ContinuousDynamicsEnsemble.fit(
            dataset, beliefs, self.num_actions, size, self.model_cfg.ridge,
            self.env_cfg.K_ref, self.seed, self.env_cfg,
        )
        self.posterior = np.full(size, 1.0 / size)
        self.reward = build_reward(self.env_cfg)
        return {"members": float(size), "simulations": float(self.simulations)}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if hasattr(self, "dynamics"):
            self.posterior = np.full(len(self.dynamics.members), 1.0 / len(self.dynamics.members))

    def _key(self, state: float, depth: int, unsafe: bool, rho: float = 0.0, kappa: float = 0.0):
        # Continuous progressive aggregation: finer near the safety boundary.
        width = 0.05 if state < 2 * self.env_cfg.safety_threshold else 0.15
        bucket = int(round(math.log1p(state / self.env_cfg.K_ref) / width))
        if control_fields_enabled(self.env_cfg):
            rho_bucket = int(round(rho / 0.02))
            kappa_bucket = int(round(kappa / 20.0))
            return depth, bucket, bool(unsafe), rho_bucket, kappa_bucket
        return depth, bucket, bool(unsafe)

    def _public_key(self, observation: float, depth: int):
        width = 0.1
        bucket = int(
            round(
                math.log1p(max(observation, 0.0) / self.public_context.observation_scale)
                / width
            )
        )
        return depth, bucket

    @staticmethod
    def _belief_from_logweights(log_weights: np.ndarray) -> np.ndarray:
        """Numerically stable categorical belief from log-weights.

        Subtracts the max before exponentiating and floors/renormalizes; on
        total underflow or a non-finite sum it resets to a uniform belief so the
        Bayes-adaptive state can never become NaN or all-zero.
        """
        log_weights = np.asarray(log_weights, dtype=np.float64)
        max_lw = np.max(log_weights)
        if not np.isfinite(max_lw):
            # All members ruled out (e.g. every log-likelihood -inf): reset to
            # uniform rather than propagating NaN.
            return np.full(len(log_weights), 1.0 / len(log_weights))
        weights = np.exp(np.clip(log_weights - max_lw, -700.0, 0.0))
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(log_weights), 1.0 / len(log_weights))
        return weights / total

    def _public_member_loglik(self, current: float, action: int, following: float) -> np.ndarray:
        """Per-member log P(following | current, action) in log-observation space.

        Uses only public observations and the public ensemble members -- no
        private state, reward, r, K, or safety quantity enters the update.  The
        public reward is produced by the *shared* surrogate (not per-member), so
        there is no per-member reward likelihood to include (documented Eq.-4
        adaptation: transition-likelihood only).
        """
        scale = self.public_context.observation_scale
        z = math.log1p(max(following, 0.0) / scale)
        logliks = np.empty(len(self.dynamics.members))
        for i, member in enumerate(self.dynamics.members):
            mean_obs = float(member.mean_next(np.asarray([current]), np.asarray([action]))[0])
            mean_log = math.log1p(max(mean_obs, 0.0) / scale)
            sigma = max(float(member.residual_sigma), 0.03)
            residual = z - mean_log
            logliks[i] = -0.5 * (residual / sigma) ** 2 - math.log(sigma)
        return logliks

    def _simulate_public(self, previous, current, log_belief, depth, tree, timestep):
        if depth >= self.depth:
            return 0.0
        key = self._public_key(current, depth)
        node = tree.setdefault(
            key,
            _Node(0, np.zeros(self.num_actions, dtype=int), np.zeros(self.num_actions)),
        )
        unvisited = np.flatnonzero(node.action_visits == 0)
        if len(unvisited):
            action = int(self.rng.choice(unvisited))
        else:
            score = node.action_value + self.exploration * np.sqrt(
                math.log(node.visits + 1.0) / node.action_visits
            )
            action = int(np.argmax(score))
        # Bayes-adaptive state = (observation, categorical belief over ensemble
        # members).  Sample the successor model from the CURRENT node belief
        # (not a fixed root model), then update the belief with the realized
        # transition likelihood (Eq. 4, Chen et al. 2026) before recursing.
        belief = self._belief_from_logweights(log_belief)
        member_idx = int(self.rng.choice(len(self.dynamics.members), p=belief))
        member = self.dynamics.members[member_idx]
        following = float(
            member.sample_next(np.asarray([current]), np.asarray([action]), self.rng)[0]
        )
        log_belief_next = log_belief + self._public_member_loglik(current, action, following)
        log_belief_next -= np.max(log_belief_next)
        reward, _risk = self.public_context.surrogate.predict(
            np.asarray([previous]),
            np.asarray([current]),
            np.asarray([following]),
            np.asarray([action]),
            np.asarray([timestep]),
            np.asarray([self.public_context.pop_id]),
        )
        disagreement = float(
            self.dynamics.disagreement(
                np.asarray([current]), np.asarray([action])
            )[0]
        ) / max(current * current, 1.0)
        value = float(reward[0]) - self.planner_cfg.pessimism * disagreement
        value += self.planner_cfg.discount * self._simulate_public(
            current, following, log_belief_next, depth + 1, tree, timestep + 1
        )
        node.visits += 1
        node.action_visits[action] += 1
        count = node.action_visits[action]
        node.action_value[action] += (value - node.action_value[action]) / count
        return value

    def _simulate(self, state, regime, member_idx, depth, tree, entered, rho=0.0, kappa=0.0):
        if depth >= self.depth or state == 0.0:
            return 0.0
        key = self._key(state, depth, entered, rho, kappa)
        node = tree.setdefault(
            key,
            _Node(0, np.zeros(self.num_actions, dtype=int), np.zeros(self.num_actions)),
        )
        unvisited = np.flatnonzero(node.action_visits == 0)
        if len(unvisited):
            action = int(self.rng.choice(unvisited))
        else:
            ucb = node.action_value + self.exploration * np.sqrt(
                math.log(node.visits + 1.0) / node.action_visits
            )
            action = int(np.argmax(ucb))
        member = self.dynamics.members[member_idx]
        next_state = float(member.sample_next(
            np.asarray([state]), np.asarray([action]), self.rng,
            np.asarray([rho]), np.asarray([kappa]),
        )[0])
        next_rho, next_kappa, _next_K_eff = (
            advance_public_controls(self.env_cfg, action, rho, kappa)
            if control_fields_enabled(self.env_cfg)
            else (np.asarray(rho), np.asarray(kappa), np.asarray(self.env_cfg.K_base))
        )
        entry = (
            (not entered)
            and state > self.env_cfg.safety_threshold
            and next_state <= self.env_cfg.safety_threshold
        )
        # Benefit on the predicted TRUE next state s_{t+1} (spec E6); penalty mode-aware.
        penalty_indicator = bool(
            safety_penalty_indicator(self.env_cfg, state, next_state, entry)
        )
        reward = self.reward.state_reward(next_state, action, penalty_indicator)
        disagreement = float(
            self.dynamics.disagreement(
                np.asarray([state]), np.asarray([action]), np.asarray([rho]), np.asarray([kappa])
            )[0]
        ) / max(state * state, 1.0)
        value = reward - self.planner_cfg.pessimism * disagreement
        value += self.planner_cfg.discount * self._simulate(
            next_state, regime, member_idx, depth + 1, tree, entered or entry,
            float(next_rho), float(next_kappa),
        )
        node.visits += 1
        node.action_visits[action] += 1
        n = node.action_visits[action]
        node.action_value[action] += (value - node.action_value[action]) / n
        return value

    def act(self, belief: BeliefState, observation: float) -> int:
        if self.hidden:
            tree: dict[tuple, _Node] = {}
            indices = belief.sample_indices(self.simulations, self.rng)
            previous = (
                belief.contexts[:, 0]
                if belief.contexts.ndim > 1 else np.full(len(belief.states), observation)
            )
            # Each simulation starts from the deployment posterior over ensemble
            # members and updates that belief *inside* the tree (Eq. 4), rather
            # than fixing one root-sampled model for the whole rollout.
            root_log_belief = np.log(self.posterior + 1e-12)
            for index in indices:
                self._simulate_public(
                    float(previous[index]),
                    float(belief.states[index]),
                    root_log_belief.copy(),
                    0,
                    tree,
                    belief.timestep,
                )
            roots = [node for key, node in tree.items() if key[0] == 0]
            values = np.sum(
                [node.action_value * np.maximum(node.action_visits, 1) for node in roots],
                axis=0,
            )
            counts = np.sum(
                [np.maximum(node.action_visits, 1) for node in roots], axis=0
            )
            q = values / counts
            self.last_diagnostics = {"tree_nodes": float(len(tree)), "root_q": q.tolist()}
            return int(np.argmax(q))
        tree: dict[tuple, _Node] = {}
        indices = belief.sample_indices(self.simulations, self.rng)
        member_idx = self.rng.choice(
            len(self.dynamics.members), size=self.simulations, p=self.posterior
        )
        for i, m in zip(indices, member_idx):
            self._simulate(
                float(belief.states[i]), int(belief.regimes[i]), int(m), 0, tree,
                bool(belief.states[i] <= self.env_cfg.safety_threshold),
                0.0 if belief.rho is None else float(belief.rho),
                0.0 if belief.kappa is None else float(belief.kappa),
            )
        root_key = self._key(
            belief.mean_state(), 0, belief.mean_state() <= self.env_cfg.safety_threshold,
            0.0 if belief.rho is None else float(belief.rho),
            0.0 if belief.kappa is None else float(belief.kappa),
        )
        root = tree.get(root_key)
        if root is None:
            # Aggregation can put sampled roots in neighboring keys.
            roots = [node for key, node in tree.items() if key[0] == 0]
            values = np.sum([n.action_value * np.maximum(n.action_visits, 1) for n in roots], axis=0)
            counts = np.sum([np.maximum(n.action_visits, 1) for n in roots], axis=0)
            q = values / counts
        else:
            q = root.action_value
        self.last_diagnostics = {"tree_nodes": float(len(tree)), "root_q": q.tolist()}
        return int(np.argmax(q))

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        state = np.asarray([belief.mean_state()])
        predicted = np.asarray([
            member.mean_next(state, np.asarray([action]))[0]
            for member in self.dynamics.members
        ])
        scale = np.asarray([m.residual_sigma for m in self.dynamics.members]) + 0.05
        residual = np.log(max(result.observation, 1e-8)) - np.log(np.maximum(predicted, 1e-8))
        logp = np.log(self.posterior + 1e-300) - 0.5 * (residual / scale) ** 2 - np.log(scale)
        logp -= np.max(logp)
        self.posterior = np.exp(logp)
        self.posterior /= self.posterior.sum()
