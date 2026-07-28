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
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


@dataclass
class _Node:
    visits: int = 0
    action_visits: np.ndarray | None = None
    action_value: np.ndarray | None = None


class BAMCTSPolicy(BasePolicy):
    name = "bamcts"

    def __init__(self, *args, simulations: int = 128, depth: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = simulations
        self.depth = depth
        self.exploration = 1.25

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("BA-MCTS requires cached shared beliefs")
        size = max(self.model_cfg.ensemble_size, 5)
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
