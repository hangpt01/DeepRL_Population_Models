"""Seeded finite-horizon, context-conditioned point-based value iteration."""

from __future__ import annotations

import time

import numpy as np

from ..config import FaithfulPlannerConfig
from ..faithful_pomdp import CandidateBelief, CandidatePOMDP
from .base import PlannerProvenance


PBVI_VERSION = "context_pbvi_v1"


class PointBasedPlanner:
    """Finite-horizon backups over a seeded graph of reachable belief points.

    Capacity, public survey history, and time remain explicit context at every point.
    The graph is rebuilt from the current public history; no stationary reward shortcut
    or QMDP reduction is used.
    """

    def __init__(
        self,
        model: CandidatePOMDP,
        config: FaithfulPlannerConfig,
        discount: float,
        seed: int,
    ):
        if config.name != "pbvi":
            raise ValueError("PointBasedPlanner requires faithful planner name 'pbvi'")
        self.model = model
        self.config = config
        self.discount = float(discount)
        self.seed = int(seed)
        self.invocation_count = 0
        self.elapsed_seconds = 0.0
        self.last_action_values = np.zeros(model.context.num_actions, dtype=np.float64)

    def _condition(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> CandidateBelief:
        updated, _ = self.model.update(belief, action, observation)
        return updated

    @staticmethod
    def _distance(left: CandidateBelief, right: CandidateBelief) -> float:
        probability = float(np.sum(np.abs(left.probabilities - right.probabilities)))
        capacity = abs(left.capacity - right.capacity) / max(left.capacity, right.capacity, 1e-9)
        current = abs(left.current_observation - right.current_observation) / max(
            left.current_observation, right.current_observation, 1.0
        )
        return probability + 0.2 * capacity + 0.1 * current

    def _reachable_graph(self, root: CandidateBelief) -> list[list[CandidateBelief]]:
        layers: list[list[CandidateBelief]] = [[root]]
        max_points = self.config.belief_points
        for depth in range(self.config.horizon):
            previous = layers[-1]
            following: list[CandidateBelief] = []
            cursor = 0
            while len(following) < max_points and previous:
                parent = previous[cursor % len(previous)]
                action = (cursor // max(len(previous), 1) + depth) % self.model.context.num_actions
                predicted = self.model.predict(parent, action)
                observations, _ = self.model.representative_observations(
                    predicted, self.config.observation_branches
                )
                branch = (cursor * 104729 + self.seed + depth) % len(observations)
                following.append(self._condition(parent, action, float(observations[branch])))
                cursor += 1
            layers.append(following or [previous[0]])
        return layers

    def action_values(self, belief: CandidateBelief) -> np.ndarray:
        started = time.perf_counter()
        layers = self._reachable_graph(belief)
        next_values = np.zeros(len(layers[-1]), dtype=np.float64)
        root_q = np.zeros(self.model.context.num_actions, dtype=np.float64)
        for depth in range(self.config.horizon - 1, -1, -1):
            points = layers[depth]
            following = layers[depth + 1]
            values = np.zeros(len(points), dtype=np.float64)
            for point_index, point in enumerate(points):
                q_values = np.zeros(self.model.context.num_actions, dtype=np.float64)
                for action in range(self.model.context.num_actions):
                    predicted = self.model.predict(point, action)
                    immediate = self.model.expected_public_reward(point, action, predicted)
                    observations, weights = self.model.representative_observations(
                        predicted, self.config.observation_branches
                    )
                    continuation = 0.0
                    for observation, weight in zip(observations, weights):
                        child = self._condition(point, action, float(observation))
                        nearest = min(
                            range(len(following)),
                            key=lambda index: self._distance(child, following[index]),
                        )
                        continuation += float(weight) * float(next_values[nearest])
                    q_values[action] = immediate + self.discount * continuation
                values[point_index] = float(np.max(q_values))
                if depth == 0 and point_index == 0:
                    root_q = q_values
            next_values = values
        self.invocation_count += 1
        self.elapsed_seconds += time.perf_counter() - started
        self.last_action_values = root_q.copy()
        return root_q

    def provenance(self) -> PlannerProvenance:
        return PlannerProvenance(
            name="pbvi",
            implementation_version=PBVI_VERSION,
            seed=self.seed,
            model_hash=self.model.model_hash(),
            invocation_count=self.invocation_count,
            external_invocation=False,
            elapsed_seconds=self.elapsed_seconds,
        )
