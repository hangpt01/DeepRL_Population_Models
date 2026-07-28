"""Three-controller decision-relevance gate under matched observation noise."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .actions import resolve_actions
from .backend import get_active_backend
from .beliefs import (
    MechanisticProposal,
    ParticleFilter,
    real_action_lookups,
    real_next_states,
)
from .config import BenchmarkConfig, is_setpoint_cumulative
from .envs import ContinuousEcologyEnv, make_env
from .observation import LogNormalObservationModel
from .planning import ParticleMPC
from .reward import build_reward
from .types import BeliefState


class ExactEpisodeProposal:
    def __init__(self, env: ContinuousEcologyEnv, hidden: dict[str, float]):
        self.env = env
        self.hidden = hidden
        self.actions = resolve_actions(env.cfg)
        self.backend = get_active_backend()
        self._delta_r_dev, self._delta_K_dev, self._stock_dev = real_action_lookups(
            self.backend, self.actions
        )

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        states = np.asarray(states, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), states.shape)
        rho_values = np.zeros_like(states) if rho is None else np.broadcast_to(
            np.asarray(rho, dtype=np.float64), states.shape
        )
        kappa_values = np.zeros_like(states) if kappa is None else np.broadcast_to(
            np.asarray(kappa, dtype=np.float64), states.shape
        )
        if is_setpoint_cumulative(self.env.cfg):
            # Clairvoyant uses the true (fixed) episode params for every particle.
            C = np.full_like(states, self.hidden["C"])
            theta = np.full_like(states, self.hidden["theta"])
            out = real_next_states(
                self.env.cfg, self.backend, self._delta_r_dev, self._delta_K_dev,
                self._stock_dev, states, actions, C, theta, np.asarray(regimes),
                rho_values, kappa_values,
            )
        else:
            out = np.empty_like(states)
            for i, (state, aid) in enumerate(zip(states, actions)):
                out[i] = self.env.transition_value(
                    float(state), self.actions[int(aid)], self.hidden["r_base"],
                    self.hidden["C"], self.hidden["theta"], int(regimes[i]), 0.0,
                    float(rho_values[i]), float(kappa_values[i]),
                )
        next_regime = regimes.copy()
        if self.env.cfg.kind == "regime":
            switch = rng.random(len(states)) > self.env.cfg.regime_persistence
            next_regime[switch] = 1 - next_regime[switch]
        return out, next_regime


def _point_belief(
    state: float,
    observation: float,
    regime: int,
    particles: int,
    public_info: dict[str, object] | None = None,
):
    controls = {}
    if public_info and "rho" in public_info:
        controls = {
            "rho": float(public_info["rho"]),
            "kappa": float(public_info["kappa"]),
            "K_eff": float(public_info["K_eff"]),
        }
    return BeliefState(
        np.full(particles, state), np.zeros((particles, 3)),
        np.full(particles, regime, dtype=np.int8),
        np.full(particles, -np.log(particles)), observation,
        **controls,
    )


def _run_controller(
    cfg: BenchmarkConfig,
    seed: int,
    controller: str,
) -> tuple[float, int, list[int]]:
    env = make_env(cfg.environment)
    reset = env.reset(seed)
    hidden = reset.evaluator_info
    reward = build_reward(cfg.environment)
    planner = ParticleMPC(
        cfg.environment, cfg.planner, reward,
        LogNormalObservationModel(cfg.environment.observation_noise_sigma), seed + 101,
    )
    if controller == "belief_oracle":
        proposal = MechanisticProposal(cfg.environment, cfg.environment.kind)
        filt = ParticleFilter(cfg.environment, cfg.filter, proposal)
        belief = filt.reset(reset.observation, seed + 202)
    elif controller == "belief_ricker":
        r_mid = 0.5 * (cfg.environment.r_base_low + cfg.environment.r_base_high)
        proposal = MechanisticProposal(cfg.environment, "ricker", r_mid)
        filt = ParticleFilter(cfg.environment, cfg.filter, proposal)
        belief = filt.reset(reset.observation, seed + 202)
    elif controller == "clairvoyant":
        proposal = ExactEpisodeProposal(env, hidden)
        filt = None
        belief = _point_belief(
            hidden["state"], reset.observation, int(hidden["regime"]),
            cfg.filter.particles, reset.public_info,
        )
    else:
        raise ValueError(controller)
    total = 0.0
    gamma = 1.0
    collapsed = 0
    actions = []
    for _ in range(cfg.evaluation.horizon):
        action, _diag = planner.plan(belief, proposal, pessimism=0.0)
        result = env.step(action)
        actions.append(action)
        total += gamma * result.reward
        collapsed |= int(result.evaluator_info["entered_safety_region"])
        gamma *= cfg.evaluation.discount
        if result.done:
            break
        if controller == "clairvoyant":
            belief = _point_belief(
                result.evaluator_info["state"], result.observation,
                int(result.evaluator_info["regime"]), cfg.filter.particles,
                result.public_info,
            )
        else:
            belief = filt.update(belief, action, result.observation)
    return float(total), int(collapsed), actions


def gate_decision(
    environment: str,
    reward_gap: float,
    collapse_gap: float,
    reward_threshold: float = 1.0,
    collapse_threshold: float = 0.05,
) -> dict[str, object]:
    """Apply the predeclared hard/diagnostic gate rule to measured gaps."""

    hard = environment in {"allee", "regime"}
    active_reward_threshold = reward_threshold if hard else 0.5
    active_collapse_threshold = collapse_threshold if hard else 0.0
    return {
        "hard_gate": hard,
        "passed": bool(
            reward_gap >= active_reward_threshold
            and collapse_gap >= active_collapse_threshold
        ),
        "reward_threshold": active_reward_threshold,
        "collapse_threshold": active_collapse_threshold,
    }


def run_decision_gate(
    cfg: BenchmarkConfig,
    episodes: int = 20,
    reward_threshold: float = 1.0,
    collapse_threshold: float = 0.05,
) -> dict[str, object]:
    from .backend import resolve_backend_for_workload, set_active_backend

    backend = set_active_backend(
        resolve_backend_for_workload(cfg.compute, "gate", cfg.environment)
    )
    records = {name: [] for name in ("belief_oracle", "belief_ricker", "clairvoyant")}
    for episode in range(episodes):
        seed = cfg.seed + episode
        for name in records:
            ret, collapse, actions = _run_controller(cfg, seed, name)
            records[name].append({"return": ret, "collapse": collapse, "actions": actions})
    def mean(name, key):
        return float(np.mean([row[key] for row in records[name]]))
    reward_gap = mean("belief_oracle", "return") - mean("belief_ricker", "return")
    collapse_gap = mean("belief_ricker", "collapse") - mean("belief_oracle", "collapse")
    disagreement = []
    for oracle, ricker in zip(records["belief_oracle"], records["belief_ricker"]):
        n = min(len(oracle["actions"]), len(ricker["actions"]))
        if n:
            disagreement.extend(
                int(a != b) for a, b in zip(oracle["actions"][:n], ricker["actions"][:n])
            )
    decision = gate_decision(
        cfg.environment.kind,
        reward_gap,
        collapse_gap,
        reward_threshold,
        collapse_threshold,
    )
    return {
        "environment": cfg.environment.kind,
        "expose_rk": cfg.environment.expose_rk,
        "regime_label": (
            "hidden-demographics_structure-unknown"
            if cfg.environment.expose_rk == "hidden"
            else "public-demographics_parameter-known"
        ),
        "num_actions": cfg.environment.num_actions,
        "sigma_obs": cfg.environment.observation_noise_sigma,
        "episodes": episodes,
        **backend.to_dict(),
        **decision,
        "reward_gap": reward_gap,
        "collapse_gap": collapse_gap,
        "action_disagreement_rate": float(np.mean(disagreement)) if disagreement else 0.0,
        "belief_oracle_return": mean("belief_oracle", "return"),
        "belief_ricker_return": mean("belief_ricker", "return"),
        "clairvoyant_return": mean("clairvoyant", "return"),
        "belief_oracle_collapse": mean("belief_oracle", "collapse"),
        "belief_ricker_collapse": mean("belief_ricker", "collapse"),
        "records": records,
    }


def save_gate(result: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
