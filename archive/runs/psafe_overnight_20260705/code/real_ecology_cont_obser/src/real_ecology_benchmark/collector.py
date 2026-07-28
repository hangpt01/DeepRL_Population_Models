"""Privileged-state behavior mixture and trajectory collection.

The behavior mixture is parameterized by an explicit, serializable
``CollectorProfile`` instead of hidden constants.  A profile may depend on
(environment family, action count) but must be fixed across observation-noise
levels, and the full profile is recorded in public dataset metadata so the data
generating process is auditable and reproducible.  The biological priors,
equations, action tables, reward/collapse semantics, ``safety_threshold``,
``initial_log_sigma``, ``low_start_probability`` and privileged true-state
behavior are NOT collector settings and are never changed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .actions import action_table, action_table_hash, resolve_actions
from .config import EnvironmentConfig
from .controls import control_fields_enabled
from .dataset import PrivateTrajectoryData, TrajectoryDataset
from .envs import ContinuousEcologyEnv, real_initial_state


@dataclass(frozen=True)
class CollectorProfile:
    """Serializable behavior-mixture calibration profile.

    Only behavior-mixture intensities are tunable.  ``component_probabilities``
    is the mixture over (random, harvest_probe, rescue_dwell, threshold_probe);
    the remaining fields scale how destructive/supportive each component is.
    Reducing harvesting/random probing and increasing rescue/support lowers the
    healthy-start incident collapse rate without touching biology.
    """

    name: str
    component_probabilities: tuple[float, float, float, float]
    harvest_probe_prob: float = 0.85
    rescue_harvest_prob: float = 0.45
    rescue_donothing_prob: float = 0.60

    def validate(self) -> None:
        probs = np.asarray(self.component_probabilities, dtype=np.float64)
        if probs.shape != (4,):
            raise ValueError("component_probabilities must have 4 entries")
        if np.any(probs < 0):
            raise ValueError("component probabilities must be non-negative")
        if not np.isclose(float(probs.sum()), 1.0, atol=1e-9):
            raise ValueError("component probabilities must sum to 1")
        for value in (self.harvest_probe_prob, self.rescue_harvest_prob,
                      self.rescue_donothing_prob):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("intensity probabilities must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["component_probabilities"] = [float(p) for p in self.component_probabilities]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectorProfile":
        payload = dict(data)
        payload["component_probabilities"] = tuple(
            float(p) for p in payload["component_probabilities"]
        )
        return cls(**payload)


# The default profile reproduces the originally deployed behavior mixture
# exactly; it is used for every cell without an explicit override (notably the
# already-completed allee_10a and regime_10a families) so their data is
# unchanged.
DEFAULT_PROFILE = CollectorProfile(
    name="default",
    component_probabilities=(0.25, 0.30, 0.25, 0.20),
    harvest_probe_prob=0.85,
    rescue_harvest_prob=0.45,
    rescue_donothing_prob=0.60,
)

# Per-(family, action-count) overrides for cells whose default-profile data is
# outside the [0.15, 0.24] healthy-start collapse band.  Tuned with
# scripts/calibrate_profiles.py and verified at the production seed (116) and
# budget (75k); each lands ~0.20 and is observation-noise independent.
#
# allee_5a / regime_5a: rescue-tilt (cut abundance-blind probing that crashes
#   fragile low-but-healthy starts) -> 0.21 / 0.20.
# theta_5a / theta_10a: harvest_probe-dominant regulation to a safe mid-band;
#   stocking is counterproductive because the high-r/high-theta map overshoots
#   and crashes when pushed above K -> 0.21 / 0.21.
PROFILES: dict[tuple[str, int], CollectorProfile] = {
    ("allee", 5): CollectorProfile(
        name="allee_5a_rescue_tilt",
        component_probabilities=(0.187, 0.285, 0.376, 0.152),
        harvest_probe_prob=0.775,
        rescue_harvest_prob=0.375,
        rescue_donothing_prob=0.645,
    ),
    ("regime", 5): CollectorProfile(
        name="regime_5a_rescue_tilt",
        component_probabilities=(0.187, 0.285, 0.376, 0.152),
        harvest_probe_prob=0.775,
        rescue_harvest_prob=0.375,
        rescue_donothing_prob=0.645,
    ),
    ("theta", 5): CollectorProfile(
        name="theta_5a_regulate",
        component_probabilities=(0.08, 0.70, 0.18, 0.04),
        harvest_probe_prob=0.85,
        rescue_harvest_prob=0.45,
        rescue_donothing_prob=0.40,
    ),
    ("theta", 10): CollectorProfile(
        name="theta_10a_regulate",
        component_probabilities=(0.08, 0.70, 0.18, 0.04),
        harvest_probe_prob=0.85,
        rescue_harvest_prob=0.45,
        rescue_donothing_prob=0.40,
    ),
}


# Real-ecology behaviour mixture: harvest-tilted so the *exploitable* populations
# reach the [0.15, 0.24] healthy-start collapse band.  Populations whose measured
# heaviest-exploitation rate (a2) keeps lambda near 1 (e.g. crab-eating fox,
# jaguar, lynx) are structurally robust and stay below the band regardless of the
# mixture -- that is a property of their real demographics, reported per
# population rather than forced.
REAL_DEFAULT_PROFILE = CollectorProfile(
    name="real_harvest_tilt",
    component_probabilities=(0.20, 0.45, 0.10, 0.25),
    harvest_probe_prob=0.95,
    rescue_harvest_prob=0.65,
    rescue_donothing_prob=0.30,
)


def collector_profile_for(kind: str, num_actions: int) -> CollectorProfile:
    if int(num_actions) == 11:  # real-ecology 11-action menu
        return PROFILES.get((kind, 11), REAL_DEFAULT_PROFILE)
    return PROFILES.get((kind, int(num_actions)), DEFAULT_PROFILE)


class MixedDangerZonePolicy:
    components = ("random", "harvest_probe", "rescue_dwell", "threshold_probe")

    def __init__(
        self,
        num_actions: int,
        profile: CollectorProfile = DEFAULT_PROFILE,
        privileged: bool = True,
        abundance_scale: float = 500.0,
    ):
        profile.validate()
        self.num_actions = num_actions
        self.profile = profile
        self.privileged = privileged
        self.probabilities = np.asarray(profile.component_probabilities, dtype=np.float64)
        self.component = "random"
        # Decision thresholds are a fraction of carrying capacity; the legacy
        # K_base=500 reproduces the original absolute cutoffs (250/350/150) so
        # the Tier-2/3 mixtures are unchanged, while real populations scale to
        # their own K_base.
        self.scale = float(abundance_scale)
        self.high = 0.5 * self.scale
        self.very_high = 0.7 * self.scale
        self.support_cut = 0.3 * self.scale

    def reset(self, rng: np.random.Generator) -> None:
        self.component = str(rng.choice(self.components, p=self.probabilities))

    def _harvest(self, rng: np.random.Generator) -> int:
        if self.num_actions == 5:
            return 1
        if self.num_actions == 11:
            # Real menu: a1 sustainable / a2 aggressive harvest.
            return int(rng.choice([1, 2], p=[0.4, 0.6]))
        return int(rng.choice([1, 2], p=[0.25, 0.75]))

    def _support(self, abundance: float, rng: np.random.Generator) -> int:
        if self.num_actions == 11:
            # Real menu: rate/capacity/conservation a3..a9 plus translocation a10.
            if abundance <= self.support_cut:
                return int(rng.choice(
                    [3, 4, 7, 8, 9, 10], p=[.18, .18, .16, .16, .16, .16]
                ))
            return int(rng.choice([0, 3, 5, 6], p=[.40, .25, .20, .15]))
        if self.num_actions == 5:
            if abundance <= self.support_cut:
                return int(rng.choice([2, 3, 4], p=[0.45, 0.40, 0.15]))
            return int(rng.choice([0, 2, 3], p=[0.35, 0.40, 0.25]))
        if abundance <= self.support_cut:
            return int(rng.choice([3, 4, 5, 6, 8, 9], p=[.20, .20, .20, .20, .10, .10]))
        return int(rng.choice([0, 3, 4, 5, 6], p=[.30, .20, .20, .15, .15]))

    def act(
        self,
        true_state: float,
        observation: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> int:
        abundance = true_state if self.privileged else observation
        profile = self.profile
        if self.component == "random":
            return int(rng.integers(0, self.num_actions))
        if self.component == "harvest_probe":
            if abundance > self.high and rng.random() < profile.harvest_probe_prob:
                return self._harvest(rng)
            return self._support(abundance, rng)
        if self.component == "rescue_dwell":
            if abundance <= self.high:
                return self._support(abundance, rng)
            if abundance > self.very_high and rng.random() < profile.rescue_harvest_prob:
                return self._harvest(rng)
            return 0 if rng.random() < profile.rescue_donothing_prob else int(
                rng.integers(0, self.num_actions)
            )
        if timestep < 8:
            return 0 if timestep % 2 == 0 else self._harvest(rng)
        if abundance <= self.high:
            return self._support(abundance, rng)
        return int(rng.integers(0, self.num_actions))


def collect_dataset(
    env: ContinuousEcologyEnv,
    target_transitions: int,
    episode_length: int,
    seed: int,
    privileged_behavior: bool | None = None,
    profile: CollectorProfile | None = None,
    start_low_probability: float = 0.45,
    start_log_sigma: float = 0.15,
) -> tuple[TrajectoryDataset, PrivateTrajectoryData]:
    """Collect complete episodes; final count may exceed target by one episode.

    For the real setting the collector samples its own episode start spread
    (``start_low_probability``/``start_log_sigma``) and injects it via
    ``state_override``.  The environment's own reset stays at the spec's
    deterministic ``s0 = N0`` so evaluation/gate semantics are unaffected.
    """
    if target_transitions <= 0 or episode_length <= 0:
        raise ValueError("target_transitions and episode_length must be positive")
    privileged = env.cfg.privileged_behavior if privileged_behavior is None else privileged_behavior
    if profile is None:
        profile = collector_profile_for(env.cfg.kind, env.num_actions)
    behavior = MixedDangerZonePolicy(
        env.num_actions, profile, privileged, abundance_scale=env.cfg.K_base
    )
    rng = np.random.default_rng(seed)
    public: dict[str, list[Any]] = {
        key: []
        for key in (
            "observations", "actions", "rewards", "next_observations",
            "dones", "episode_id", "timestep",
        )
    }
    if control_fields_enabled(env.cfg):
        for key in ("rho", "kappa", "K_eff", "next_rho", "next_kappa", "next_K_eff"):
            public[key] = []
    private: dict[str, list[Any]] = {
        key: []
        for key in (
            "states", "next_states", "r_base", "C", "theta", "regime",
            "next_regime", "entry", "reward_true", "initially_unsafe",
        )
    }
    if control_fields_enabled(env.cfg):
        private["r_eff_true"] = []
    is_real = env.cfg.control_mode == "real_setpoint"
    episode = 0
    while len(public["actions"]) < target_transitions:
        episode_seed = int(rng.integers(0, 2**31 - 1))
        if is_real:
            start = real_initial_state(
                rng, env.cfg.N0, env.cfg.safety_threshold,
                start_low_probability, start_log_sigma,
            )
            reset = env.reset(episode_seed, state_override=start)
        else:
            reset = env.reset(episode_seed)
        behavior.reset(rng)
        observation = reset.observation
        public_info = reset.public_info
        initial_state = float(reset.evaluator_info["state"])
        initially_unsafe = initial_state <= env.cfg.safety_threshold
        for t in range(episode_length):
            state = env.state
            action = behavior.act(state, observation, t, rng)
            result = env.step(action)
            forced_end = t == episode_length - 1 and not result.done
            done = bool(result.done or forced_end)
            public["observations"].append(observation)
            public["actions"].append(action)
            public["rewards"].append(result.reward)
            public["next_observations"].append(result.observation)
            public["dones"].append(done)
            public["episode_id"].append(episode)
            public["timestep"].append(t)
            if control_fields_enabled(env.cfg):
                public["rho"].append(public_info["rho"])
                public["kappa"].append(public_info["kappa"])
                public["K_eff"].append(public_info["K_eff"])
                public["next_rho"].append(result.public_info["rho"])
                public["next_kappa"].append(result.public_info["kappa"])
                public["next_K_eff"].append(result.public_info["K_eff"])
            info = result.evaluator_info
            private["states"].append(info["state_previous"])
            private["next_states"].append(info["state"])
            private["r_base"].append(info["r_base"])
            private["C"].append(info["C"])
            private["theta"].append(info["theta"])
            private["regime"].append(info["regime_previous"])
            private["next_regime"].append(info["regime"])
            private["entry"].append(info["entered_safety_region"])
            private["reward_true"].append(info["reward_true"])
            private["initially_unsafe"].append(initially_unsafe)
            if control_fields_enabled(env.cfg):
                private["r_eff_true"].append(info["r_eff_true"])
            observation = result.observation
            public_info = result.public_info
            if done:
                break
        episode += 1
    metadata = {
        "schema_version": 3 if control_fields_enabled(env.cfg) else 2,
        "seed": int(seed),
        "target_transitions": int(target_transitions),
        "actual_transitions": len(public["actions"]),
        "episodes": int(episode),
        "episode_length": int(episode_length),
        "environment": env.config_dict(),
        "action_table_hash": action_table_hash(env.actions),
        "privileged_behavior": bool(privileged),
        "collector_profile": profile.to_dict(),
        "collector_start_low_probability": float(start_low_probability) if is_real else None,
        "collector_start_log_sigma": float(start_log_sigma) if is_real else None,
        "truth_derived_public_signal": "logged reward only; policy/filter transition excludes reward",
    }
    dataset = TrajectoryDataset(
        observations=np.asarray(public["observations"], dtype=np.float64),
        actions=np.asarray(public["actions"], dtype=np.int16),
        rewards=np.asarray(public["rewards"], dtype=np.float64),
        next_observations=np.asarray(public["next_observations"], dtype=np.float64),
        dones=np.asarray(public["dones"], dtype=bool),
        episode_id=np.asarray(public["episode_id"], dtype=np.int32),
        timestep=np.asarray(public["timestep"], dtype=np.int16),
        metadata=metadata,
        **({
            "rho": np.asarray(public["rho"], dtype=np.float64),
            "kappa": np.asarray(public["kappa"], dtype=np.float64),
            "K_eff": np.asarray(public["K_eff"], dtype=np.float64),
            "next_rho": np.asarray(public["next_rho"], dtype=np.float64),
            "next_kappa": np.asarray(public["next_kappa"], dtype=np.float64),
            "next_K_eff": np.asarray(public["next_K_eff"], dtype=np.float64),
        } if control_fields_enabled(env.cfg) else {}),
    )
    truth = PrivateTrajectoryData(
        states=np.asarray(private["states"], dtype=np.float64),
        next_states=np.asarray(private["next_states"], dtype=np.float64),
        r_base=np.asarray(private["r_base"], dtype=np.float64),
        C=np.asarray(private["C"], dtype=np.float64),
        theta=np.asarray(private["theta"], dtype=np.float64),
        regime=np.asarray(private["regime"], dtype=np.int8),
        next_regime=np.asarray(private["next_regime"], dtype=np.int8),
        entry=np.asarray(private["entry"], dtype=bool),
        reward_true=np.asarray(private["reward_true"], dtype=np.float64),
        initially_unsafe=np.asarray(private["initially_unsafe"], dtype=bool),
        metadata={
            "schema_version": 3 if control_fields_enabled(env.cfg) else 2,
            "public_metadata": metadata,
        },
        **({
            "r_eff_true": np.asarray(private["r_eff_true"], dtype=np.float64),
        } if control_fields_enabled(env.cfg) else {}),
    )
    dataset.validate()
    truth.validate(dataset)
    return dataset, truth


def calibration_summary(
    dataset: TrajectoryDataset,
    truth: PrivateTrajectoryData,
    safety_threshold: float,
    mvp_threshold: float | None = None,
) -> dict[str, Any]:
    truth.validate(dataset)
    episodes = np.unique(dataset.episode_id)
    incident = []
    initial_unsafe = []
    harvest_driven_delayed = []
    env_meta = dataset.metadata.get("environment", {})
    control_mode = env_meta.get("control_mode", "tier2_one_step")
    if mvp_threshold is None:
        mvp_threshold = float(env_meta.get("mvp_threshold", 50.0))
    if control_mode == "real_setpoint":
        specs = resolve_actions(EnvironmentConfig(**env_meta))
    else:
        specs = action_table(int(env_meta.get("num_actions", 5)), control_mode)
    # Harvest = exploitation (revenue) or direct removal.  Under set-point r the
    # set-point can be positive even for a harvest action (growing populations),
    # so identify harvest by negative cost (revenue) rather than by delta_r sign.
    harvest_actions = {
        spec.id for spec in specs if spec.cost < 0 or spec.harvest_fraction > 0
    }
    for episode in episodes:
        idx = np.flatnonzero(dataset.episode_id == episode)
        initial_unsafe.append(bool(truth.initially_unsafe[idx[0]]))
        entry_positions = np.flatnonzero(truth.entry[idx])
        had_incident = bool(len(entry_positions))
        incident.append(had_incident)
        if had_incident:
            first_entry_local = int(entry_positions[0])
            actions_before_entry = dataset.actions[idx[: first_entry_local + 1]]
            first_harvest = next(
                (j for j, action in enumerate(actions_before_entry)
                 if int(action) in harvest_actions),
                None,
            )
            harvest_driven_delayed.append(
                bool(first_harvest is not None and first_entry_local > first_harvest)
            )
        else:
            harvest_driven_delayed.append(False)
    healthy = np.logical_not(initial_unsafe)
    incident_arr = np.asarray(incident, dtype=bool)
    incident_rate = float(
        np.mean(incident_arr[healthy]) if np.any(healthy) else np.nan
    )
    return {
        "transitions": len(dataset),
        "episodes": len(episodes),
        "initially_unsafe_rate": float(np.mean(initial_unsafe)),
        "incident_collapse_rate_healthy_starts": incident_rate,
        "target_collapse_band": [0.15, 0.24],
        "collapse_band_pass": bool(0.15 <= incident_rate <= 0.24)
        if np.isfinite(incident_rate) else False,
        "safety_threshold": float(safety_threshold),
        "safety_penalty_mode": env_meta.get("safety_penalty_mode", "crossing"),
        "unsafe_occupancy": float(np.mean(truth.states <= safety_threshold)),
        "mvp_threshold": float(mvp_threshold),
        "mvp_occupancy": float(np.mean(truth.states <= mvp_threshold)),
        "observation_quantiles": np.quantile(
            dataset.observations, [0.01, 0.1, 0.5, 0.9, 0.99]
        ).tolist(),
        "action_frequency": {
            str(int(a)): float(np.mean(dataset.actions == a))
            for a in np.unique(dataset.actions)
        },
        "collector_profile": dataset.metadata.get("collector_profile"),
        "entry_count": int(np.sum(truth.entry)),
        "harvest_driven_delayed_collapse_count": int(np.sum(harvest_driven_delayed)),
        "harvest_driven_delayed_collapse_rate": float(np.mean(harvest_driven_delayed))
        if len(harvest_driven_delayed) else 0.0,
        "non_finite_count": int(
            np.sum(~np.isfinite(dataset.observations))
            + np.sum(~np.isfinite(dataset.next_observations))
        ),
    }
