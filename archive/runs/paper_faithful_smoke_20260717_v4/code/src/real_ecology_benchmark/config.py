"""Configuration dataclasses and YAML loading.

``EnvironmentConfig`` describes one ecological cell under one dynamics family.
Real and dummy cells use set-point growth with cumulative carrying-capacity
control: the public ``rho`` control stores the action's intrinsic-rate set point
and ``kappa`` accumulates capacity changes.  Synthetic continuous-state cells
keep the older one-step and cumulative-control modes for regression/reference
experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import yaml

from . import dummydata, realdata


SETPOINT_CUMULATIVE = "setpoint_cumulative"
LEGACY_REAL_SETPOINT = "real_setpoint"
CONTROL_MODE_ALIASES = {LEGACY_REAL_SETPOINT: SETPOINT_CUMULATIVE}
CONTROL_MODES = {"tier2_one_step", "cumulative_capped", SETPOINT_CUMULATIVE}
DATA_MODES = {"real", "dummy", "synthetic"}
SAFETY_PENALTY_MODES = {"crossing", "occupancy"}
EXPOSE_RK_MODES = {"hidden", "full"}

# Depletion-aware safety-tier convention used by default_safety_fraction().  The
# TeX spec gives ranges rather than exact cutoffs; keep the chosen cutoffs named
# so calibration can revise them deliberately instead of hunting magic numbers.
SMALL_K_THRESHOLD = 75.0
MID_K_THRESHOLD = 120.0
ENDANGERED_N_THRESHOLD = 50.0
SEVERE_DEPLETION_RATIO = 0.20
MODERATE_DEPLETION_RATIO = 0.50
HEALTHY_SAFETY_FRACTION = 0.10
MID_SAFETY_FRACTION = 0.20
HIGH_RISK_SAFETY_FRACTION = 0.25
DEFAULT_MVP_THRESHOLD = 50.0


def normalize_control_mode(mode: str) -> str:
    """Return the canonical control-mode name, accepting legacy aliases."""

    return CONTROL_MODE_ALIASES.get(str(mode), str(mode))


def is_setpoint_cumulative(cfg_or_mode: Any) -> bool:
    """True when a config/mode uses set-point ``r`` and cumulative ``K``."""

    mode = getattr(cfg_or_mode, "control_mode", cfg_or_mode)
    return normalize_control_mode(mode) == SETPOINT_CUMULATIVE


@dataclass
class EnvironmentConfig:
    kind: str = "ricker"
    num_actions: int = 11
    control_mode: str = SETPOINT_CUMULATIVE
    data_mode: str = "real"
    expose_rk: str = "full"
    horizon: int = 50
    # Per-population scale (filled from species.csv via real_environment()).
    population: str = "Amur tiger"
    N0: float = 200.0
    K_base: float = 250.0
    # Structural-stress parameters.  For the real setting the Allee threshold and
    # the regime thresholds are scaled to the population's K_base (the synthetic
    # absolute values 90/150/180 are nonsensical for K in 31..325); these absolute
    # fields are refreshed from the fractions in real_environment().
    C_low: float = 45.0
    C_high: float = 75.0
    theta_low: float = 3.0
    theta_high: float = 6.0
    regime_persistence: float = 0.90
    regime_threshold_low: float = 45.0
    regime_threshold_high: float = 90.0
    regime_weak_multiplier: float = 0.65
    process_noise_sigma: float = 0.0
    observation_noise_sigma: float = 0.2
    # Reward / safety (per population).  The default real_environment() path
    # computes a depletion-aware safety fraction from the population table; an
    # explicit safety_fraction override pins this value.
    safety_threshold: float = 25.0
    safety_fraction: float = 0.1
    safety_fraction_auto: bool = True
    # "safe" mode defaults to occupancy so populations that start below s_safe
    # are penalized until they recover.  "crossing" preserves the old event-only
    # convention for explicit ablations.
    safety_penalty_mode: str = "occupancy"
    mvp_threshold: float = DEFAULT_MVP_THRESHOLD
    # Calibrated so one collapse outweighs a default healthy episode at the real
    # reward scale: 0.5 * sum_{t<50} 0.95^t ~= 9.23.
    collapse_penalty: float = 10.0
    alpha: float = 1.0
    K_ref: float = 250.0
    # Real cells start deterministically at N0 (E1); collector start coverage is
    # supplied separately via collect_dataset(start_low_probability, start_log_sigma),
    # so evaluation/gate reset semantics stay at s0 = N0.
    initial_log_sigma: float = 0.0
    low_start_probability: float = 0.0
    privileged_behavior: bool = True
    # Set-point caps (data-derived per population/family).
    r_base_low: float = 0.0
    r_base_high: float = 0.0
    r_min: float = -0.4753
    r_max: float = 0.0707
    theta_stability_margin: float = 1.6
    theta_base_stability_margin: float = 1.2
    K_min: float = 250.0
    K_max: float = 500.0
    accumulator_decay_r: float = 1.0
    accumulator_decay_K: float = 0.0
    # Real-ecology reward setting (E6'): "safe" -> collapse penalty P>0 (headline,
    # collapse-aware); "yield" -> P=0 (baseline-style, matches PLUS/MOOR).  The
    # legacy synthetic values {observed, belief_expected} are still accepted.
    reward_mode: str = "safe"
    # Fractions used to refresh the structural-stress + safety absolutes.
    allee_C_frac_low: float = 0.18
    allee_C_frac_high: float = 0.30
    regime_threshold_low_frac: float = 0.18
    regime_threshold_high_frac: float = 0.36
    data_dir: str = ""

    def __post_init__(self) -> None:
        self.control_mode = normalize_control_mode(self.control_mode)

    def validate(self) -> None:
        if self.kind not in {"ricker", "allee", "theta", "regime"}:
            raise ValueError(f"unknown environment kind: {self.kind}")
        self.control_mode = normalize_control_mode(self.control_mode)
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(
                f"unknown control_mode: {self.control_mode}; "
                f"valid modes are {sorted(CONTROL_MODES)} "
                f"({LEGACY_REAL_SETPOINT!r} is accepted as a legacy alias for "
                f"{SETPOINT_CUMULATIVE!r})"
            )
        if self.data_mode not in DATA_MODES:
            raise ValueError(f"unknown data_mode: {self.data_mode}")
        if self.expose_rk not in EXPOSE_RK_MODES:
            raise ValueError(
                f"unknown expose_rk: {self.expose_rk}; "
                f"valid modes are {sorted(EXPOSE_RK_MODES)}"
            )
        if is_setpoint_cumulative(self):
            if self.data_mode == "real" and self.num_actions != realdata.NUM_REAL_ACTIONS:
                raise ValueError("setpoint_cumulative real data requires num_actions == 11")
            if (
                self.data_mode == "dummy"
                and self.num_actions not in dummydata.SUPPORTED_ACTION_COUNTS
            ):
                raise ValueError(
                    "dummy setpoint_cumulative requires num_actions in "
                    f"{sorted(dummydata.SUPPORTED_ACTION_COUNTS)}"
                )
            if self.data_mode == "synthetic":
                raise ValueError(
                    "synthetic data uses one-step or cumulative-control synthetic modes"
                )
        elif self.num_actions not in {5, 10}:
            raise ValueError("num_actions must be 5 or 10")
        if self.data_mode == "dummy" and not is_setpoint_cumulative(self):
            raise ValueError(
                f"dummy data currently requires control_mode={SETPOINT_CUMULATIVE!r}"
            )
        if self.data_mode == "synthetic" and is_setpoint_cumulative(self):
            raise ValueError(
                f"synthetic data cannot use control_mode={SETPOINT_CUMULATIVE!r}"
            )
        if self.horizon <= 0 or self.K_base <= 0 or self.K_ref <= 0:
            raise ValueError("horizon, K_base, and K_ref must be positive")
        if self.r_base_low > self.r_base_high:
            raise ValueError("invalid r_base range")
        if self.observation_noise_sigma < 0 or self.process_noise_sigma < 0:
            raise ValueError("noise scales must be non-negative")
        if not 0 <= self.low_start_probability <= 1:
            raise ValueError("low_start_probability must be in [0,1]")
        if self.r_min > self.r_max:
            raise ValueError("invalid r cap range")
        if self.K_min <= 0 or self.K_min > self.K_max:
            raise ValueError("invalid K cap range")
        if not 0.0 < self.safety_fraction < 1.0:
            raise ValueError("safety_fraction must lie in (0, 1)")
        if self.safety_threshold < 0.0:
            raise ValueError("safety_threshold must be non-negative")
        if self.safety_penalty_mode not in SAFETY_PENALTY_MODES:
            raise ValueError(f"unknown safety_penalty_mode: {self.safety_penalty_mode}")
        if self.mvp_threshold < 0.0:
            raise ValueError("mvp_threshold must be non-negative")
        if not 0.0 <= self.accumulator_decay_r <= 1.0:
            raise ValueError("accumulator_decay_r must be in [0,1]")
        if not 0.0 <= self.accumulator_decay_K <= 1.0:
            raise ValueError("accumulator_decay_K must be in [0,1]")
        if self.reward_mode not in {"yield", "safe", "observed", "belief_expected"}:
            raise ValueError(f"unknown reward_mode: {self.reward_mode}")


def uses_real_data(cfg: EnvironmentConfig) -> bool:
    return cfg.data_mode == "real"


def uses_dummy_data(cfg: EnvironmentConfig) -> bool:
    return cfg.data_mode == "dummy"


def uses_synthetic_data(cfg: EnvironmentConfig) -> bool:
    return cfg.data_mode == "synthetic"


def hides_rk(cfg: EnvironmentConfig) -> bool:
    return cfg.data_mode == "real" and cfg.expose_rk == "hidden"


def opaque_population_id(population: str) -> str:
    """Stable categorical token with no embedded species/table name."""

    value = hashlib.sha256(
        f"hidden-rk-population-v1:{population.strip().casefold()}".encode("utf-8")
    ).hexdigest()[:16]
    return f"pop_{value}"


@dataclass(frozen=True)
class MethodContext:
    """The complete method-visible configuration for hidden real-ecology runs."""

    num_actions: int
    action_costs: tuple[float, ...]
    observation_noise_sigma: float
    horizon: int
    observation_scale: float
    pop_id: str
    expose_rk: str = "hidden"
    regime_label: str = "hidden-demographics_structure-unknown"
    reward_mode: str = "safe"
    surrogate: Any | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.expose_rk != "hidden":
            raise ValueError("MethodContext is only valid for expose_rk='hidden'")
        if self.num_actions <= 0 or len(self.action_costs) != self.num_actions:
            raise ValueError("action_costs must cover every public action")
        if self.observation_scale <= 0 or self.observation_noise_sigma < 0:
            raise ValueError("invalid public observation scale/noise")
        if not self.pop_id.startswith("pop_"):
            raise ValueError("pop_id must be an opaque categorical token")


def default_safety_fraction(pop: realdata.RealPopulation) -> float:
    """Depletion-aware default c_safe for the real setting.

    The latest spec rejects the old absolute floor and says a flat 0.10*K rule
    is too permissive for small, depleted, or endangered-count populations.  This
    deterministic rule keeps large healthy populations at 0.10, raises depleted
    or mid-scale cells to 0.20, and uses 0.25 for the smallest/depleted cases.
    """

    depletion_ratio = pop.N0 / pop.K_base
    fraction = HEALTHY_SAFETY_FRACTION
    if (
        pop.K_base <= SMALL_K_THRESHOLD
        or pop.N0 <= ENDANGERED_N_THRESHOLD
        or depletion_ratio < SEVERE_DEPLETION_RATIO
    ):
        fraction = max(fraction, HIGH_RISK_SAFETY_FRACTION)
    elif pop.K_base <= MID_K_THRESHOLD or depletion_ratio < MODERATE_DEPLETION_RATIO:
        fraction = max(fraction, MID_SAFETY_FRACTION)
    return fraction


def real_environment(
    population: str,
    family: str = "ricker",
    *,
    data_dir: str | Path = realdata.DATA_DIR,
    **overrides: Any,
) -> EnvironmentConfig:
    """Build a real-ecology :class:`EnvironmentConfig` for one population/family.

    Fills ``K_base/K_max/N0``, the data-derived set-point caps (Ricker columns
    for ricker/allee/regime, LGM columns for theta), the depletion-aware
    per-population safety floor ``s_safe = c_safe(pop) * K_base``, and the
    K-scaled structural stress thresholds.  Set-point growth uses
    ``accumulator_decay_r=1.0`` and action-table set-points.
    """

    data_mode = overrides.pop("data_mode", "real")
    if data_mode != "real":
        raise ValueError("real_environment requires data_mode='real'")
    control_mode = normalize_control_mode(overrides.pop("control_mode", SETPOINT_CUMULATIVE))
    if not is_setpoint_cumulative(control_mode):
        raise ValueError(f"real_environment requires control_mode={SETPOINT_CUMULATIVE!r}")
    pops = realdata.pops_for(data_dir)
    if population not in pops:
        raise KeyError(f"unknown population {population!r}; have {list(pops)}")
    pop = pops[population]
    _r_base, r_min, r_max = pop.caps(family)
    explicit_safety_fraction = "safety_fraction" in overrides
    safety_fraction = float(
        overrides.pop(
            "safety_fraction",
            default_safety_fraction(pop),
        )
    )
    safety_fraction_auto = bool(
        overrides.pop("safety_fraction_auto", not explicit_safety_fraction)
    )
    if explicit_safety_fraction:
        safety_fraction_auto = False
    c_low_frac = float(overrides.pop("allee_C_frac_low", 0.18))
    c_high_frac = float(overrides.pop("allee_C_frac_high", 0.30))
    reg_low_frac = float(overrides.pop("regime_threshold_low_frac", 0.18))
    reg_high_frac = float(overrides.pop("regime_threshold_high_frac", 0.36))
    values: dict[str, Any] = dict(
        kind=family,
        num_actions=realdata.NUM_REAL_ACTIONS,
        control_mode=SETPOINT_CUMULATIVE,
        data_mode="real",
        expose_rk=str(overrides.pop("expose_rk", "hidden")),
        population=population,
        N0=pop.N0,
        K_base=pop.K_base,
        K_ref=pop.K_base,
        K_min=pop.K_base,
        K_max=pop.K_max,
        safety_fraction=safety_fraction,
        safety_fraction_auto=safety_fraction_auto,
        safety_threshold=safety_fraction * pop.K_base,
        mvp_threshold=DEFAULT_MVP_THRESHOLD,
        r_base_low=0.0,
        r_base_high=0.0,
        r_min=r_min,
        r_max=r_max,
        accumulator_decay_r=1.0,
        accumulator_decay_K=0.0,
        allee_C_frac_low=c_low_frac,
        allee_C_frac_high=c_high_frac,
        C_low=c_low_frac * pop.K_base,
        C_high=c_high_frac * pop.K_base,
        regime_threshold_low_frac=reg_low_frac,
        regime_threshold_high_frac=reg_high_frac,
        regime_threshold_low=reg_low_frac * pop.K_base,
        regime_threshold_high=reg_high_frac * pop.K_base,
        data_dir=str(data_dir),
    )
    values.update(overrides)
    cfg = EnvironmentConfig(**values)
    cfg.validate()
    return cfg


def dummy_environment(
    population: str = dummydata.DEFAULT_POPULATION,
    family: str = "ricker",
    **overrides: Any,
) -> EnvironmentConfig:
    """Build a synthetic set-point-r / cumulative-K dummy environment.

    The dummy setting intentionally shares ``control_mode="setpoint_cumulative"``
    with real data so all set-point/cumulative semantics take the same code path.
    It differs only in the population/action/cap source.
    """

    data_mode = overrides.pop("data_mode", "dummy")
    if data_mode != "dummy":
        raise ValueError("dummy_environment requires data_mode='dummy'")
    control_mode = normalize_control_mode(
        overrides.pop("control_mode", SETPOINT_CUMULATIVE)
    )
    if not is_setpoint_cumulative(control_mode):
        raise ValueError(f"dummy_environment requires control_mode={SETPOINT_CUMULATIVE!r}")
    overrides.pop("data_dir", None)
    num_actions = int(overrides.pop("num_actions", dummydata.NUM_DUMMY_ACTIONS))
    dummydata.actions_for(num_actions)
    pops = dummydata.pops_for()
    if population not in pops:
        raise KeyError(f"unknown dummy population {population!r}; have {list(pops)}")
    pop = pops[population]
    _r_base, r_min, r_max = pop.caps(family)
    explicit_safety_fraction = "safety_fraction" in overrides
    safety_fraction = float(overrides.pop("safety_fraction", HEALTHY_SAFETY_FRACTION))
    safety_fraction_auto = bool(
        overrides.pop("safety_fraction_auto", not explicit_safety_fraction)
    )
    if explicit_safety_fraction:
        safety_fraction_auto = False
    c_low_frac = float(overrides.pop("allee_C_frac_low", 0.18))
    c_high_frac = float(overrides.pop("allee_C_frac_high", 0.30))
    reg_low_frac = float(overrides.pop("regime_threshold_low_frac", 0.18))
    reg_high_frac = float(overrides.pop("regime_threshold_high_frac", 0.36))
    values: dict[str, Any] = dict(
        kind=family,
        num_actions=num_actions,
        control_mode=SETPOINT_CUMULATIVE,
        data_mode="dummy",
        expose_rk=str(overrides.pop("expose_rk", "full")),
        population=population,
        N0=pop.N0,
        K_base=pop.K_base,
        K_ref=pop.K_base,
        K_min=pop.K_base,
        K_max=pop.K_max,
        safety_fraction=safety_fraction,
        safety_fraction_auto=safety_fraction_auto,
        safety_threshold=safety_fraction * pop.K_base,
        mvp_threshold=DEFAULT_MVP_THRESHOLD,
        r_base_low=0.0,
        r_base_high=0.0,
        r_min=r_min,
        r_max=r_max,
        accumulator_decay_r=1.0,
        accumulator_decay_K=0.0,
        allee_C_frac_low=c_low_frac,
        allee_C_frac_high=c_high_frac,
        C_low=c_low_frac * pop.K_base,
        C_high=c_high_frac * pop.K_base,
        regime_threshold_low_frac=reg_low_frac,
        regime_threshold_high_frac=reg_high_frac,
        regime_threshold_low=reg_low_frac * pop.K_base,
        regime_threshold_high=reg_high_frac * pop.K_base,
        data_dir="",
    )
    values.update(overrides)
    cfg = EnvironmentConfig(**values)
    cfg.validate()
    return cfg


def synthetic_environment(
    kind: str = "allee",
    **overrides: Any,
) -> EnvironmentConfig:
    """Build the continuous-state synthetic environment defaults."""

    values: dict[str, Any] = dict(
        kind=kind,
        num_actions=5,
        control_mode="tier2_one_step",
        data_mode="synthetic",
        expose_rk=str(overrides.pop("expose_rk", "full")),
        horizon=50,
        population="synthetic",
        N0=500.0,
        K_base=500.0,
        C_low=90.0,
        C_high=150.0,
        theta_low=3.0,
        theta_high=6.0,
        regime_persistence=0.90,
        regime_threshold_low=90.0,
        regime_threshold_high=180.0,
        regime_weak_multiplier=0.65,
        process_noise_sigma=0.0,
        observation_noise_sigma=0.2,
        safety_threshold=50.0,
        safety_fraction=0.1,
        safety_fraction_auto=False,
        mvp_threshold=DEFAULT_MVP_THRESHOLD,
        safety_penalty_mode="crossing",
        collapse_penalty=20.0,
        alpha=1.0,
        K_ref=500.0,
        initial_log_sigma=0.7,
        low_start_probability=0.2,
        privileged_behavior=True,
        r_base_low=0.12,
        r_base_high=0.30,
        r_min=-0.15,
        r_max=0.55,
        theta_stability_margin=1.6,
        theta_base_stability_margin=1.2,
        K_min=500.0,
        K_max=1500.0,
        accumulator_decay_r=0.0,
        accumulator_decay_K=0.0,
        reward_mode="observed",
        data_dir="",
    )
    values.update(overrides)
    cfg = EnvironmentConfig(**values)
    cfg.validate()
    return cfg


@dataclass
class DatasetConfig:
    transitions: int = 2000
    episode_length: int = 25
    output: str = "outputs/dataset_public.npz"
    private_output: str = "outputs/dataset_private.npz"


@dataclass
class FilterConfig:
    particles: int = 256
    proposal: str = "learned"
    ess_fraction: float = 0.5
    proposal_sigma: float = 0.18
    context_rejuvenation: float = 0.04


@dataclass
class ModelConfig:
    ensemble_size: int = 5
    ridge: float = 1e-3
    native_state_bins: int = 51
    native_fit_grid: int = 31
    native_vi_iterations: int = 250
    native_vi_tolerance: float = 1e-7


@dataclass
class PlannerConfig:
    horizon: int = 5
    sequences: int = 96
    particles: int = 32
    discount: float = 0.95
    pessimism: float = 0.5
    # Per-method search budgets.  These were hardcoded literals inside the policies,
    # which made a "we tuned the general baselines" sweep a silent no-op for BA-MCTS
    # and OGSRL.  The defaults reproduce those literals exactly, so surfacing them
    # here changes no result.  Note OGSRL's rollout horizon is 6, NOT planner.horizon's
    # 5 -- which is why these are separate fields rather than a repointing at `horizon`;
    # repointing would silently alter the completed motivation run's semantics.
    bamcts_depth: int = 5
    bamcts_simulations: int = 128
    ogsrl_rollout_horizon: int = 6


@dataclass
class FaithfulFitConfig:
    """Registered public-history fitting budget for paper-faithful methods."""

    optimizer: str = "torch_lbfgs"
    starts: int = 8
    iterations: int = 100
    mc_paths: int = 16
    history_fraction: float = 0.8
    learning_rate: float = 0.5
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    shrinkage: float = 1e-3
    group_penalty: float = 2e-3
    complementarity_penalty: float = 2e-3
    sparse_action_rows: int = 25

    def validate(self) -> None:
        if self.optimizer != "torch_lbfgs":
            raise ValueError("faithful fitting requires optimizer='torch_lbfgs'")
        if self.starts < 1 or self.iterations < 1 or self.mc_paths < 1:
            raise ValueError("faithful starts, iterations, and mc_paths must be positive")
        if not 0.0 < self.history_fraction < 1.0:
            raise ValueError("faithful history_fraction must be in (0,1)")
        if self.learning_rate <= 0.0:
            raise ValueError("faithful learning_rate must be positive")
        if self.tolerance_grad <= 0.0 or self.tolerance_change <= 0.0:
            raise ValueError("faithful optimizer tolerances must be positive")
        if min(self.shrinkage, self.group_penalty, self.complementarity_penalty) < 0.0:
            raise ValueError("faithful regularization coefficients must be non-negative")
        if self.sparse_action_rows < 1:
            raise ValueError("faithful sparse_action_rows must be positive")


@dataclass
class FaithfulModelConfig:
    """Mechanistic candidate-bank choices fixed before real-data fitting."""

    equation_version: str = "faithful_ecology_v1"
    forms: tuple[str, ...] = ("ricker", "allee", "theta", "regime")
    candidates_per_form: int = 4
    prior: str = "uniform"
    prior_temperature: float = 1.0
    minimum_candidate_distance: float = 1e-4
    process_scale_upper: float = 1.25
    abundance_upper: float = 12.0
    capacity_upper: float = 16.0
    action_growth_upper: float = 2.0
    action_mortality_upper: float = 1.5
    action_capacity_upper: float = 1.0
    action_stocking_upper: float = 3.0

    def validate(self) -> None:
        allowed = {"ricker", "allee", "theta", "regime"}
        if tuple(dict.fromkeys(self.forms)) != self.forms or set(self.forms) != allowed:
            raise ValueError("faithful forms must contain each registered form exactly once")
        if self.candidates_per_form not in {1, 2, 4, 8}:
            raise ValueError("faithful candidates_per_form must be one of 1,2,4,8")
        if self.prior not in {"uniform", "public_history_likelihood"}:
            raise ValueError("unknown faithful candidate prior")
        if self.prior_temperature <= 0.0 or self.minimum_candidate_distance < 0.0:
            raise ValueError("invalid faithful prior/diversity setting")
        bounds = (
            self.process_scale_upper,
            self.abundance_upper,
            self.capacity_upper,
            self.action_growth_upper,
            self.action_mortality_upper,
            self.action_capacity_upper,
            self.action_stocking_upper,
        )
        if min(bounds) <= 0.0:
            raise ValueError("faithful parameter upper bounds must be positive")


@dataclass
class FaithfulPlannerConfig:
    """Finite-horizon point-based planning and optional external solver settings."""

    name: str = "pbvi"
    state_bins: int = 41
    capacity_bins: int = 9
    observation_bins: int = 41
    transition_samples: int = 256
    observation_samples: int = 256
    belief_points: int = 32
    observation_branches: int = 7
    horizon: int = 5
    despot_path: str = ""
    timeout_seconds: int = 60

    def validate(self) -> None:
        if self.name not in {"pbvi", "despot"}:
            raise ValueError("faithful planner must be 'pbvi' or 'despot'")
        if min(self.state_bins, self.observation_bins) < 5:
            raise ValueError("faithful state/observation grids require at least five bins")
        if self.capacity_bins < 2:
            raise ValueError("faithful capacity grid requires at least two bins")
        if min(self.transition_samples, self.observation_samples, self.belief_points) < 1:
            raise ValueError("faithful discretization/planning budgets must be positive")
        if self.observation_branches < 2 or self.horizon < 1:
            raise ValueError("faithful branch count/horizon is invalid")
        if self.timeout_seconds < 1:
            raise ValueError("faithful planner timeout must be positive")


@dataclass
class FaithfulConfig:
    model: FaithfulModelConfig = field(default_factory=FaithfulModelConfig)
    fit: FaithfulFitConfig = field(default_factory=FaithfulFitConfig)
    planner: FaithfulPlannerConfig = field(default_factory=FaithfulPlannerConfig)

    def validate(self) -> None:
        self.model.validate()
        self.fit.validate()
        self.planner.validate()


@dataclass
class EvaluationConfig:
    seeds: list[int] = field(default_factory=lambda: [7001, 7051, 7101, 7151, 7201])
    episodes_per_seed: int = 4
    horizon: int = 50
    discount: float = 0.95
    output_dir: str = "outputs/evaluation"


@dataclass
class ComputeConfig:
    """Compute backend selection (see backend.py).

    ``backend="numpy"`` is the CPU reference and default; ``"cupy"`` requests a
    CUDA GPU.  ``strict=True`` fails if the requested backend is unavailable;
    ``strict=False`` allows a recorded CuPy->NumPy fallback.
    """

    backend: str = "numpy"
    device: int = 0
    strict: bool = True

    def validate(self) -> None:
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError(f"unknown compute backend: {self.backend}")
        if self.device < 0:
            raise ValueError("compute device id must be non-negative")


@dataclass
class TrainingConfig:
    """Local/optional-W&B training diagnostics.

    The monitor is local-first: every method row saves a CSV/JSON history and,
    when matplotlib is importable, a PNG plot under the row output directory.
    ``wandb`` is opt-in so benchmark jobs do not depend on external service state.
    """

    enabled: bool = True
    holdout_fraction: float = 0.2
    split_seed_offset: int = 50_000
    plot: bool = True
    wandb: bool = False
    wandb_project: str = "real-ecology-cont-obser"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None

    def validate(self) -> None:
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("training.holdout_fraction must be in [0,1)")
        if self.split_seed_offset < 0:
            raise ValueError("training.split_seed_offset must be non-negative")


@dataclass
class BenchmarkConfig:
    seed: int = 116
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    faithful: FaithfulConfig = field(default_factory=FaithfulConfig)

    def validate(self) -> None:
        self.environment.validate()
        self.compute.validate()
        self.training.validate()
        self.faithful.validate()
        if self.dataset.transitions <= 0 or self.dataset.episode_length <= 0:
            raise ValueError("dataset sizes must be positive")
        if self.filter.particles < 8:
            raise ValueError("particle count must be at least 8")
        if self.model.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        if self.model.native_state_bins < 12:
            raise ValueError("model.native_state_bins must be at least 12")
        if self.model.native_fit_grid < 2:
            raise ValueError("model.native_fit_grid must be at least 2")
        if self.model.native_vi_iterations < 1:
            raise ValueError("model.native_vi_iterations must be positive")
        if self.model.native_vi_tolerance <= 0:
            raise ValueError("model.native_vi_tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    env_raw = dict(raw.get("environment") or {})
    # Allow a compact ``population`` + ``kind`` spec that expands to the full
    # per-population real config, with any explicit overrides layered on top.
    control_mode = normalize_control_mode(
        env_raw.get("control_mode", SETPOINT_CUMULATIVE)
    )
    data_mode = env_raw.get(
        "data_mode",
        "real" if is_setpoint_cumulative(control_mode) else "synthetic",
    )
    if (
        data_mode == "real"
        and "population" in env_raw
        and is_setpoint_cumulative(control_mode)
    ):
        population = env_raw.pop("population")
        family = env_raw.pop("kind", "ricker")
        env_raw.pop("control_mode", None)
        env_raw.pop("data_mode", None)
        data_dir = env_raw.pop("data_dir", realdata.DATA_DIR)
        environment = real_environment(
            population, family, data_dir=data_dir, **env_raw
        )
    elif data_mode == "dummy" and is_setpoint_cumulative(control_mode):
        population = env_raw.pop("population", dummydata.DEFAULT_POPULATION)
        family = env_raw.pop("kind", "ricker")
        env_raw.pop("control_mode", None)
        env_raw.pop("data_mode", None)
        env_raw.pop("data_dir", None)
        environment = dummy_environment(population, family, **env_raw)
    else:
        env_raw["control_mode"] = control_mode
        env_raw.setdefault("data_mode", data_mode)
        environment = _construct(EnvironmentConfig, env_raw)
    faithful_raw = dict(raw.get("faithful") or {})
    faithful_model_raw = dict(faithful_raw.get("model") or {})
    if "forms" in faithful_model_raw:
        faithful_model_raw["forms"] = tuple(faithful_model_raw["forms"])
    faithful = FaithfulConfig(
        model=_construct(FaithfulModelConfig, faithful_model_raw),
        fit=_construct(FaithfulFitConfig, faithful_raw.get("fit")),
        planner=_construct(FaithfulPlannerConfig, faithful_raw.get("planner")),
    )
    cfg = BenchmarkConfig(
        seed=int(raw.get("seed", 116)),
        environment=environment,
        dataset=_construct(DatasetConfig, raw.get("dataset")),
        filter=_construct(FilterConfig, raw.get("filter")),
        model=_construct(ModelConfig, raw.get("model")),
        planner=_construct(PlannerConfig, raw.get("planner")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        compute=_construct(ComputeConfig, raw.get("compute")),
        training=_construct(TrainingConfig, raw.get("training")),
        faithful=faithful,
    )
    cfg.validate()
    return cfg


# Fields that are NOT derived from (population, family) and so must be carried
# over when re-building a real cell for a different population/family.
_CARRYOVER_FIELDS = (
    "horizon", "observation_noise_sigma", "process_noise_sigma",
    "collapse_penalty", "alpha", "initial_log_sigma", "low_start_probability",
    "privileged_behavior", "regime_persistence", "regime_weak_multiplier",
    "theta_low", "theta_high", "theta_stability_margin",
    "theta_base_stability_margin", "reward_mode", "safety_fraction",
    "safety_penalty_mode", "mvp_threshold", "allee_C_frac_low", "allee_C_frac_high",
    "regime_threshold_low_frac", "regime_threshold_high_frac",
    "expose_rk",
)


def real_environment_like(
    cfg: EnvironmentConfig, population: str, family: str
) -> EnvironmentConfig:
    """Rebuild a real cell for ``(population, family)`` carrying ``cfg``'s settings.

    Per-population scale (K/N0/caps/s_safe) is refreshed from the data; everything
    the user set on the source cell (observation noise, horizon, penalty, the
    structural-stress fractions, ...) is preserved so a CLI ``--population`` or
    ``--environment`` switch does not silently revert those overrides.
    """

    overrides = {}
    for key in _CARRYOVER_FIELDS:
        if key == "safety_fraction" and getattr(cfg, "safety_fraction_auto", False):
            continue
        overrides[key] = getattr(cfg, key)
    return real_environment(
        population, family, data_dir=cfg.data_dir or realdata.DATA_DIR, **overrides
    )


def dummy_environment_like(
    cfg: EnvironmentConfig, population: str, family: str
) -> EnvironmentConfig:
    """Rebuild a dummy cell carrying user-set non-source settings."""

    overrides = {}
    for key in _CARRYOVER_FIELDS:
        if key == "safety_fraction" and getattr(cfg, "safety_fraction_auto", False):
            continue
        overrides[key] = getattr(cfg, key)
    overrides["num_actions"] = cfg.num_actions
    return dummy_environment(population, family, **overrides)


def environment_with_kind_defaults(cfg: EnvironmentConfig, kind: str) -> EnvironmentConfig:
    """Switch the map family for a real-population cell, refreshing data caps.

    Re-reads the per-population set-point caps for the new family (Ricker vs LGM
    columns) and keeps every other (population-specific) field.
    """

    if not is_setpoint_cumulative(cfg):
        values = dict(cfg.__dict__)
        values["kind"] = kind
        if cfg.data_mode == "synthetic":
            if cfg.control_mode == "cumulative_capped":
                if kind == "theta":
                    values.update(r_base_low=0.12, r_base_high=0.40)
                else:
                    values.update(r_base_low=0.12, r_base_high=0.30)
            else:
                if kind == "theta":
                    values.update(r_base_low=0.18, r_base_high=0.40)
                elif kind == "ricker":
                    values.update(r_base_low=0.95, r_base_high=1.00)
                else:
                    values.update(r_base_low=0.12, r_base_high=0.30)
        result = EnvironmentConfig(**values)
        result.validate()
        return result
    if uses_dummy_data(cfg):
        return dummy_environment_like(cfg, cfg.population, kind)
    return real_environment_like(cfg, cfg.population, kind)
