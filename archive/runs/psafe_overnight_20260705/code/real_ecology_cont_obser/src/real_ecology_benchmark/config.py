"""Configuration dataclasses and YAML loading (real-ecology / Tier-4).

This is the adapted copy of the Tier-2/3 ``config.py``.  ``EnvironmentConfig``
keeps its name (every vendored module imports it) but now describes a single
*real population* under one map *family*: per-population ``K_base/K_max/N0``,
data-derived set-point caps ``r_min/r_max``, a per-population safety floor, and
the new ``control_mode="real_setpoint"``.  Set-point growth is realised by the
existing cumulative-control plumbing with ``accumulator_decay_r=1.0`` and
``r_base in [0,0]`` (so ``r_eff = clip(rho, r_min, r_max)`` where ``rho`` is the
action's set-point), while capacity stays cumulative.  Nothing in the Tier-3
package is modified; this file lives only inside ``real_ecology_cont_obser/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import realdata


CONTROL_MODES = {"tier2_one_step", "cumulative_capped", "real_setpoint"}
SAFETY_PENALTY_MODES = {"crossing", "occupancy"}

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


@dataclass
class EnvironmentConfig:
    kind: str = "ricker"
    num_actions: int = 11
    control_mode: str = "real_setpoint"
    horizon: int = 50
    # Per-population scale (filled from species.csv via real_environment()).
    population: str = "Amur tiger"
    N0: float = 200.0
    K_base: float = 250.0
    # Structural-stress parameters.  For the real setting the Allee threshold and
    # the regime thresholds are scaled to the population's K_base (the Tier-3
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
    # legacy Tier-3 values {observed, belief_expected} are still accepted (dead).
    reward_mode: str = "safe"
    # Fractions used to refresh the structural-stress + safety absolutes.
    allee_C_frac_low: float = 0.18
    allee_C_frac_high: float = 0.30
    regime_threshold_low_frac: float = 0.18
    regime_threshold_high_frac: float = 0.36
    data_dir: str = ""

    def validate(self) -> None:
        if self.kind not in {"ricker", "allee", "theta", "regime"}:
            raise ValueError(f"unknown environment kind: {self.kind}")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"unknown control_mode: {self.control_mode}")
        if self.control_mode == "real_setpoint":
            if self.num_actions != realdata.NUM_REAL_ACTIONS:
                raise ValueError("real_setpoint requires num_actions == 11")
        elif self.num_actions not in {5, 10}:
            raise ValueError("num_actions must be 5 or 10")
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
        control_mode="real_setpoint",
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


@dataclass
class PlannerConfig:
    horizon: int = 5
    sequences: int = 96
    particles: int = 32
    discount: float = 0.95
    pessimism: float = 0.5


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

    def validate(self) -> None:
        self.environment.validate()
        self.compute.validate()
        self.training.validate()
        if self.dataset.transitions <= 0 or self.dataset.episode_length <= 0:
            raise ValueError("dataset sizes must be positive")
        if self.filter.particles < 8:
            raise ValueError("particle count must be at least 8")
        if self.model.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")

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
    if "population" in env_raw and env_raw.get("control_mode", "real_setpoint") == "real_setpoint":
        population = env_raw.pop("population")
        family = env_raw.pop("kind", "ricker")
        env_raw.pop("control_mode", None)
        data_dir = env_raw.pop("data_dir", realdata.DATA_DIR)
        environment = real_environment(
            population, family, data_dir=data_dir, **env_raw
        )
    else:
        environment = _construct(EnvironmentConfig, env_raw)
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


def environment_with_kind_defaults(cfg: EnvironmentConfig, kind: str) -> EnvironmentConfig:
    """Switch the map family for a real-population cell, refreshing data caps.

    Re-reads the per-population set-point caps for the new family (Ricker vs LGM
    columns) and keeps every other (population-specific) field.  This mirrors the
    Tier-3 helper but is population-aware.
    """

    if cfg.control_mode != "real_setpoint":
        values = dict(cfg.__dict__)
        values["kind"] = kind
        result = EnvironmentConfig(**values)
        result.validate()
        return result
    return real_environment_like(cfg, cfg.population, kind)
