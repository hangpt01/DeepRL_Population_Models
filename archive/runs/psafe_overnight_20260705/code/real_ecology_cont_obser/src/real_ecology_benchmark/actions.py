"""Management action tables.

This is the adapted copy for the real-ecology setting.  The Tier-2/3 tables are
retained verbatim for reference and regression, but the live setting uses the
11-action, per-population, set-point table built from
``real_ecology_data`` via :func:`real_action_table` /
:func:`resolve_actions`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Iterable

from . import realdata

if TYPE_CHECKING:  # avoid a runtime import cycle (config imports actions indirectly)
    from .config import EnvironmentConfig


@dataclass(frozen=True)
class ActionSpec:
    id: int
    name: str
    delta_r: float
    delta_K: float
    harvest_fraction: float
    stocking_delta: float
    cost: float

    def validate(self) -> None:
        if self.id < 0:
            raise ValueError("action id must be non-negative")
        if not 0.0 <= self.harvest_fraction <= 1.0:
            raise ValueError("harvest_fraction must be in [0, 1]")
        if self.stocking_delta < 0.0:
            raise ValueError("stocking_delta must be non-negative")


TIER2_FIVE_ACTIONS = (
    ActionSpec(0, "Do Nothing", 0.00, 0.0, 0.00, 0.0, 0.00),
    ActionSpec(1, "Aggressive Harvest", -0.02, 0.0, 0.50, 0.0, -0.40),
    ActionSpec(2, "Predator / Disease Control", 0.02, 0.0, 0.00, 60.0, 0.20),
    ActionSpec(3, "Intensive Restoration", 0.02, 200.0, 0.00, 100.0, 0.30),
    ActionSpec(4, "Flagship Conservation", 0.04, 200.0, 0.00, 160.0, 0.60),
)

TIER2_TEN_ACTIONS = (
    ActionSpec(0, "Do Nothing", 0.00, 0.0, 0.00, 0.0, 0.00),
    ActionSpec(1, "Sustainable Harvest", -0.01, 0.0, 0.25, 0.0, -0.20),
    ActionSpec(2, "Aggressive Harvest", -0.02, 0.0, 0.50, 0.0, -0.40),
    ActionSpec(3, "Predator / Disease Control", 0.01, 0.0, 0.00, 40.0, 0.20),
    ActionSpec(4, "Breeding / Recruitment Support", 0.02, 0.0, 0.00, 60.0, 0.40),
    ActionSpec(5, "Moderate Restoration", 0.00, 100.0, 0.00, 80.0, 0.15),
    ActionSpec(6, "Intensive Restoration", 0.00, 200.0, 0.00, 100.0, 0.30),
    ActionSpec(7, "Integrated Conservation (light)", 0.01, 100.0, 0.00, 80.0, 0.35),
    ActionSpec(8, "Adaptive Conservation Trial", 0.01, 200.0, 0.00, 120.0, 0.50),
    ActionSpec(9, "Flagship Conservation Programme", 0.02, 200.0, 0.00, 160.0, 0.60),
)

TIER3_FIVE_ACTIONS = (
    ActionSpec(0, "Do Nothing", 0.00, 0.0, 0.00, 0.0, 0.00),
    ActionSpec(1, "Aggressive Harvest", -0.02, 0.0, 0.00, 0.0, -0.30),
    ActionSpec(2, "Predator / Disease Control", 0.01, 0.0, 0.00, 0.0, 0.12),
    ActionSpec(3, "Intensive Restoration", 0.00, 40.0, 0.00, 0.0, 0.16),
    ActionSpec(4, "Flagship Conservation", 0.02, 40.0, 0.00, 0.0, 0.32),
)

# Costs are C0-smoke calibrated from the tex starting values: the original
# restoration costs made restoration monotonically dominated after direct
# stocking was removed, violating the final plan's economics gate.
TIER3_TEN_ACTIONS = (
    ActionSpec(0, "Do Nothing", 0.00, 0.0, 0.00, 0.0, 0.00),
    ActionSpec(1, "Sustainable Harvest", -0.01, 0.0, 0.00, 0.0, -0.15),
    ActionSpec(2, "Aggressive Harvest", -0.02, 0.0, 0.00, 0.0, -0.30),
    ActionSpec(3, "Predator / Disease Control", 0.01, 0.0, 0.00, 0.0, 0.12),
    ActionSpec(4, "Breeding / Recruitment Support", 0.02, 0.0, 0.00, 0.0, 0.24),
    ActionSpec(5, "Moderate Restoration", 0.00, 20.0, 0.00, 0.0, 0.08),
    ActionSpec(6, "Intensive Restoration", 0.00, 40.0, 0.00, 0.0, 0.16),
    ActionSpec(7, "Integrated Conservation (light)", 0.01, 20.0, 0.00, 0.0, 0.20),
    ActionSpec(8, "Adaptive Conservation Trial", 0.01, 40.0, 0.00, 0.0, 0.24),
    ActionSpec(9, "Flagship Conservation Programme", 0.02, 40.0, 0.00, 0.0, 0.32),
)

# Backwards-compatible names for tests and older imports.
FIVE_ACTIONS = TIER2_FIVE_ACTIONS
TEN_ACTIONS = TIER2_TEN_ACTIONS


def validate_actions(actions: Iterable[ActionSpec]) -> tuple[ActionSpec, ...]:
    result = tuple(actions)
    for item in result:
        item.validate()
    if [a.id for a in result] != list(range(len(result))):
        raise ValueError("action ids must be contiguous and ordered")
    return result


def action_table(
    num_actions: int,
    control_mode: str = "tier2_one_step",
) -> tuple[ActionSpec, ...]:
    if control_mode == "tier2_one_step":
        if num_actions == 5:
            return TIER2_FIVE_ACTIONS
        if num_actions == 10:
            return TIER2_TEN_ACTIONS
    elif control_mode == "cumulative_capped":
        if num_actions == 5:
            return TIER3_FIVE_ACTIONS
        if num_actions == 10:
            return TIER3_TEN_ACTIONS
    else:
        raise ValueError(f"unknown control_mode: {control_mode}")
    raise ValueError("num_actions must be 5 or 10")


def action_table_hash(actions: Iterable[ActionSpec]) -> str:
    payload = json.dumps([asdict(a) for a in validate_actions(actions)], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- Real-ecology 11-action set (set-point r, cumulative K, translocation) ----
#
# delta_r  = the set-point intrinsic rate for the action (Ricker ln(lambda) for
#            ricker/allee/regime, LGM lambda-1 for theta), realised through the
#            cumulative-control plumbing with accumulator_decay_r=1 and r_base=0.
# delta_K  = cumulative capacity increment dK_step(population, action).
# stocking_delta = dN(population, action): only a10 (translocation) is non-zero.
# cost     = portal-grounded cost_step in [-0.10, 1.00] (harvest negative=revenue).
_REAL_ACTION_CACHE: dict[tuple[str, str, str], tuple[ActionSpec, ...]] = {}


def real_action_table(
    population: str,
    family: str,
    data_dir=realdata.DATA_DIR,
) -> tuple[ActionSpec, ...]:
    key = (str(data_dir), population, family)
    cached = _REAL_ACTION_CACHE.get(key)
    if cached is not None:
        return cached
    effects = realdata.effects_for(data_dir)
    names = realdata.actions_for(data_dir)
    specs = []
    for i in range(realdata.NUM_REAL_ACTIONS):
        action = f"a{i}"
        effect = effects[(population, action)]
        specs.append(
            ActionSpec(
                id=i,
                name=names[action].name_original,
                delta_r=float(effect.r_setpoint(family)),
                delta_K=float(effect.dK_step),
                harvest_fraction=0.0,
                stocking_delta=float(effect.dN),
                cost=float(effect.cost_step),
            )
        )
    result = validate_actions(specs)
    _REAL_ACTION_CACHE[key] = result
    return result


def resolve_actions(cfg: "EnvironmentConfig") -> tuple[ActionSpec, ...]:
    """Return the action table for a config, dispatching on ``control_mode``.

    The real setting needs the population-and-family-specific table; all other
    modes fall back to the global Tier-2/3 tables.  Every vendored call site that
    previously used ``action_table(cfg.num_actions, cfg.control_mode)`` is routed
    through here so set-point r and per-population dK/dN flow consistently.
    """

    if cfg.control_mode == "real_setpoint":
        return real_action_table(
            cfg.population, cfg.kind, cfg.data_dir or realdata.DATA_DIR
        )
    return action_table(cfg.num_actions, cfg.control_mode)
