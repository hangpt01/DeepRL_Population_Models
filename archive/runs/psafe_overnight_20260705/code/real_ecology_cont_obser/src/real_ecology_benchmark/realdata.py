"""Loader for the real-ecology parameter tables (``real_ecology_data``).

This is the single source of truth for the Tier-4 / real-ecology setting.  It
reads the shipped CSVs and exposes three lookups used to build the environment:

* ``ACTIONS``  -- the species-independent 11-action menu (``actions.csv``):
  ``channel, lambda_source, K_multiplier, dN_fraction, cost_step``.
* ``POPS``     -- per-population rows (``species.csv``): ``N0, K_base, K_max`` and
  the set-point clip bounds ``r_{base,min,max}_{ricker,lgm}``.
* ``EFFECTS``  -- the precomputed ``(population, action)`` table
  (``action_effects_long.csv``): ``lambda, r_setpoint_ricker/lgm, dK_step, dN,
  cost_step`` and the clip bounds.

The growth-rate conversions are ``r_Ricker = ln(lambda)`` (Ricker / Allee /
regime families) and ``r_LGM = lambda - 1`` (theta-logistic).  The env reads
``EFFECTS`` directly; the relational core is kept for provenance/regeneration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# The authoritative table lives once at ``discrete_action_cont_obser/real_ecology_data``.
# Keep the runtime pointed there so the benchmark cannot drift from a redundant
# package-local copy.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT.parent / "real_ecology_data"

# Families whose set-point uses the Ricker conversion r = ln(lambda); the
# theta-logistic family uses the LGM conversion r = lambda - 1 instead.
RICKER_FAMILIES = ("ricker", "allee", "regime")
THETA_FAMILY = "theta"

NUM_REAL_ACTIONS = 11
NUM_REAL_POPULATIONS = 9
RATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RealAction:
    action: str          # "a0" .. "a10"
    index: int           # 0 .. 10
    name_original: str
    channel: str         # none | rate | capacity | rate+capacity | state
    lambda_source: str   # which lambda_id the set-point reuses
    K_multiplier: float
    dN_fraction: float
    cost_step: float


@dataclass(frozen=True)
class RealPopulation:
    common_name: str
    scientific_name: str
    N0: float
    K_base: float
    K_max: float
    r_base_ricker: float
    r_min_ricker: float
    r_max_ricker: float
    r_base_lgm: float
    r_min_lgm: float
    r_max_lgm: float

    def caps(self, family: str) -> tuple[float, float, float]:
        """Return ``(r_base, r_min, r_max)`` for the given map family."""
        if family == THETA_FAMILY:
            return self.r_base_lgm, self.r_min_lgm, self.r_max_lgm
        return self.r_base_ricker, self.r_min_ricker, self.r_max_ricker


@dataclass(frozen=True)
class RealEffect:
    common_name: str
    action: str
    index: int
    channel: str
    lambda_value: float
    r_setpoint_ricker: float
    r_setpoint_lgm: float
    dK_step: float
    dN: float
    cost_step: float

    def r_setpoint(self, family: str) -> float:
        return self.r_setpoint_lgm if family == THETA_FAMILY else self.r_setpoint_ricker


def _action_index(action: str) -> int:
    return int(action[1:])  # "a0" -> 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required real-ecology table is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required real-ecology table is empty: {path}")
    return rows


def _require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    present = set(rows[0])
    missing = required - present
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def _maybe_float(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    if value == "" or value == "-":
        return default
    return float(value)


def load_actions(data_dir: Path | str = DATA_DIR) -> dict[str, RealAction]:
    path = Path(data_dir) / "actions.csv"
    rows = _read_csv(path)
    _require_columns(
        path,
        rows,
        {
            "action", "name_original", "channel", "lambda_source",
            "K_multiplier", "dN_fraction", "cost_step",
        },
    )
    actions: dict[str, RealAction] = {}
    for row in rows:
        actions[row["action"]] = RealAction(
            action=row["action"],
            index=_action_index(row["action"]),
            name_original=row["name_original"],
            channel=row["channel"],
            lambda_source=row["lambda_source"],
            K_multiplier=_maybe_float(row["K_multiplier"], 1.0),
            dN_fraction=_maybe_float(row["dN_fraction"], 0.0),
            cost_step=_maybe_float(row["cost_step"], 0.0),
        )
    expected = {f"a{i}" for i in range(NUM_REAL_ACTIONS)}
    if set(actions) != expected:
        raise ValueError(
            f"actions.csv must contain exactly {NUM_REAL_ACTIONS} actions "
            f"{sorted(expected)}; found {sorted(actions)}"
        )
    return actions


def load_pops(data_dir: Path | str = DATA_DIR) -> dict[str, RealPopulation]:
    path = Path(data_dir) / "species.csv"
    rows = _read_csv(path)
    _require_columns(
        path,
        rows,
        {
            "common_name", "scientific_name", "N0", "K_base", "K_max",
            "r_base_ricker", "r_min_ricker", "r_max_ricker",
            "r_base_lgm", "r_min_lgm", "r_max_lgm",
        },
    )
    pops: dict[str, RealPopulation] = {}
    for row in rows:
        pops[row["common_name"]] = RealPopulation(
            common_name=row["common_name"],
            scientific_name=row["scientific_name"],
            N0=_maybe_float(row["N0"]),
            K_base=_maybe_float(row["K_base"]),
            K_max=_maybe_float(row["K_max"]),
            r_base_ricker=_maybe_float(row["r_base_ricker"]),
            r_min_ricker=_maybe_float(row["r_min_ricker"]),
            r_max_ricker=_maybe_float(row["r_max_ricker"]),
            r_base_lgm=_maybe_float(row["r_base_lgm"]),
            r_min_lgm=_maybe_float(row["r_min_lgm"]),
            r_max_lgm=_maybe_float(row["r_max_lgm"]),
        )
    if len(pops) != NUM_REAL_POPULATIONS:
        raise ValueError(
            f"species.csv must contain {NUM_REAL_POPULATIONS} populations; "
            f"found {len(pops)}"
        )
    for pop in pops.values():
        if pop.N0 <= 0.0 or pop.K_base <= 0.0 or pop.K_max < pop.K_base:
            raise ValueError(f"invalid population scale for {pop.common_name}")
        for family in (*RICKER_FAMILIES, THETA_FAMILY):
            r_base, r_min, r_max = pop.caps(family)
            if not (r_min <= r_base <= r_max):
                raise ValueError(
                    f"{pop.common_name} {family} rate caps do not bracket baseline"
                )
    return pops


def load_effects(data_dir: Path | str = DATA_DIR) -> dict[tuple[str, str], RealEffect]:
    path = Path(data_dir) / "action_effects_long.csv"
    rows = _read_csv(path)
    _require_columns(
        path,
        rows,
        {
            "common_name", "action", "channel", "lambda", "r_setpoint_ricker",
            "r_setpoint_lgm", "dK_step", "dN", "cost_step",
        },
    )
    effects: dict[tuple[str, str], RealEffect] = {}
    for row in rows:
        effect = RealEffect(
            common_name=row["common_name"],
            action=row["action"],
            index=_action_index(row["action"]),
            channel=row["channel"],
            lambda_value=_maybe_float(row["lambda"]),
            r_setpoint_ricker=_maybe_float(row["r_setpoint_ricker"]),
            r_setpoint_lgm=_maybe_float(row["r_setpoint_lgm"]),
            dK_step=_maybe_float(row["dK_step"]),
            dN=_maybe_float(row["dN"]),
            cost_step=_maybe_float(row["cost_step"]),
        )
        effects[(row["common_name"], row["action"])] = effect
    actions = load_actions(data_dir)
    pops = load_pops(data_dir)
    missing = [
        (pop, action)
        for pop in pops
        for action in actions
        if (pop, action) not in effects
    ]
    if missing:
        raise ValueError(
            "action_effects_long.csv missing required population/action rows: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    for (pop_name, action), effect in effects.items():
        if pop_name not in pops:
            raise ValueError(f"unknown population in action_effects_long.csv: {pop_name}")
        if action not in actions:
            raise ValueError(f"unknown action in action_effects_long.csv: {action}")
        for family in (*RICKER_FAMILIES, THETA_FAMILY):
            _r_base, r_min, r_max = pops[pop_name].caps(family)
            r_value = effect.r_setpoint(family)
            if r_value < r_min - RATE_TOLERANCE or r_value > r_max + RATE_TOLERANCE:
                raise ValueError(
                    f"{pop_name}/{action}/{family} set-point {r_value} "
                    f"outside data caps [{r_min}, {r_max}]"
                )
    return effects


# Module-level caches keyed by resolved data directory.
_ACTIONS_CACHE: dict[str, dict[str, RealAction]] = {}
_POPS_CACHE: dict[str, dict[str, RealPopulation]] = {}
_EFFECTS_CACHE: dict[str, dict[tuple[str, str], RealEffect]] = {}


def actions_for(data_dir: Path | str = DATA_DIR) -> dict[str, RealAction]:
    key = str(Path(data_dir))
    if key not in _ACTIONS_CACHE:
        _ACTIONS_CACHE[key] = load_actions(data_dir)
    return _ACTIONS_CACHE[key]


def pops_for(data_dir: Path | str = DATA_DIR) -> dict[str, RealPopulation]:
    key = str(Path(data_dir))
    if key not in _POPS_CACHE:
        _POPS_CACHE[key] = load_pops(data_dir)
    return _POPS_CACHE[key]


def effects_for(data_dir: Path | str = DATA_DIR) -> dict[tuple[str, str], RealEffect]:
    key = str(Path(data_dir))
    if key not in _EFFECTS_CACHE:
        _EFFECTS_CACHE[key] = load_effects(data_dir)
    return _EFFECTS_CACHE[key]


def population_names(data_dir: Path | str = DATA_DIR) -> list[str]:
    return list(pops_for(data_dir).keys())


def recoverable_population_names(data_dir: Path | str = DATA_DIR) -> list[str]:
    """Populations whose strongest recovery makes them grow (Ricker r_max > 0).

    The two demographic sinks (Egyptian vulture, bottlenose dolphin) have
    r_max < 0 and are excluded; they are reported separately, never pooled into
    the headline collapse-band metric.
    """
    return [
        name for name, pop in pops_for(data_dir).items() if pop.r_max_ricker > 0.0
    ]


def sink_population_names(data_dir: Path | str = DATA_DIR) -> list[str]:
    return [
        name for name, pop in pops_for(data_dir).items() if pop.r_max_ricker <= 0.0
    ]
