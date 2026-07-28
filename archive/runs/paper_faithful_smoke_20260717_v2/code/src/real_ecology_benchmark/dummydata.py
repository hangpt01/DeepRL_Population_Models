"""Small in-code dummy data source for set-point-r / cumulative-K experiments.

This module deliberately mirrors the real-data access shape without reading any
CSV files.  The values are toy/synthetic, but their semantics match the real
setting: action ``delta_r`` values are absolute set-points, while ``delta_K`` is
a per-step cumulative capacity increment.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_POPULATION = "Dummy baseline"
NUM_DUMMY_ACTIONS = 5
SUPPORTED_ACTION_COUNTS = {NUM_DUMMY_ACTIONS}


@dataclass(frozen=True)
class DummyAction:
    action: str
    index: int
    name_original: str
    cost_step: float


@dataclass(frozen=True)
class DummyPopulation:
    common_name: str
    N0: float
    K_base: float
    K_max: float
    r_base: float
    r_min: float
    r_max: float

    def caps(self, family: str) -> tuple[float, float, float]:
        return self.r_base, self.r_min, self.r_max


@dataclass(frozen=True)
class DummyEffect:
    common_name: str
    action: str
    index: int
    r_setpoint_value: float
    dK_step: float
    dN: float
    cost_step: float

    def r_setpoint(self, family: str) -> float:
        return self.r_setpoint_value


_ACTIONS = {
    "a0": DummyAction("a0", 0, "Do Nothing", 0.00),
    "a1": DummyAction("a1", 1, "Reduced Growth Pressure", -0.08),
    "a2": DummyAction("a2", 2, "Recruitment Support", 0.12),
    "a3": DummyAction("a3", 3, "Moderate Habitat Restoration", 0.16),
    "a4": DummyAction("a4", 4, "Integrated Conservation", 0.32),
}

_POPS = {
    DEFAULT_POPULATION: DummyPopulation(
        common_name=DEFAULT_POPULATION,
        N0=200.0,
        K_base=500.0,
        K_max=1000.0,
        r_base=0.12,
        r_min=-0.15,
        r_max=0.40,
    )
}

_EFFECTS = {
    (DEFAULT_POPULATION, "a0"): DummyEffect(DEFAULT_POPULATION, "a0", 0, 0.12, 0.0, 0.0, 0.00),
    (DEFAULT_POPULATION, "a1"): DummyEffect(DEFAULT_POPULATION, "a1", 1, -0.04, 0.0, 0.0, -0.08),
    (DEFAULT_POPULATION, "a2"): DummyEffect(DEFAULT_POPULATION, "a2", 2, 0.20, 0.0, 0.0, 0.12),
    (DEFAULT_POPULATION, "a3"): DummyEffect(DEFAULT_POPULATION, "a3", 3, 0.12, 75.0, 0.0, 0.16),
    (DEFAULT_POPULATION, "a4"): DummyEffect(DEFAULT_POPULATION, "a4", 4, 0.24, 125.0, 0.0, 0.32),
}


def actions_for(num_actions: int = NUM_DUMMY_ACTIONS) -> dict[str, DummyAction]:
    _validate_num_actions(num_actions)
    return {key: _ACTIONS[key] for key in _action_keys(num_actions)}


def pops_for() -> dict[str, DummyPopulation]:
    return dict(_POPS)


def effects_for(num_actions: int = NUM_DUMMY_ACTIONS) -> dict[tuple[str, str], DummyEffect]:
    _validate_num_actions(num_actions)
    keys = set(_action_keys(num_actions))
    return {
        key: effect
        for key, effect in _EFFECTS.items()
        if key[1] in keys
    }


def population_names() -> list[str]:
    return list(_POPS)


def _action_keys(num_actions: int) -> list[str]:
    return [f"a{i}" for i in range(num_actions)]


def _validate_num_actions(num_actions: int) -> None:
    if int(num_actions) not in SUPPORTED_ACTION_COUNTS:
        raise ValueError(
            f"dummy data supports action counts {sorted(SUPPORTED_ACTION_COUNTS)}; "
            f"got {num_actions}"
        )
    for key in _action_keys(num_actions):
        if key not in _ACTIONS:
            raise ValueError(f"dummy action table missing {key}")
        if (DEFAULT_POPULATION, key) not in _EFFECTS:
            raise ValueError(f"dummy effect table missing {DEFAULT_POPULATION}/{key}")
