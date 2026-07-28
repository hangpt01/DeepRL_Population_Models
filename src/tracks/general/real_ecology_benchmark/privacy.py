"""Fail-loud guards for hidden-r/K method-facing objects and metadata."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_METHOD_NAMES = frozenset(
    {
        "r",
        "rho",
        "kappa",
        "r_base",
        "r_eff",
        "r_eff_true",
        "k_base",
        "k_ref",
        "k_eff",
        "k_min",
        "k_max",
        "delta_r",
        "delta_k",
        "c_low",
        "c_high",
        "s_safe",
        "c_safe",
        "safety_threshold",
        "safety_fraction",
        "safety_margin",
        "initially_unsafe",
        "unsafe_label",
        "entry",
        "reward_true",
        "benefit",
        "kind",
        "family",
        "true_family",
        "population",
        "data_dir",
        "action_effects",
        "action_specs",
        "lambda_profile",
        "k_multiplier",
        "dn_fraction",
        "lambda_source",
        "lambda",
        "r_setpoint_ricker",
        "r_setpoint_lgm",
        "dk_step",
        "dn",
        "common_name",
        "scientific_name",
        "name_mechanistic",
        "name_original",
        "interpretation",
        "cost_step",
    }
)


def forbidden_method_name(name: str) -> bool:
    raw = str(name).strip()
    if raw == "K":
        return True
    normalized = raw.casefold()
    return (
        normalized in FORBIDDEN_METHOD_NAMES
        or normalized.startswith("regime_threshold_")
        or normalized.startswith("r_setpoint_")
    )


def forbidden_paths(value: Any, root: str = "root") -> list[str]:
    """Return forbidden field paths without traversing code or array payloads."""

    found: list[str] = []
    seen: set[int] = set()

    def visit(current: Any, path: str) -> None:
        if current is None or isinstance(
            current, (str, bytes, int, float, bool, np.generic, Path)
        ):
            return
        if isinstance(current, np.ndarray):
            return
        identity = id(current)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(current, dict):
            for key, child in current.items():
                child_path = f"{path}.{key}"
                if forbidden_method_name(str(key)):
                    found.append(child_path)
                visit(child, child_path)
            return
        if isinstance(current, (list, tuple, set, frozenset)):
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]")
            return
        if is_dataclass(current):
            names = [item.name for item in fields(current)]
        elif hasattr(current, "__dict__"):
            names = list(vars(current))
        else:
            return
        for name in names:
            child_path = f"{path}.{name}"
            if forbidden_method_name(name):
                found.append(child_path)
            visit(getattr(current, name), child_path)

    visit(value, root)
    return sorted(set(found))


def assert_hidden_method_artifact(value: Any, root: str = "root") -> None:
    found = forbidden_paths(value, root)
    if found:
        raise AssertionError("hidden method artifact exposes private names: " + ", ".join(found))
