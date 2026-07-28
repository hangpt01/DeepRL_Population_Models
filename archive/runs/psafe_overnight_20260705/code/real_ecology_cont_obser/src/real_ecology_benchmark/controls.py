"""Public control helpers (real-ecology / Tier-4 adaptation).

Capacity control is cumulative (``kappa`` accumulator -> public ``K_eff``).  The
growth control ``rho`` carries the **set-point** intrinsic rate of the action in
force (the accumulator runs with decay 1, so ``rho`` equals the action's
set-point rather than a running sum).  Because the population identity is known
to the agent in the real setting, ``rho`` and the resulting
``r_eff = clip(rho, r_min, r_max)`` are not leaky; the per-population data caps
``r_min/r_max`` come straight from ``species.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import ActionSpec, action_table, resolve_actions
from .config import EnvironmentConfig


@dataclass(frozen=True)
class PublicControls:
    rho: float
    kappa: float
    K_eff: float

    def to_public_info(self) -> dict[str, float]:
        return {
            "rho": float(self.rho),
            "kappa": float(self.kappa),
            "K_eff": float(self.K_eff),
        }


def control_fields_enabled(cfg: EnvironmentConfig) -> bool:
    return cfg.control_mode in {"cumulative_capped", "real_setpoint"}


def initial_controls(cfg: EnvironmentConfig) -> PublicControls:
    rho = 0.0
    if cfg.control_mode == "real_setpoint":
        rho = float(resolve_actions(cfg)[0].delta_r)
    return public_controls(cfg, rho, 0.0)


def public_controls(cfg: EnvironmentConfig, rho: float, kappa: float) -> PublicControls:
    K_eff = float(np.clip(cfg.K_base + float(kappa), cfg.K_min, cfg.K_max))
    return PublicControls(float(rho), float(kappa), K_eff)


def advance_public_controls(
    cfg: EnvironmentConfig,
    action: int | np.ndarray,
    rho: float | np.ndarray,
    kappa: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(action, dtype=int)
    rho_arr = np.asarray(rho, dtype=np.float64)
    kappa_arr = np.asarray(kappa, dtype=np.float64)
    actions, rho_arr, kappa_arr = np.broadcast_arrays(actions, rho_arr, kappa_arr)
    specs = resolve_actions(cfg)
    delta_r = np.asarray([specs[int(a)].delta_r for a in actions.reshape(-1)])
    delta_K = np.asarray([specs[int(a)].delta_K for a in actions.reshape(-1)])
    delta_r = delta_r.reshape(actions.shape)
    delta_K = delta_K.reshape(actions.shape)
    next_rho = (1.0 - cfg.accumulator_decay_r) * rho_arr + delta_r
    next_kappa = (1.0 - cfg.accumulator_decay_K) * kappa_arr + delta_K
    K_eff = np.clip(cfg.K_base + next_kappa, cfg.K_min, cfg.K_max)
    return next_rho.astype(np.float64), next_kappa.astype(np.float64), K_eff.astype(np.float64)


def private_r_eff(
    cfg: EnvironmentConfig,
    r_base: float | np.ndarray,
    rho: float | np.ndarray,
    theta: float | np.ndarray | None = None,
) -> np.ndarray:
    r_base_arr = np.asarray(r_base, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    if cfg.control_mode == "real_setpoint":
        # rho already carries the action's set-point rate (decay_r == 1, r_base ==
        # 0); the caps are data-derived per population/family.  Valid action-table
        # set-points must already be inside those caps, so clipping is only a
        # numerical guard after a loud validation check.
        if np.any((rho_arr < cfg.r_min - 1e-9) | (rho_arr > cfg.r_max + 1e-9)):
            raise ValueError(
                "real set-point rate outside data-derived caps: "
                f"min={float(np.min(rho_arr))}, max={float(np.max(rho_arr))}, "
                f"caps=[{cfg.r_min}, {cfg.r_max}]"
            )
        return np.clip(rho_arr, cfg.r_min, cfg.r_max)
    if cfg.kind == "theta":
        if theta is None:
            raise ValueError("theta is required for theta-specific r cap")
        r_max = cfg.theta_stability_margin / np.asarray(theta, dtype=np.float64)
    else:
        r_max = cfg.r_max
    return np.clip(r_base_arr + rho_arr, cfg.r_min, r_max)


def requires_direct_stock(action: ActionSpec) -> bool:
    return action.harvest_fraction != 0.0 or action.stocking_delta != 0.0
