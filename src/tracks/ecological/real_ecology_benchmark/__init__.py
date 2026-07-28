"""Unified continuous-state ecological offline-RL benchmark.

The package hosts the real-ecology data setting, dummy set-point experiments,
and legacy synthetic control modes under one implementation.  Shared POMDP
plumbing, dynamics families, methods, particle filters, MPC, evaluation,
seed protocol, and public/private schema guards all live here.
"""

from . import dummydata, realdata
from .actions import ActionSpec, action_table, dummy_action_table, real_action_table, resolve_actions
from .backend import Backend, resolve_backend
from .config import (
    BenchmarkConfig,
    ComputeConfig,
    EnvironmentConfig,
    MethodContext,
    LEGACY_REAL_SETPOINT,
    SETPOINT_CUMULATIVE,
    environment_with_kind_defaults,
    dummy_environment,
    is_setpoint_cumulative,
    hides_rk,
    load_config,
    normalize_control_mode,
    real_environment,
    synthetic_environment,
)
from .envs import ContinuousEcologyEnv, make_env

__all__ = [
    "ActionSpec",
    "Backend",
    "BenchmarkConfig",
    "ComputeConfig",
    "ContinuousEcologyEnv",
    "EnvironmentConfig",
    "MethodContext",
    "LEGACY_REAL_SETPOINT",
    "SETPOINT_CUMULATIVE",
    "dummy_action_table",
    "dummy_environment",
    "dummydata",
    "is_setpoint_cumulative",
    "hides_rk",
    "resolve_backend",
    "action_table",
    "environment_with_kind_defaults",
    "load_config",
    "make_env",
    "normalize_control_mode",
    "real_action_table",
    "real_environment",
    "realdata",
    "resolve_actions",
    "synthetic_environment",
]

__version__ = "0.1.0"
