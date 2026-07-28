"""Real-ecology (Tier-4) continuous-state ecological offline-RL benchmark.

A self-contained vendored adaptation of the Tier-2/3 ``tier2_benchmark`` package.
The POMDP, the four dynamics families, the seven methods, the particle filter,
the MPC, the unified evaluator, the seed protocol, and the public/private schema
guard are reused verbatim; only the data layer (real populations + portal costs),
the environment (set-point growth, cumulative capacity, translocation), and the
per-population reward/safety/calibration are changed.  Nothing in the original
``discrete_action_cont_obser/`` package is modified.
"""

from . import realdata
from .actions import ActionSpec, action_table, real_action_table, resolve_actions
from .backend import Backend, resolve_backend
from .config import (
    BenchmarkConfig,
    ComputeConfig,
    EnvironmentConfig,
    environment_with_kind_defaults,
    load_config,
    real_environment,
)
from .envs import ContinuousEcologyEnv, make_env

__all__ = [
    "ActionSpec",
    "Backend",
    "BenchmarkConfig",
    "ComputeConfig",
    "ContinuousEcologyEnv",
    "EnvironmentConfig",
    "resolve_backend",
    "action_table",
    "environment_with_kind_defaults",
    "load_config",
    "make_env",
    "real_action_table",
    "real_environment",
    "realdata",
    "resolve_actions",
]

__version__ = "0.1.0"
