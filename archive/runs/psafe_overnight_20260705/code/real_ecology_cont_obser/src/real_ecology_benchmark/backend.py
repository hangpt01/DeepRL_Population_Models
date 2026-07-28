"""Compute backend selection (NumPy reference, CuPy/GPU optional).

The vectorized real-ecology transition used by mechanistic proposals is written
against an array module ``xp`` that is either ``numpy`` (CPU reference, default)
or ``cupy`` (CUDA GPU).  The NumPy branch is the reference implementation and
always available; the CuPy branch is used only when ``backend="cupy"`` is
requested *and* CuPy imports with a usable device.

Fairness: a ranking run declares one backend; every summary records both the
*requested* and the *effective* backend so CPU and GPU rows are never silently
mixed.  In ``strict`` mode an unavailable/unsupported backend fails loudly rather
than falling back.

This module has no hard dependency on CuPy; importing it on a CPU-only node is
safe.  GPU execution is only exercised on a node where ``import cupy`` succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


SUPPORTED_BACKENDS = ("numpy", "cupy")
GPU_SUPPORTED_WORKLOADS = frozenset({
    "mechanistic_transition",
    "method:plus",
    "oracle_ablation:plus",
})
PLUS_GPU_WORKLOADS = frozenset({"method:plus", "oracle_ablation:plus"})


class BackendUnavailable(RuntimeError):
    """Raised in strict mode when the requested backend cannot be used."""


def _try_import_cupy(device: int):
    """Return (cupy_module, error). error is None on success."""
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # ImportError or CUDA init error
        return None, exc
    try:
        cp.cuda.Device(int(device)).use()
        # Touch the device so a driver/runtime failure surfaces here, not mid-run.
        _ = cp.asarray([0.0]) + 0.0
    except Exception as exc:  # no device / driver mismatch
        return None, exc
    return cp, None


@dataclass(frozen=True)
class Backend:
    """Resolved compute backend.

    ``xp`` is the array module actually in use (numpy or cupy).  ``requested`` is
    what the config asked for; ``name`` is what is effectively used (they differ
    only when ``strict=False`` allowed a CuPy->NumPy fallback).
    """

    name: str          # effective backend actually used
    requested: str     # backend requested by config
    device: int
    strict: bool
    xp: Any
    fallback_reason: str | None = None
    workload: str = "unspecified"
    acceleration_scope: str = "full_numpy"

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def asarray(self, array):
        """Move a host array onto the backend device."""
        return self.xp.asarray(array)

    def to_numpy(self, array) -> np.ndarray:
        """Return a host NumPy array regardless of backend (API boundary)."""
        if self.name == "cupy":
            return self.xp.asnumpy(array)
        return np.asarray(array)

    def to_dict(self) -> dict[str, object]:
        return {
            "compute_backend_requested": self.requested,
            "compute_backend_effective": self.name,
            "compute_device": int(self.device),
            "compute_backend_device": int(self.device),
            "compute_backend_strict": bool(self.strict),
            "compute_backend_fallback_reason": self.fallback_reason,
            "compute_backend_workload": self.workload,
            "compute_backend_acceleration_scope": self.acceleration_scope,
        }


_NUMPY_BACKEND = Backend(
    name="numpy", requested="numpy", device=-1, strict=True, xp=np, fallback_reason=None
)


def numpy_backend() -> Backend:
    return _NUMPY_BACKEND


def resolve_backend(compute_cfg: Any | None) -> Backend:
    """Resolve a :class:`Backend` from a compute config (or None -> numpy).

    ``compute_cfg`` is duck-typed: it needs ``backend`` (str), ``device`` (int),
    ``strict`` (bool).  A ``None`` config yields the NumPy reference backend.
    """

    if compute_cfg is None:
        return _NUMPY_BACKEND
    requested = str(getattr(compute_cfg, "backend", "numpy"))
    device = int(getattr(compute_cfg, "device", 0))
    strict = bool(getattr(compute_cfg, "strict", True))
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown compute backend: {requested!r}")
    if requested == "numpy":
        return Backend("numpy", "numpy", device, strict, np, None)
    # requested == "cupy"
    cp, error = _try_import_cupy(device)
    if cp is not None:
        return Backend("cupy", "cupy", device, strict, cp, None)
    message = f"requested backend 'cupy' is unavailable: {error!r}"
    if strict:
        raise BackendUnavailable(message)
    return Backend("numpy", "cupy", device, strict, np, str(error))


def _cupy_scope_for_workload(workload: str, env_cfg: Any | None = None) -> str | None:
    if workload == "mechanistic_transition":
        return "mechanistic_transition"
    if workload in PLUS_GPU_WORKLOADS:
        if getattr(env_cfg, "control_mode", None) == "real_setpoint":
            # PLUS keeps CPU orchestration (planner, posterior weights, I/O), but
            # its candidate-bank transition batches run through MechanisticProposal
            # on the active CuPy backend.  This is the actual PLUS hot path.
            return "plus_mechanistic_transition"
        return None
    return None


def ensure_backend_supports_workload(
    backend: Backend,
    workload: str,
    env_cfg: Any | None = None,
) -> Backend:
    """Return a backend that is honest for ``workload``.

    CuPy support is currently limited to the vectorized real-transition kernel
    used inside mechanistic proposals, plus PLUS real-setpoint rows whose hot
    path is exactly that candidate-bank mechanistic transition.  Other full
    method/gate rows still contain NumPy-only planner/model paths, so strict
    CuPy runs must fail rather than being mislabeled as GPU results.
    """

    if backend.name != "cupy":
        scope = "fallback_numpy" if backend.requested != backend.name else "full_numpy"
        return replace(backend, workload=workload, acceleration_scope=scope)
    scope = _cupy_scope_for_workload(workload, env_cfg)
    if scope is not None:
        return replace(backend, workload=workload, acceleration_scope=scope)
    message = (
        "requested backend 'cupy' is unsupported for workload "
        f"{workload!r}: method/gate planning and learned dynamics are still "
        "NumPy-only, except PLUS real_setpoint rows whose mechanistic-transition "
        "hot path is CuPy-enabled"
    )
    if backend.strict:
        raise BackendUnavailable(message)
    return Backend(
        "numpy", backend.requested, backend.device, backend.strict, np, message,
        workload=workload, acceleration_scope="fallback_numpy",
    )


def resolve_backend_for_workload(
    compute_cfg: Any | None,
    workload: str,
    env_cfg: Any | None = None,
) -> Backend:
    """Resolve the requested backend and enforce the workload support matrix."""

    return ensure_backend_supports_workload(resolve_backend(compute_cfg), workload, env_cfg)


# Per-process active backend.  A Slurm task runs one cell in one process, so the
# runner resolves the backend once (from ``cfg.compute``) and sets it here; the
# hot paths (mechanistic proposal, gate proposal) read it as their default so the
# backend does not have to be threaded through every method constructor.  Tests
# and explicit callers can still pass a Backend directly (explicit wins).
_ACTIVE: Backend = _NUMPY_BACKEND


def get_active_backend() -> Backend:
    return _ACTIVE


def set_active_backend(backend: Backend) -> Backend:
    global _ACTIVE
    _ACTIVE = backend
    return _ACTIVE


def reset_active_backend() -> Backend:
    return set_active_backend(_NUMPY_BACKEND)
