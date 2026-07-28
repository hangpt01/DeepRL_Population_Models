"""Lightweight runtime and memory telemetry helpers."""

from __future__ import annotations

import resource
import sys
import time


def perf_seconds() -> float:
    """Monotonic wall-clock timer for phase durations."""

    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return float(time.perf_counter() - start)


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB.

    Linux reports ``ru_maxrss`` in KiB; macOS reports bytes.  The experiment
    jobs run on Linux, but keeping the conversion portable makes local tests
    less surprising.
    """

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0
