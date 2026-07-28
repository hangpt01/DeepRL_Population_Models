#!/usr/bin/env python3
"""Run one registered Python entry point with all numerical threads fixed to one."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: ricker_only_thread_entry.py RUNNER [ARGS...]")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runner = Path(sys.argv[1]).resolve()
    sys.argv = [str(runner), *sys.argv[2:]]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
