#!/usr/bin/env python3
"""Launch-only entry point that fixes numerical-library and Torch thread counts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    parser.add_argument("runner", type=Path)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    value = str(args.threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    if torch.get_num_threads() != args.threads or torch.get_num_interop_threads() != args.threads:
        raise SystemExit("failed to fix Torch thread counts")
    sys.argv = [str(args.runner), *args.runner_args]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
