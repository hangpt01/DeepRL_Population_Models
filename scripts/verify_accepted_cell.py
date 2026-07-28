#!/usr/bin/env python3
"""Run one accepted ecological cell using clean code/config and external data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-project")
    parser.add_argument("--cell", default="A6")
    parser.add_argument("--side", default="moor", choices=("moor", "plus"))
    parser.add_argument("--output-root")
    parser.add_argument("--gate-self-test-corrupt", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not args.gate_self_test_corrupt and not args.scratch_project:
        parser.error("--scratch-project is required for an accepted-cell replay")
    scratch = (
        Path(args.scratch_project).resolve()
        if args.scratch_project
        else Path("/nonexistent-gate-self-test")
    )
    output = (
        Path(args.output_root).resolve()
        if args.output_root
        else root / ".verification" / "diagnostic_replay"
    )
    harness = root / "scripts" / "diagnostics" / "replay" / "run_diagnostic_replay.py"
    env = os.environ.copy()
    env.update(
        {
            "DEEPRL_SCRATCH_PROJECT": str(scratch),
            "DEEPRL_P10_PACKAGE": str(root / "experiments" / "accepted_p10"),
            "DEEPRL_ACCEPTED_CSV": str(
                root / "results" / "accepted" / "MATCHED_P10_144_METHOD_CELLS.csv"
            ),
            "DEEPRL_ECOLOGICAL_SRC": str(root / "src" / "tracks" / "ecological"),
            "DEEPRL_ECOLOGICAL_SCRIPTS": str(root / "scripts" / "ecological"),
            "DEEPRL_REPLAY_OUTPUT": str(output),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    command = (
        [sys.executable, str(harness), "--gate-self-test-corrupt"]
        if args.gate_self_test_corrupt
        else [sys.executable, str(harness), args.cell, args.side]
    )
    print("running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
