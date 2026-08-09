#!/usr/bin/env python3
"""Non-overwriting technical retry for the four ecological canary rows.

The initial array used /usr/bin/python, which lacks the accepted CPU PyTorch
dependency.  This wrapper changes only the new output subroot; its Slurm launcher
uses the exact accepted paper-faithful interpreter.
"""

from __future__ import annotations

import argparse

import run_arm_o_canary as canary


AUTHORIZED_TASKS = frozenset({0, 1, 6, 7})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_index", type=int)
    args = parser.parse_args()
    if args.task_index not in AUTHORIZED_TASKS:
        raise SystemExit("retry1 is restricted to ecological tasks 0,1,6,7")
    canary.OUTPUT_ROOT = canary.OUTPUT_ROOT / "ecological_retry1"
    canary.run(args.task_index)


if __name__ == "__main__":
    main()
