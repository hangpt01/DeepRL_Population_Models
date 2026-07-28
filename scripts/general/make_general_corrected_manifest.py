#!/usr/bin/env python3
"""Prepare (do NOT submit) the corrected hidden general-RL comparison manifest.

Routes the retained four general baselines -- RefPlan, OGSRL, BA-MCTS, and the
bootstrap Q-ensemble value-disagreement method -- through the hidden public
pipeline at the matched 4,000-transition budget.  MOPO is intentionally excluded
(archived, not deleted).  The ecological natives (moor_native / plus_native)
remain in their own manifest and are not part of this generator.

This only writes a manifest CSV; it launches nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.manifest import make_manifest


# Retained general-RL baselines, all over the shared public learned pipeline.
ROUTING = {
    "refplan": ("learned",),
    "ogsrl": ("learned",),
    "bamcts": ("learned",),
    "ensemble_value_disagreement_pessimism": ("learned",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "real_ecology_runs" / "general_corrected_prepared" / "manifest_general_hidden.csv"),
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = make_manifest(
        args.output,
        methods=tuple(ROUTING),
        data_mode="real",
        method_filters=ROUTING,
        expose_rk="hidden",
    )
    print(json.dumps(
        {
            "output": args.output,
            "rows": rows,
            "methods": list(ROUTING),
            "expose_rk": "hidden",
            "target_transitions": 4000,
            "submitted": False,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
