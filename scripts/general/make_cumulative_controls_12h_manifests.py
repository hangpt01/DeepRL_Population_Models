#!/usr/bin/env python3
"""Create contiguous-index manifests for the cumulative-control 12h run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["index", "environment", "num_actions", "sigma_obs", "method", "filter"]
ENVS = ("allee", "theta", "regime")
ACTIONS = (5, 10)
SIGMAS = (0.0, 0.1, 0.2, 0.4)
LEARNED_METHODS = (
    "mopo", "refplan", "bamcts", "plus", "moor",
    "ensemble_value_disagreement_pessimism", "ogsrl",
)


def _all_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for env in ENVS:
        for num_actions in ACTIONS:
            for sigma in SIGMAS:
                for method in LEARNED_METHODS:
                    rows.append(
                        {
                            "environment": env,
                            "num_actions": num_actions,
                            "sigma_obs": sigma,
                            "method": method,
                            "filter": "learned",
                        }
                    )
                for method in ("plus", "moor"):
                    rows.append(
                        {
                            "environment": env,
                            "num_actions": num_actions,
                            "sigma_obs": sigma,
                            "method": method,
                            "filter": "ricker",
                        }
                    )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "cumulative_controls_12h"),
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = _all_rows()
    prewarm = [
        row for row in rows
        if row["method"] == "moor" and row["filter"] in {"learned", "ricker"}
    ]
    main_rows = [row for row in rows if row not in prewarm]
    probe = [
        {
            "environment": "allee",
            "num_actions": 10,
            "sigma_obs": 0.4,
            "method": "ogsrl",
            "filter": "learned",
        }
    ]

    _write(root / "manifest_probe.csv", probe)
    _write(root / "manifest_prewarm.csv", prewarm)
    _write(root / "manifest_main.csv", main_rows)
    print(
        f"probe={len(probe)} prewarm={len(prewarm)} main={len(main_rows)} "
        f"total={len(prewarm) + len(main_rows)} root={root}"
    )


if __name__ == "__main__":
    main()
