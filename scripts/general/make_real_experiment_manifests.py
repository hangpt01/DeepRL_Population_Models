#!/usr/bin/env python3
"""Create contiguous-index manifests for the real-ecology reward-mode runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata


FIELDS = [
    "index",
    "reward_mode",
    "population",
    "recoverable",
    "environment",
    "sigma_obs",
    "method",
    "filter",
    "expose_rk",
    "target_rows",
    "actual_rows",
    "overshoot_rows",
    "episode_count",
    "job_kind",
]
METHODS = (
    "mopo", "refplan", "bamcts", "plus", "moor",
    "ensemble_value_disagreement_pessimism", "ogsrl",
)
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS_FULL = (0.0, 0.1, 0.2, 0.4)
SIGMAS_PILOT = (0.0, 0.4)
REWARD_MODES = ("safe", "yield")
PILOT_POPULATIONS = (
    "Amur tiger",
    "Spotted turtle",
    "Egyptian vulture",
    "Bottlenose dolphin",
    "Jaguar",
    "Crab-eating fox",
)
PROBE_ROWS = (
    ("safe", "Amur tiger", "allee", 0.4, "bamcts", "learned"),
    ("safe", "Amur tiger", "regime", 0.4, "ogsrl", "learned"),
    ("yield", "Egyptian vulture", "theta", 0.4, "ogsrl", "learned"),
    ("safe", "Jaguar", "regime", 0.4, "mopo", "learned"),
    ("safe", "Amur tiger", "allee", 0.4, "plus", "learned"),
)


def _recoverable(population: str) -> bool:
    return population in set(realdata.recoverable_population_names())


def _row(
    *,
    reward_mode: str,
    population: str,
    family: str,
    sigma: float,
    method: str = "",
    filter_mode: str = "",
    job_kind: str,
) -> dict[str, object]:
    return {
        "reward_mode": reward_mode,
        "population": population,
        "recoverable": _recoverable(population),
        "environment": family,
        "sigma_obs": sigma,
        "method": method,
        "filter": filter_mode,
        "expose_rk": "hidden",
        "target_rows": 4000,
        "actual_rows": "",
        "overshoot_rows": "",
        "episode_count": "",
        "job_kind": job_kind,
    }


def _cell_rows(
    *,
    populations: Iterable[str],
    sigmas: Iterable[float],
    include_raw: bool,
    job_kind: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reward_mode in REWARD_MODES:
        for population in populations:
            for family in FAMILIES:
                for sigma in sigmas:
                    for method in METHODS:
                        rows.append(
                            _row(
                                reward_mode=reward_mode,
                                population=population,
                                family=family,
                                sigma=sigma,
                                method=method,
                                filter_mode="learned",
                                job_kind=job_kind,
                            )
                        )
                    if include_raw:
                        for method in METHODS:
                            rows.append(
                                _row(
                                    reward_mode=reward_mode,
                                    population=population,
                                    family=family,
                                    sigma=sigma,
                                    method=method,
                                    filter_mode="raw",
                                    job_kind=job_kind,
                                )
                            )
    return rows


def _split_plus(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    plus = [row for row in rows if row["method"] == "plus"]
    fast = [row for row in rows if row["method"] != "plus"]
    return fast, plus


def _write(path: Path, rows: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, **row})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "real_reward_modes_20260703"))
    parser.add_argument(
        "--skip-all-filters",
        action="store_true",
        help="omit the optional raw-filter full-grid manifests",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    all_populations = tuple(realdata.population_names())
    probe = [
        _row(
            reward_mode=reward_mode,
            population=population,
            family=family,
            sigma=sigma,
            method=method,
            filter_mode=filter_mode,
            job_kind="probe",
        )
        for reward_mode, population, family, sigma, method, filter_mode in PROBE_ROWS
    ]
    gates = [
        _row(
            reward_mode=reward_mode,
            population=population,
            family=family,
            sigma=sigma,
            job_kind="gate",
        )
        for reward_mode in REWARD_MODES
        for population in all_populations
        for family in FAMILIES
        for sigma in SIGMAS_FULL
    ]
    pilot = _cell_rows(
        populations=PILOT_POPULATIONS,
        sigmas=SIGMAS_PILOT,
        include_raw=False,
        job_kind="pilot",
    )
    full_learned = _cell_rows(
        populations=all_populations,
        sigmas=SIGMAS_FULL,
        include_raw=False,
        job_kind="full_learned",
    )

    outputs: dict[str, int] = {}
    outputs["manifest_probe.csv"] = _write(root / "manifest_probe.csv", probe)
    outputs["manifest_gates.csv"] = _write(root / "manifest_gates.csv", gates)
    outputs["manifest_pilot.csv"] = _write(root / "manifest_pilot.csv", pilot)
    pilot_fast, pilot_plus = _split_plus(pilot)
    outputs["manifest_pilot_fast.csv"] = _write(root / "manifest_pilot_fast.csv", pilot_fast)
    outputs["manifest_pilot_plus.csv"] = _write(root / "manifest_pilot_plus.csv", pilot_plus)
    outputs["manifest_full_learned.csv"] = _write(root / "manifest_full_learned.csv", full_learned)
    full_fast, full_plus = _split_plus(full_learned)
    outputs["manifest_full_learned_fast.csv"] = _write(
        root / "manifest_full_learned_fast.csv", full_fast
    )
    outputs["manifest_full_learned_plus.csv"] = _write(
        root / "manifest_full_learned_plus.csv", full_plus
    )
    if not args.skip_all_filters:
        full_all = _cell_rows(
            populations=all_populations,
            sigmas=SIGMAS_FULL,
            include_raw=True,
            job_kind="full_all_filters",
        )
        outputs["manifest_full_all_filters.csv"] = _write(
            root / "manifest_full_all_filters.csv", full_all
        )
        full_all_fast, full_all_plus = _split_plus(full_all)
        outputs["manifest_full_all_filters_fast.csv"] = _write(
            root / "manifest_full_all_filters_fast.csv", full_all_fast
        )
        outputs["manifest_full_all_filters_plus.csv"] = _write(
            root / "manifest_full_all_filters_plus.csv", full_all_plus
        )

    print(json.dumps({"output_root": str(root), "rows": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
