"""Repository and external-data paths shared by diagnostic entry points.

Scientific code always resolves inside this repository. External scratch roots
are used only for datasets, fit caches, receipts, and run outputs.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("DEEPRL_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
SCRATCH_PROJECT = Path(
    os.environ.get(
        "DEEPRL_SCRATCH_PROJECT",
        "/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models",
    )
).resolve()
RUNS_ROOT = Path(
    os.environ.get("DEEPRL_RUNS_ROOT", SCRATCH_PROJECT / "real_ecology_runs")
).resolve()

P10_DATA_ROOT = Path(
    os.environ.get(
        "DEEPRL_P10_DATA_ROOT",
        RUNS_ROOT / "three_species_ecological_p10_correction_20260723_v1",
    )
).resolve()
P10_PACKAGE = Path(
    os.environ.get("DEEPRL_P10_PACKAGE", REPO_ROOT / "experiments" / "accepted_p10")
).resolve()
ACCEPTED_CSV = Path(
    os.environ.get(
        "DEEPRL_ACCEPTED_CSV",
        REPO_ROOT / "results" / "accepted" / "MATCHED_P10_144_METHOD_CELLS.csv",
    )
).resolve()

ECOLOGICAL_SRC = Path(
    os.environ.get(
        "DEEPRL_ECOLOGICAL_SRC", REPO_ROOT / "src" / "tracks" / "ecological"
    )
).resolve()
ECOLOGICAL_SCRIPTS = Path(
    os.environ.get(
        "DEEPRL_ECOLOGICAL_SCRIPTS", REPO_ROOT / "scripts" / "ecological"
    )
).resolve()
GENERAL_SRC = Path(
    os.environ.get("DEEPRL_GENERAL_SRC", REPO_ROOT / "src" / "tracks" / "general")
).resolve()
GENERAL_SCRIPTS = Path(
    os.environ.get("DEEPRL_GENERAL_SCRIPTS", REPO_ROOT / "scripts" / "general")
).resolve()

GENERAL_RUN_ROOT = Path(
    os.environ.get(
        "DEEPRL_GENERAL_RUN_ROOT",
        "/home/hphung/ce25_scratch2/general_rl_phase2_iso/real_ecology_runs/"
        "general_phase2e_full_sigma01_02_20260720_v1",
    )
).resolve()
GENERAL_DATA_ROOT = Path(
    os.environ.get("DEEPRL_GENERAL_DATA_ROOT", GENERAL_RUN_ROOT / "quarantine")
).resolve()
GENERAL_PACKAGE = Path(
    os.environ.get(
        "DEEPRL_GENERAL_PACKAGE", REPO_ROOT / "experiments" / "accepted_general"
    )
).resolve()

REPLAY_OUTPUT = Path(
    os.environ.get(
        "DEEPRL_REPLAY_OUTPUT", REPO_ROOT / ".verification" / "diagnostic_replay"
    )
).resolve()
FOLLOWUPS_OUTPUT = Path(
    os.environ.get(
        "DEEPRL_FOLLOWUPS_OUTPUT", REPO_ROOT / ".verification" / "followups"
    )
).resolve()


def require_scientific_tables() -> Path:
    """Fail early if the frozen packages' repository-root data link is unavailable."""
    path = REPO_ROOT / "src" / "tracks" / "real_ecology_data"
    required = ("actions.csv", "action_effects_long.csv", "species.csv")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "scientific-table compatibility link is missing or incomplete: "
            f"{path} (missing {', '.join(missing)}); restore "
            "src/tracks/real_ecology_data -> ../../configs/ecology"
        )
    return path
