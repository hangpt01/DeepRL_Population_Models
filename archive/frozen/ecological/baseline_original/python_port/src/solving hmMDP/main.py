"""Solve the generated POMDPX with SARSOP (same behavior as R script)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from config_loader import load_example_from_env, repo_root_from_here


def _resolve_sarsop_binary(root: Path) -> Path:
    """Resolve SARSOP executable across Linux/macOS/Windows builds."""
    candidates = [
        root / "sarsop" / "src" / "pomdpsol",
        root / "sarsop" / "src" / "pomdpsol.exe",
        root / "sarsop" / "src" / "pomdpsol.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find a SARSOP solver binary. Looked for:\n"
        f"{tried}\n"
        "Build SARSOP first or adjust this path."
    )


def main() -> None:
    cfg = load_example_from_env(__file__, default_module="examples2states6actions")
    root = repo_root_from_here(__file__)
    solver_bin = _resolve_sarsop_binary(root)

    precision = float(os.environ.get("UAMS_PRECISION", "0.1"))
    timeout = int(os.environ.get("UAMS_TIMEOUT", "3600"))
    cmd = [
        str(solver_bin),
        str(root / cfg["file_pomdpx"]),
        "--precision",
        str(precision),
        "--timeout",
        str(timeout),
        "--output",
        str(root / cfg["file_outpolicy"]),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
