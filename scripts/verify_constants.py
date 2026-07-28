#!/usr/bin/env python3
"""Independent constants acceptance check for the diagnostic pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "diagnostics" / "replay_analysis"))

from constants import dominated_actions, max_reachable_abundance  # noqa: E402


def main() -> int:
    bound = max_reachable_abundance("egyptian_vulture")
    expected = {
        "egyptian_vulture": {5: 0, 6: 0, 7: 3, 8: 3, 9: 4},
        "amur_tiger": {5: 0, 6: 0},
        "crab_eating_fox": {},
    }
    assert abs(bound - 42.93729603220856) <= 1e-12, bound
    observed = {species: dominated_actions(species) for species in expected}
    assert observed == expected, observed
    print(f"PASS vulture_bound={bound:.12f}")
    for species, actions in observed.items():
        print(f"PASS dominated_actions[{species}]={actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

