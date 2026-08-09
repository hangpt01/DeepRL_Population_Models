#!/usr/bin/env python3
"""Recovery gate preflight plus the original fail-closed Arm O gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


DOC = Path(__file__).resolve().parent
EXPECTED = {
    "run_arm_t_task.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "run_arm_t_task_recovery1.py": "c74fcb0cdee951ab608f3d2ac94d8666d9e74ccc359894f0ec1f6f1893a6996d",
    "arm_o_gate.py": "239fa3377cac41cd3f5fd6d42dd7849560e16fcd4b7379dd6697612d6639147b",
    "ARM_T_TASKS.json": "f48e4940c4711f811008b17e131d7d08de119c9c6ae60c15e8d637ecf6a44970",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name, expected in EXPECTED.items():
        observed = sha(DOC / name)
        if observed != expected:
            raise RuntimeError(f"recovery gate identity mismatch for {name}: {observed}")
    tasks = json.loads((DOC / "ARM_T_TASKS.json").read_text(encoding="utf-8"))
    if tasks.get("task_count") != 12 or len(tasks.get("tasks", [])) != 12:
        raise RuntimeError("Arm T task count is not exactly twelve")
    if tasks.get("common", {}).get("slurm_constraint") != "xenon-8452Y":
        raise RuntimeError("pinned 8452Y profile is absent")
    spec = importlib.util.spec_from_file_location("original_i2b_arm_o_gate", DOC / "arm_o_gate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original Arm O gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
