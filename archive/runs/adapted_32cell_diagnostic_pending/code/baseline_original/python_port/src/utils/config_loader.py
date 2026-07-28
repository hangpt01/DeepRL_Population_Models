"""Helpers to load problem configurations from python_port/data."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict


def repo_root_from_here(current_file: str) -> Path:
    # python_port/src/... -> project root is 3 levels up
    return Path(current_file).resolve().parents[3]


def python_port_root_from_here(current_file: str) -> Path:
    # python_port/src/... -> python_port root is 2 levels up
    return Path(current_file).resolve().parents[2]


def load_example_from_env(
    current_file: str,
    default_module: str,
    env_key: str = "EXAMPLE_MODULE",
) -> Dict[str, Any]:
    """
    Load an example module from python_port/data.

    Example env values:
      - examples2states2actions
      - gouldian
      - data.gouldian
      - gouldian.py
    """
    module_name = os.environ.get(env_key, default_module).strip()
    module_name = module_name.replace(".py", "")
    if module_name.startswith("data."):
        module_name = module_name.split(".", 1)[1]

    python_port_root = python_port_root_from_here(current_file)
    data_dir = python_port_root / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))

    module = importlib.import_module(module_name)
    cfg = dict(module.CONFIG)
    cfg["module_name"] = module_name
    return cfg
