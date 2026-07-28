#!/usr/bin/env bash
# Freeze the approved hidden-r/K implementation and generate matched full/hidden manifests.
set -euo pipefail

RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RUN/../.." && pwd)"
SNAP="$RUN/code"

if [[ -e "$SNAP" ]]; then
  echo "refusing to overwrite frozen snapshot: $SNAP" >&2
  exit 1
fi

mkdir -p "$SNAP" "$RUN/manifests" "$RUN/logs" "$RUN/analysis"
for path in src configs scripts real_ecology_data pyproject.toml Makefile; do
  cp -r "$REPO/$path" "$SNAP/"
done
mkdir -p "$SNAP/docs/benchmark/hide_r_k"
cp "$REPO"/docs/benchmark/hide_r_k/* "$SNAP/docs/benchmark/hide_r_k/"
cp "$REPO/docs/benchmark/SERVER_HANDOFF_hidden_rk_experiment_setting.md" \
  "$SNAP/docs/benchmark/"

{
  echo "run:        hidden_rk_comparison_20260716"
  echo "frozen_at:  $(date -Is)"
  echo "repo:       $REPO"
  echo "git_commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "git_status:"
  git -C "$REPO" status --short
} > "$RUN/PROVENANCE.txt"

export PYTHONPATH="$SNAP/src"
python "$SNAP/scripts/make_hidden_rk_manifest.py" \
  --output "$RUN/manifests/manifest_hidden.csv"
python "$SNAP/scripts/make_motivation_native_manifest.py" \
  --output "$RUN/manifests/manifest_full.csv"

python - "$RUN/manifests/manifest_hidden.csv" "$RUN/manifests/manifest_full.csv" \
  "$SNAP/configs/hidden_rk.yaml" <<'PY'
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

from real_ecology_benchmark.config import load_config

hidden_path, full_path, config_path = map(Path, sys.argv[1:])
hidden = list(csv.DictReader(hidden_path.open()))
full = list(csv.DictReader(full_path.open()))
if len(hidden) != 1440 or len(full) != 1440:
    raise SystemExit(f"expected 1440 rows per arm, got hidden={len(hidden)} full={len(full)}")

key = lambda row: (
    row["reward_mode"], row["population"], row["environment"],
    row["sigma_obs"], row["method"], row["filter"],
)
hidden_keys = [key(row) for row in hidden]
full_keys = [key(row) for row in full]
if len(set(hidden_keys)) != len(hidden_keys) or len(set(full_keys)) != len(full_keys):
    raise SystemExit("a manifest contains duplicate scientific rows")
if set(hidden_keys) != set(full_keys):
    raise SystemExit("full and hidden manifests do not contain the same scientific rows")
if {row["expose_rk"] for row in hidden} != {"hidden"}:
    raise SystemExit("hidden manifest has an invalid regime label")
if {row["expose_rk"] for row in full} != {"full"}:
    raise SystemExit("full manifest has an invalid regime label")

cell = lambda row: (
    row["reward_mode"], row["population"], row["environment"], row["sigma_obs"],
)
for name, rows in (("hidden", hidden), ("full", full)):
    bad = [i for i in range(0, len(rows), 5) if len({cell(r) for r in rows[i:i + 5]}) != 1]
    if bad:
        raise SystemExit(f"{name} manifest is not five-row cell aligned at {bad[:5]}")

cfg = load_config(str(config_path))
if cfg.environment.collapse_penalty != 5.0:
    raise SystemExit(f"registered safe penalty must be 5.0, got {cfg.environment.collapse_penalty}")
if cfg.dataset.transitions != 4000 or cfg.dataset.episode_length != 25:
    raise SystemExit("registered collection target must be 4000 rows with 25-step episodes")

resolved = {
    "rows_per_arm": len(hidden),
    "cells_per_arm": len(hidden) // 5,
    "methods": sorted({row["method"] for row in hidden}),
    "filters": sorted({row["filter"] for row in hidden}),
    "reward_modes": sorted({row["reward_mode"] for row in hidden}),
    "collapse_penalty_safe": cfg.environment.collapse_penalty,
    "target_rows": cfg.dataset.transitions,
    "episode_length": cfg.dataset.episode_length,
    "planner": asdict(cfg.planner),
    "evaluation": asdict(cfg.evaluation),
}
(hidden_path.parent / "registration.json").write_text(
    json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(resolved, indent=2, sort_keys=True))
PY

echo "snapshot ready: $SNAP"
