"""Tolerant loader for diagnostic-replay logs.

The replay writes .npz + .csv.gz (no pyarrow in the run venv).  The exact
serialisation of the vector fields is not pinned by the spec, so this loader
accepts several encodings and normalises them into one in-memory record.
Missing fields are reported, never guessed.

Canonical in-memory form: a ReplayLog with
  .frame   pandas DataFrame, one row per (seed, t), scalar columns
  .vectors dict name -> ndarray with leading axis aligned to .frame rows
"""
from __future__ import annotations

import glob
import gzip
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCALAR_FIELDS = [
    "seed", "t", "x_true_t", "x_true_next", "obs_t", "action_t", "cost_t",
    "utility_t", "penalty_flag_t", "reward_t", "r_setpoint_t", "r_pos_t",
    "r_mort_t", "k_t", "m_t", "regime_z_t", "regime_switched_t",
    "allee_C_episode", "theta_exponent_episode", "surrogate_reward_t",
]
PLUS_SCALARS = ["argmax_weighted", "argmax_map_only", "argmax_uniform",
                "margin_top1_top2"]
MOOR_SCALARS = ["argmax", "margin_top1_top2", "belief_mean", "belief_entropy"]

VECTOR_SPECS = {
    "w_t": r"^w_(\d+)$",
    "q_weighted": r"^q_weighted_(\d+)$",
    "argmax_cand": r"^argmax_cand_(\d+)$",
    "belief_mean_cand": r"^belief_mean_(\d+)$",
    "belief_entropy_cand": r"^belief_entropy_(\d+)$",
    "true_reward_all_actions": r"^true_reward_a(\d+)$",
    "q": r"^q_(\d+)$",
    "qbar": r"^qbar_(\d+)$",
    "qvar": r"^qvar_(\d+)$",
    "w_m": r"^w_m_(\d+)$",
    "root_visits": r"^root_visits_(\d+)$",
}
MATRIX_SPECS = {"q_cand": r"^q_cand_(\d+)_(\d+)$"}   # [candidate, action]


@dataclass
class ReplayLog:
    cell_id: str
    method: str
    frame: pd.DataFrame
    vectors: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    source: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def seeds(self) -> np.ndarray:
        return np.unique(self.frame["seed"].to_numpy())

    def episode_slices(self):
        """Yield (seed, boolean mask) in seed order."""
        s = self.frame["seed"].to_numpy()
        for seed in self.seeds:
            yield seed, (s == seed)

    def has(self, *names: str) -> bool:
        return all(n in self.frame.columns or n in self.vectors for n in names)

    def require(self, *names: str):
        absent = [n for n in names
                  if n not in self.frame.columns and n not in self.vectors]
        if absent:
            raise KeyError(
                f"{self.cell_id}/{self.method}: missing {absent}. "
                f"Present scalars={sorted(self.frame.columns)[:12]}..., "
                f"vectors={sorted(self.vectors)}")


def _collapse_vectors(df: pd.DataFrame):
    """Pull wide indexed columns (w_0..w_7) into stacked ndarrays."""
    vectors, consumed = {}, set()
    for name, pattern in VECTOR_SPECS.items():
        idx = {}
        for col in df.columns:
            m = re.match(pattern, col)
            if m:
                idx[int(m.group(1))] = col
        if idx:
            order = [idx[i] for i in sorted(idx)]
            vectors[name] = df[order].to_numpy(dtype=float)
            consumed.update(order)
    for name, pattern in MATRIX_SPECS.items():
        cells = {}
        for col in df.columns:
            m = re.match(pattern, col)
            if m:
                cells[(int(m.group(1)), int(m.group(2)))] = col
        if cells:
            n_i = max(k[0] for k in cells) + 1
            n_j = max(k[1] for k in cells) + 1
            arr = np.full((len(df), n_i, n_j), np.nan)
            for (i, j), col in cells.items():
                arr[:, i, j] = df[col].to_numpy(dtype=float)
            vectors[name] = arr
            consumed.update(cells.values())
    return vectors, consumed


def load_log(path: str, cell_id: str | None = None,
             method: str | None = None) -> ReplayLog:
    """Load one (cell, method) log from .npz, .csv.gz or .csv."""
    base = os.path.basename(path)
    if cell_id is None or method is None:
        stem = re.sub(r"\.(npz|csv\.gz|csv)$", "", base)
        parts = stem.split("__")
        cell_id = cell_id or parts[0]
        method = method or (parts[1] if len(parts) > 1 else "unknown")
    method = method.lower()

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        scalars, vectors = {}, {}
        for k in keys:
            arr = z[k]
            if arr.ndim == 1:
                scalars[k] = arr
            else:
                vectors[k] = arr
        df = pd.DataFrame(scalars)
    else:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            df = pd.read_csv(fh)
        vectors, consumed = _collapse_vectors(df)
        df = df.drop(columns=[c for c in consumed if c in df.columns])

    expected = SCALAR_FIELDS + (PLUS_SCALARS if method == "plus" else
                                MOOR_SCALARS if method == "moor" else [])
    missing = [c for c in expected if c not in df.columns and c not in vectors]
    if "t" in df.columns and "seed" in df.columns:
        df = df.sort_values(["seed", "t"], kind="stable").reset_index(drop=True)
    return ReplayLog(cell_id, method, df, vectors, missing, path)


def load_run(root: str) -> dict[tuple[str, str], ReplayLog]:
    """Load every log under {root}/logs/."""
    out = {}
    for pat in ("*.npz", "*.csv.gz", "*.csv"):
        for p in sorted(glob.glob(os.path.join(root, "logs", pat))):
            log = load_log(p)
            out[(log.cell_id, log.method)] = log
    return out


def schema_report(logs: dict[tuple[str, str], ReplayLog]) -> pd.DataFrame:
    """One row per log: what arrived, what is missing.  Run this first."""
    rows = []
    for (cell, method), log in sorted(logs.items()):
        rows.append(dict(
            cell_id=cell, method=method, rows=log.n_rows,
            episodes=len(log.seeds),
            steps_per_episode=(log.n_rows // max(len(log.seeds), 1)),
            vectors=",".join(sorted(log.vectors)) or "-",
            missing=",".join(log.missing) or "-",
        ))
    return pd.DataFrame(rows)
