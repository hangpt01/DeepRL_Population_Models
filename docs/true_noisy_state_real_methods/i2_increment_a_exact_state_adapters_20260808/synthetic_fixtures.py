"""Deterministic synthetic fixtures; no real array or accepted dataset is read."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from information_boundary import REGISTERED_KEY_ORDER, REGISTERED_ROW_COUNT, SyntheticProvenance


SYNTHETIC_BINDING = "synthetic_public_dataset_sha256_" + "a" * 64
SYNTHETIC_PROVENANCE = SyntheticProvenance(generator="synthetic_fixtures.py:v1")


def make_base_public_features(rows: int = 16) -> np.ndarray:
    index = np.arange(rows, dtype=np.float64)
    features = np.empty((rows, 10), dtype=np.float64)
    features[:, 0] = 0.2 + index * 0.03
    features[:, 1] = 0.05 + index * 0.001
    features[:, 2] = features[:, 0] - 0.1
    features[:, 3] = features[:, 0]
    features[:, 4] = features[:, 0] + 0.1
    features[:, 5] = 0.0
    features[:, 6] = 10.0 + index
    features[:, 7] = 11.0 + index
    features[:, 8] = index
    features[:, 9] = 1.0
    return features


def make_exact_abundance(rows: int = 16) -> np.ndarray:
    return np.linspace(1.0, 100.0, rows, dtype=np.float64)


def reduction_based_sd_residue() -> np.float64:
    particles = np.repeat(np.log1p(np.float64(359.17) / np.float64(20.0)), 128)
    weights = np.exp(np.full(128, -np.log(128), dtype=np.float64))
    weights /= np.sum(weights)
    mean = np.sum(weights * particles)
    variance = np.sum(weights * (particles - mean) ** 2)
    return np.sqrt(max(np.float64(variance), np.float64(0.0)))


def make_synthetic_truth_archive() -> OrderedDict[str, np.ndarray]:
    row = np.arange(REGISTERED_ROW_COUNT, dtype=np.float64)
    archive: OrderedDict[str, np.ndarray] = OrderedDict()
    archive["C"] = 0.1 + row * 0.0
    archive["entry"] = np.zeros(REGISTERED_ROW_COUNT, dtype=bool)
    archive["initially_unsafe"] = np.zeros(REGISTERED_ROW_COUNT, dtype=bool)
    archive["metadata_json"] = np.asarray(
        json.dumps({"public_dataset_sha256": SYNTHETIC_BINDING}, sort_keys=True)
    )
    archive["next_regime"] = np.zeros(REGISTERED_ROW_COUNT, dtype=np.int8)
    archive["next_states"] = 2.0 + row * 0.01
    archive["r_base"] = 0.3 + row * 0.0
    archive["r_eff_true"] = 0.25 + row * 0.0
    archive["regime"] = np.zeros(REGISTERED_ROW_COUNT, dtype=np.int8)
    archive["reward_true"] = row * 0.001
    archive["safety_penalty_applied"] = np.zeros(REGISTERED_ROW_COUNT, dtype=bool)
    archive["states"] = 1.0 + row * 0.01
    archive["theta"] = 0.5 + row * 0.0
    assert tuple(archive) == REGISTERED_KEY_ORDER
    return archive


def synthetic_units() -> dict[str, str]:
    return {"states": "raw_abundance", "next_states": "raw_abundance"}


def synthetic_row_tokens() -> dict[str, str]:
    return {key: "synthetic_rows_0000_3999" for key in REGISTERED_KEY_ORDER if key != "metadata_json"}


def write_synthetic_npz(path: Path) -> None:
    archive = make_synthetic_truth_archive()
    np.savez(path, **archive)
