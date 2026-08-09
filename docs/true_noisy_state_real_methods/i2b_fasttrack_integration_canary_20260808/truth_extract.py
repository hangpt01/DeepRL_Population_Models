"""Allowlisted I2B truth extractor.

Only metadata_json, states, and next_states payload members are readable from the
private archive.  Public episode_id and timestep are read from immutable public.npz.
No runtime future-state artifact is produced.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


TRUTH_KEYS = (
    "C", "entry", "initially_unsafe", "metadata_json", "next_regime",
    "next_states", "r_base", "r_eff_true", "regime", "reward_true",
    "safety_penalty_applied", "states", "theta",
)
ALLOWED_PAYLOADS = frozenset({"metadata_json", "states", "next_states"})
FORBIDDEN_PAYLOADS = frozenset(TRUTH_KEYS) - ALLOWED_PAYLOADS


class ExtractionViolation(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_member(archive: zipfile.ZipFile, field: str) -> np.ndarray:
    if field not in ALLOWED_PAYLOADS and field not in {"episode_id", "timestep"}:
        raise ExtractionViolation(f"payload access denied: {field}")
    member = f"{field}.npy"
    if member not in archive.namelist():
        raise ExtractionViolation(f"missing archive member: {member}")
    with archive.open(member, "r") as stream:
        payload = stream.read()
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ExtractionViolation("member did not decode to ndarray")
    return value


def extract_cell(
    *,
    cell: str,
    truth_path: Path,
    public_path: Path,
    expected_truth_sha256: str,
    expected_public_file_sha256: str,
    expected_public_dataset_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise ExtractionViolation(f"refusing to overwrite {output_path}")
    if sha256_file(truth_path) != expected_truth_sha256:
        raise ExtractionViolation("truth archive SHA-256 mismatch")
    if sha256_file(public_path) != expected_public_file_sha256:
        raise ExtractionViolation("public archive file SHA-256 mismatch")

    materialized_truth_fields: list[str] = []
    with zipfile.ZipFile(truth_path, "r") as archive:
        names = tuple(name.removesuffix(".npy") for name in archive.namelist())
        # I1 freezes the exact schema, not ZIP member insertion order.  Both
        # registered archives use their original (non-alphabetical) insertion
        # order, so validate cardinality and exact set without reordering or
        # indexing a forbidden payload.
        if len(names) != len(TRUTH_KEYS) or set(names) != set(TRUTH_KEYS):
            raise ExtractionViolation("truth key set/cardinality differs from the registered schema")
        metadata = _read_member(archive, "metadata_json")
        materialized_truth_fields.append("metadata_json")
        try:
            parsed = json.loads(str(metadata.item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ExtractionViolation("metadata_json is not valid scalar JSON") from exc
        binding = parsed.get("public_dataset_sha256")
        if binding != expected_public_dataset_sha256:
            raise ExtractionViolation("private/public dataset binding mismatch")
        states = _read_member(archive, "states")
        materialized_truth_fields.append("states")
        next_states = _read_member(archive, "next_states")
        materialized_truth_fields.append("next_states")

    with zipfile.ZipFile(public_path, "r") as public_archive:
        episode_id = _read_member(public_archive, "episode_id")
        timestep = _read_member(public_archive, "timestep")

    arrays = {"states": states, "next_states": next_states}
    for name, value in arrays.items():
        if value.dtype != np.dtype("float64") or value.shape != (4000,):
            raise ExtractionViolation(f"{name} dtype/shape mismatch")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ExtractionViolation(f"{name} is nonfinite or negative")
    if episode_id.dtype != np.dtype("int32") or timestep.dtype != np.dtype("int16"):
        raise ExtractionViolation("public identity dtype mismatch")
    if episode_id.shape != (4000,) or timestep.shape != (4000,):
        raise ExtractionViolation("public identity shape mismatch")
    unique, counts = np.unique(episode_id, return_counts=True)
    if unique.size != 160 or not np.all(counts == 25):
        raise ExtractionViolation("public episodes are not exactly 160 x 25")
    for episode in unique:
        rows = np.flatnonzero(episode_id == episode)
        if not np.array_equal(timestep[rows], np.arange(25, dtype=np.int16)):
            raise ExtractionViolation("episode timestep/order mismatch")
        if not np.array_equal(next_states[rows[:-1]], states[rows[1:]]):
            raise ExtractionViolation("next_states[t] != states[t+1] within episode")

    row_index = np.arange(4000, dtype=np.int32)
    np.savez(
        output_path,
        states=states,
        next_states=next_states,
        row_index=row_index,
        episode_id=episode_id,
        timestep=timestep,
    )
    with np.load(output_path, allow_pickle=False) as derived:
        emitted_keys = tuple(derived.files)
    if emitted_keys != ("states", "next_states", "row_index", "episode_id", "timestep"):
        raise ExtractionViolation("derived artifact schema mismatch")
    return {
        "cell": cell,
        "truth_path": str(truth_path),
        "truth_sha256": expected_truth_sha256,
        "public_path": str(public_path),
        "public_file_sha256": expected_public_file_sha256,
        "public_dataset_sha256": expected_public_dataset_sha256,
        "truth_payload_fields_materialized": materialized_truth_fields,
        "forbidden_truth_payload_fields_materialized": [],
        "forbidden_truth_payload_bytes_read": 0,
        "metadata_keys_returned": ["public_dataset_sha256"],
        "metadata_entered_derived_artifact": False,
        "derived_path": str(output_path),
        "derived_sha256": sha256_file(output_path),
        "derived_keys": list(emitted_keys),
        "rows": 4000,
        "episodes": 160,
        "episode_length": 25,
        "units": "raw_abundance",
        "alignment": "PASS_ZERO_TOLERANCE",
        "runtime_artifact_created": False,
    }
