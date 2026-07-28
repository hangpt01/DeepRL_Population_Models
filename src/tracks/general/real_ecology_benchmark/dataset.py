"""Trajectory-preserving public datasets and evaluator-only truth sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .privacy import forbidden_method_name


PUBLIC_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "dones",
    "episode_id",
    "timestep",
)

PUBLIC_SANITIZED_FIELDS = (
    "costs",
    "pop_ids",
    "terminated",
    "truncated",
)

PUBLIC_CONTROL_FIELDS = (
    "rho",
    "kappa",
    "K_eff",
    "next_rho",
    "next_kappa",
    "next_K_eff",
)

PRIVATE_FIELDS = (
    "states",
    "next_states",
    "r_base",
    "C",
    "theta",
    "regime",
    "next_regime",
    "entry",
    "reward_true",
    "initially_unsafe",
)

PRIVATE_OPTIONAL_FIELDS = (
    "r_eff_true",
    "safety_penalty_applied",
)


@dataclass
class TrajectoryDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray
    metadata: dict[str, Any]
    costs: np.ndarray | None = None
    pop_ids: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    action_costs: np.ndarray | None = None
    rho: np.ndarray | None = None
    kappa: np.ndarray | None = None
    K_eff: np.ndarray | None = None
    next_rho: np.ndarray | None = None
    next_kappa: np.ndarray | None = None
    next_K_eff: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def validate(self) -> None:
        lengths = {len(getattr(self, key)) for key in PUBLIC_FIELDS}
        if len(lengths) != 1:
            raise ValueError(f"public arrays have mismatched lengths: {lengths}")
        if len(self) == 0:
            raise ValueError("dataset is empty")
        if np.any(self.observations < 0) or np.any(self.next_observations < 0):
            raise ValueError("observations must be non-negative")
        if np.any(self.actions < 0):
            raise ValueError("actions must be non-negative")
        if not np.all(np.isfinite(self.rewards)):
            raise ValueError("rewards must be finite")
        for episode in np.unique(self.episode_id):
            idx = np.flatnonzero(self.episode_id == episode)
            expected = np.arange(len(idx), dtype=self.timestep.dtype)
            if not np.array_equal(self.timestep[idx], expected):
                raise ValueError(f"episode {episode} timesteps are not contiguous")
            if not bool(self.dones[idx[-1]]):
                raise ValueError(f"episode {episode} does not end with done=True")
        for key in PUBLIC_CONTROL_FIELDS:
            value = getattr(self, key)
            if value is not None and len(value) != len(self):
                raise ValueError(f"public control field {key} has wrong length")
        for key in PUBLIC_SANITIZED_FIELDS:
            value = getattr(self, key)
            if value is not None and len(value) != len(self):
                raise ValueError(f"public sanitized field {key} has wrong length")
        if self.metadata.get("expose_rk") == "hidden":
            missing = [key for key in PUBLIC_SANITIZED_FIELDS if getattr(self, key) is None]
            if missing:
                raise ValueError(f"hidden public dataset missing fields: {missing}")
            if self.action_costs is None or len(self.action_costs) == 0:
                raise ValueError("hidden public dataset requires action_costs")
            leaked_controls = [
                key for key in PUBLIC_CONTROL_FIELDS if getattr(self, key) is not None
            ]
            if leaked_controls:
                raise ValueError(
                    f"hidden public dataset exposes controls: {leaked_controls}"
                )
            forbidden = _forbidden_metadata_paths(self.metadata)
            if forbidden:
                raise ValueError(
                    "hidden public metadata exposes private fields: "
                    + ", ".join(forbidden)
                )
        if self.action_costs is not None:
            if np.any(~np.isfinite(self.action_costs)):
                raise ValueError("action_costs must be finite")
            if len(self.action_costs) <= int(np.max(self.actions)):
                raise ValueError("action_costs do not cover every observed action")
            if self.costs is not None and not np.allclose(
                self.costs, self.action_costs[self.actions.astype(int)], atol=0.0, rtol=0.0
            ):
                raise ValueError("per-row costs do not match action_costs")
        if self.terminated is not None and self.truncated is not None:
            terminated = self.terminated.astype(bool)
            truncated = self.truncated.astype(bool)
            if np.any(terminated & truncated):
                raise ValueError("terminated and truncated must be mutually exclusive")
            if not np.array_equal(self.dones.astype(bool), terminated | truncated):
                raise ValueError("dones must equal terminated | truncated")

    @property
    def num_episodes(self) -> int:
        return int(len(np.unique(self.episode_id)))

    def episode_indices(self) -> list[np.ndarray]:
        return [np.flatnonzero(self.episode_id == e) for e in np.unique(self.episode_id)]

    def split_by_episode(self, fraction: float = 0.8, seed: int = 0):
        if not 0 < fraction < 1:
            raise ValueError("fraction must be in (0,1)")
        episodes = np.unique(self.episode_id)
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        cut = max(1, min(len(episodes) - 1, int(round(fraction * len(episodes)))))
        return self.subset(np.isin(self.episode_id, episodes[:cut])), self.subset(
            np.isin(self.episode_id, episodes[cut:])
        )

    def subset(self, mask: np.ndarray) -> "TrajectoryDataset":
        mask = np.asarray(mask, dtype=bool)
        old_episode = self.episode_id[mask]
        unique = np.unique(old_episode)
        mapping = {int(e): i for i, e in enumerate(unique)}
        remapped = np.asarray([mapping[int(e)] for e in old_episode], dtype=np.int32)
        result = TrajectoryDataset(
            observations=self.observations[mask],
            actions=self.actions[mask],
            rewards=self.rewards[mask],
            next_observations=self.next_observations[mask],
            dones=self.dones[mask],
            episode_id=remapped,
            timestep=self.timestep[mask],
            metadata={**self.metadata, "subset": True},
            costs=None if self.costs is None else self.costs[mask],
            pop_ids=None if self.pop_ids is None else self.pop_ids[mask],
            terminated=None if self.terminated is None else self.terminated[mask],
            truncated=None if self.truncated is None else self.truncated[mask],
            action_costs=None if self.action_costs is None else self.action_costs.copy(),
            **{
                key: getattr(self, key)[mask] if getattr(self, key) is not None else None
                for key in PUBLIC_CONTROL_FIELDS
            },
        )
        result.validate()
        return result


@dataclass
class PrivateTrajectoryData:
    states: np.ndarray
    next_states: np.ndarray
    r_base: np.ndarray
    C: np.ndarray
    theta: np.ndarray
    regime: np.ndarray
    next_regime: np.ndarray
    entry: np.ndarray
    reward_true: np.ndarray
    initially_unsafe: np.ndarray
    metadata: dict[str, Any]
    r_eff_true: np.ndarray | None = None
    safety_penalty_applied: np.ndarray | None = None

    def validate(self, public: TrajectoryDataset | None = None) -> None:
        lengths = {len(getattr(self, key)) for key in PRIVATE_FIELDS}
        if len(lengths) != 1:
            raise ValueError("private arrays have mismatched lengths")
        if public is not None and len(self.states) != len(public):
            raise ValueError("public/private transition count mismatch")
        for key in PRIVATE_OPTIONAL_FIELDS:
            value = getattr(self, key)
            if value is not None and len(value) != len(self.states):
                raise ValueError(f"private optional field {key} has wrong length")


def save_public(path: str | Path, dataset: TrajectoryDataset) -> None:
    dataset.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: getattr(dataset, key) for key in PUBLIC_FIELDS}
    for key in PUBLIC_CONTROL_FIELDS:
        value = getattr(dataset, key)
        if value is not None:
            payload[key] = value
    for key in PUBLIC_SANITIZED_FIELDS:
        value = getattr(dataset, key)
        if value is not None:
            payload[key] = value
    if dataset.action_costs is not None:
        payload["action_costs"] = dataset.action_costs
    payload["metadata_json"] = np.asarray(json.dumps(dataset.metadata, sort_keys=True))
    np.savez_compressed(target, **payload)


def load_public(path: str | Path) -> TrajectoryDataset:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        result = TrajectoryDataset(
            **{key: np.asarray(data[key]) for key in PUBLIC_FIELDS},
            metadata=metadata,
            **{
                key: np.asarray(data[key]) if key in data.files else None
                for key in PUBLIC_SANITIZED_FIELDS
            },
            action_costs=(
                np.asarray(data["action_costs"])
                if "action_costs" in data.files else None
            ),
            **{
                key: np.asarray(data[key]) if key in data.files else None
                for key in PUBLIC_CONTROL_FIELDS
            },
        )
    result.validate()
    return result


def save_private(path: str | Path, private: PrivateTrajectoryData) -> None:
    private.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: getattr(private, key) for key in PRIVATE_FIELDS}
    for key in PRIVATE_OPTIONAL_FIELDS:
        value = getattr(private, key)
        if value is not None:
            payload[key] = value
    payload["metadata_json"] = np.asarray(json.dumps(private.metadata, sort_keys=True))
    np.savez_compressed(target, **payload)


def load_private(path: str | Path) -> PrivateTrajectoryData:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        result = PrivateTrajectoryData(
            **{key: np.asarray(data[key]) for key in PRIVATE_FIELDS},
            metadata=metadata,
            **{
                key: np.asarray(data[key]) if key in data.files else None
                for key in PRIVATE_OPTIONAL_FIELDS
            },
        )
    result.validate()
    return result


def assert_public_schema(path: str | Path) -> None:
    with np.load(Path(path), allow_pickle=False) as data:
        keys = set(data.files)
        metadata = json.loads(str(data["metadata_json"].item()))
    forbidden = (set(PRIVATE_FIELDS) | set(PRIVATE_OPTIONAL_FIELDS)) & keys
    if forbidden:
        raise AssertionError(f"truth leaked into public dataset: {sorted(forbidden)}")
    missing = set(PUBLIC_FIELDS) - keys
    if missing:
        raise AssertionError(f"public dataset missing fields: {sorted(missing)}")
    if metadata.get("expose_rk") == "hidden":
        required = set(PUBLIC_SANITIZED_FIELDS) | {"action_costs"}
        missing_hidden = required - keys
        if missing_hidden:
            raise AssertionError(
                f"hidden public dataset missing fields: {sorted(missing_hidden)}"
            )
        leaked_controls = set(PUBLIC_CONTROL_FIELDS) & keys
        if leaked_controls:
            raise AssertionError(
                f"hidden public dataset exposes controls: {sorted(leaked_controls)}"
            )
        forbidden_metadata = _forbidden_metadata_paths(metadata)
        if forbidden_metadata:
            raise AssertionError(
                f"hidden metadata leaked private fields: {forbidden_metadata}"
            )


def _forbidden_metadata_paths(metadata: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if forbidden_method_name(str(key)) or str(key).casefold() == "action_table_hash":
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(metadata, "")
    return sorted(set(found))


def dataset_sha256(dataset: TrajectoryDataset) -> str:
    """Stable content hash for public offline data and non-hash metadata."""

    h = hashlib.sha256()
    for key in (*PUBLIC_FIELDS, *PUBLIC_SANITIZED_FIELDS, *PUBLIC_CONTROL_FIELDS, "action_costs"):
        value = getattr(dataset, key, None)
        if value is None:
            continue
        arr = np.ascontiguousarray(value)
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    metadata = dict(dataset.metadata)
    metadata.pop("dataset_sha256", None)
    h.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
