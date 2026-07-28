"""Training/holdout diagnostics saved beside each method result."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .beliefs import BeliefCache
from .config import BenchmarkConfig, TrainingConfig
from .dataset import PUBLIC_CONTROL_FIELDS, TrajectoryDataset


@dataclass
class TrainingHistory:
    """Long-form training metric table.

    Rows are intentionally generic because the benchmark mixes closed-form
    baselines, grid searches, fitted dynamics, and iterative offline-RL methods.
    """

    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, object]] = field(default_factory=list)

    def log(
        self,
        step: int,
        split: str,
        metrics: dict[str, float | int | np.floating | None],
        phase: str = "fit",
    ) -> None:
        for metric, value in metrics.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            self.rows.append(
                {
                    "phase": phase,
                    "step": int(step),
                    "split": str(split),
                    "metric": str(metric),
                    "value": numeric,
                }
            )

    def last_values(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for row in self.rows:
            key = f"training_{row['split']}_{row['metric']}"
            result[key] = float(row["value"])
        return result


def _subset_cache(cache: BeliefCache, mask: np.ndarray) -> BeliefCache:
    mask = np.asarray(mask, dtype=bool)
    return BeliefCache(
        features=cache.features[mask],
        next_features=cache.next_features[mask],
        mean_states=cache.mean_states[mask],
        next_mean_states=cache.next_mean_states[mask],
        metadata={**cache.metadata, "subset": True},
        **{
            key: getattr(cache, key)[mask] if getattr(cache, key) is not None else None
            for key in PUBLIC_CONTROL_FIELDS
        },
    )


def split_train_holdout(
    dataset: TrajectoryDataset,
    cache: BeliefCache,
    cfg: BenchmarkConfig,
) -> tuple[TrajectoryDataset, BeliefCache, TrajectoryDataset | None, BeliefCache | None, dict[str, object]]:
    """Episode-disjoint train/holdout split aligned with cached beliefs."""

    training = cfg.training
    if (
        not training.enabled
        or training.holdout_fraction <= 0.0
        or dataset.num_episodes < 2
    ):
        info = {
            "training_enabled": bool(training.enabled),
            "holdout_fraction": float(training.holdout_fraction),
            "train_transitions": len(dataset),
            "holdout_transitions": 0,
            "train_episodes": dataset.num_episodes,
            "holdout_episodes": 0,
        }
        return dataset, cache, None, None, info

    episodes = np.unique(dataset.episode_id)
    rng = np.random.default_rng(cfg.seed + training.split_seed_offset)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(training.holdout_fraction * len(shuffled))))
    holdout_count = min(holdout_count, len(shuffled) - 1)
    holdout_episodes = shuffled[:holdout_count]
    train_mask = ~np.isin(dataset.episode_id, holdout_episodes)
    holdout_mask = ~train_mask
    train = dataset.subset(train_mask)
    holdout = dataset.subset(holdout_mask)
    train_cache = _subset_cache(cache, train_mask)
    holdout_cache = _subset_cache(cache, holdout_mask)
    info = {
        "training_enabled": True,
        "holdout_fraction": float(training.holdout_fraction),
        "train_transitions": len(train),
        "holdout_transitions": len(holdout),
        "train_episodes": train.num_episodes,
        "holdout_episodes": holdout.num_episodes,
        "holdout_original_episode_ids": [int(x) for x in sorted(holdout_episodes)],
    }
    return train, train_cache, holdout, holdout_cache, info


def attach_training_history(
    policy: Any,
    method: str,
    cfg: BenchmarkConfig,
    split_info: dict[str, object],
    holdout_dataset: TrajectoryDataset | None,
    holdout_cache: BeliefCache | None,
) -> TrainingHistory | None:
    if not cfg.training.enabled:
        return None
    history = TrainingHistory(
        method=method,
        metadata={
            "method": method,
            "seed": cfg.seed,
            "population": cfg.environment.population,
            "environment": cfg.environment.kind,
            "reward_mode": cfg.environment.reward_mode,
            "sigma_obs": cfg.environment.observation_noise_sigma,
            **split_info,
        },
    )
    policy.training_history = history
    policy.training_holdout_dataset = holdout_dataset
    policy.training_holdout_beliefs = holdout_cache
    return history


def _log_prediction_mse(policy: Any, dataset: TrajectoryDataset, cache: BeliefCache) -> dict[str, float]:
    target_log = np.log1p(np.maximum(cache.next_mean_states, 0.0) / policy.env_cfg.K_ref)
    prediction = None
    if hasattr(policy, "dynamics"):
        prediction = policy.dynamics.predict(
            cache.mean_states, dataset.actions, cache.rho, cache.kappa
        )[0]
    elif getattr(policy, "name", "") == "moor" and hasattr(policy, "_predict"):
        prediction = policy._predict(
            cache.mean_states,
            dataset.actions,
            policy.r_hat,
            policy.K_hat,
            cache.rho,
            cache.kappa,
        )
    if prediction is None:
        return {}
    pred_log = np.log1p(np.maximum(prediction, 0.0) / policy.env_cfg.K_ref)
    mse = float(np.mean((pred_log - target_log) ** 2))
    rmse = float(np.sqrt(np.mean((np.maximum(prediction, 0.0) - cache.next_mean_states) ** 2)))
    return {"dynamics_log_mse": mse, "dynamics_rmse": rmse}


def _log_q_bellman_mse(policy: Any, dataset: TrajectoryDataset, cache: BeliefCache) -> dict[str, float]:
    q_weights = getattr(policy, "q_weights", None)
    if q_weights is None:
        return {}
    try:
        from .methods.delphic import _augment
    except Exception:
        return {}
    X = _augment(cache.features)
    X_next = _augment(cache.next_features)
    if q_weights.shape[1] != X.shape[1]:
        return {}
    q_current = X @ q_weights.T
    q_next = X_next @ q_weights.T
    target = dataset.rewards + policy.planner_cfg.discount * (~dataset.dones) * np.max(
        q_next, axis=1
    )
    observed = q_current[np.arange(len(dataset)), dataset.actions.astype(int)]
    return {"q_bellman_mse": float(np.mean((observed - target) ** 2))}


def record_final_fit_metrics(
    policy: Any,
    train_dataset: TrajectoryDataset,
    train_cache: BeliefCache,
    holdout_dataset: TrajectoryDataset | None,
    holdout_cache: BeliefCache | None,
) -> None:
    history: TrainingHistory | None = getattr(policy, "training_history", None)
    if history is None:
        return
    for split, dataset, cache in (
        ("train", train_dataset, train_cache),
        ("holdout", holdout_dataset, holdout_cache),
    ):
        if dataset is None or cache is None:
            continue
        metrics = {}
        metrics.update(_log_prediction_mse(policy, dataset, cache))
        metrics.update(_log_q_bellman_mse(policy, dataset, cache))
        if metrics:
            history.log(step=-1, split=split, metrics=metrics, phase="final")


def _write_csv(history: TrainingHistory, path: Path) -> None:
    fieldnames = ["phase", "step", "split", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history.rows)


def _write_plot(history: TrainingHistory, path: Path) -> str:
    try:
        mpl_config = path.parent / ".matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot skipped: {exc!r}"
    if not history.rows:
        return "plot skipped: no rows"
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in history.rows:
        metric = str(row["metric"])
        if not any(token in metric for token in ("loss", "mse", "rmse", "objective", "nll")):
            continue
        key = (str(row["split"]), metric)
        grouped.setdefault(key, []).append((int(row["step"]), float(row["value"])))
    if not grouped:
        return "plot skipped: no loss-like metrics"
    fig, ax = plt.subplots(figsize=(9, 5))
    for (split, metric), values in sorted(grouped.items()):
        values = sorted(values)
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=f"{split}:{metric}")
    ax.set_xlabel("training step")
    ax.set_ylabel("metric value")
    ax.set_title(f"{history.method} training diagnostics")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return "ok"


def _log_wandb(history: TrainingHistory, cfg: TrainingConfig, output_dir: Path) -> str:
    if not cfg.wandb:
        return "disabled"
    try:
        import wandb
    except Exception as exc:
        return f"wandb skipped: {exc!r}"
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_run_name,
        config=history.metadata,
        reinit=True,
    )
    try:
        for row in history.rows:
            key = f"training/{row['split']}/{row['metric']}"
            wandb.log({key: row["value"], "training/step": row["step"]}, step=int(row["step"]))
        artifact = wandb.Artifact(
            name=f"{history.method}-training-{run.id}",
            type="training-diagnostics",
        )
        for filename in ("training_history.csv", "training_history.json", "training_history.png"):
            path = output_dir / filename
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
    finally:
        run.finish()
    return "ok"


def save_training_artifacts(
    history: TrainingHistory | None,
    output_dir: str | Path,
    cfg: TrainingConfig,
) -> dict[str, object]:
    if history is None:
        return {"training_history_rows": 0}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "training_history.csv"
    json_path = target / "training_history.json"
    plot_path = target / "training_history.png"
    _write_csv(history, csv_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": history.metadata, "rows": history.rows, "config": asdict(cfg)},
            handle,
            indent=2,
            sort_keys=True,
        )
    plot_status = _write_plot(history, plot_path) if cfg.plot else "disabled"
    wandb_status = _log_wandb(history, cfg, target)
    summary: dict[str, object] = {
        "training_history_rows": len(history.rows),
        "training_history_csv": str(csv_path),
        "training_history_json": str(json_path),
        "training_plot_status": plot_status,
        "training_wandb_status": wandb_status,
    }
    if plot_path.exists():
        summary["training_history_plot"] = str(plot_path)
    summary.update(history.last_values())
    return summary
