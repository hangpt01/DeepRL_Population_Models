"""A small, readable MOBILE implementation for learning—not benchmark use.

MOBILE is "MOdel-Bellman Inconsistency penalized offLinE policy optimization"
(Sun et al., ICML 2023).  This file follows the paper's Algorithm 1 and
Equations (10)--(14), using Pendulum-v1 so every conceptual piece fits in one
file.

The complete flow taught here is:

    frozen offline data
      -> probabilistic dynamics ensemble
      -> short imagined rollouts
      -> mixed real/model SAC batches
      -> Model-Bellman Inconsistency penalty on model targets only
      -> actor/critic updates
      -> evaluation in the real environment (never added to training data)

Offline RL means that policy learning uses a fixed dataset and does not collect
new real transitions.  Model-based offline RL first learns a simulator from
that fixed data, then uses the learned simulator to augment policy training.
Those synthetic transitions improve coverage, but model errors can be exploited
by the policy.  MOBILE addresses that danger with a value-aware penalty.

This is deliberately educational.  It is not imported by, registered with, or
claimed to reproduce the repository's research framework.

Dependencies:
    pip install torch gymnasium

The repository already lists PyTorch as an optional paper-faithful dependency;
Gymnasium is needed only by this standalone example.
"""

# -----------------------------------------------------------------------------
# 2. Imports and dependency notes
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import math
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# 3. Small configuration / CLI arguments
# -----------------------------------------------------------------------------


@dataclass
class Config:
    """Tiny CPU-oriented defaults; these are not paper reproduction settings."""

    seed: int = 116
    env_id: str = "Pendulum-v1"
    num_offline_transitions: int = 2_000

    dynamics_ensemble_size: int = 3
    dynamics_train_epochs: int = 5
    dynamics_batch_size: int = 128
    dynamics_lr: float = 1e-3

    rollout_horizon: int = 1
    rollout_batch_size: int = 256
    rollout_interval: int = 10
    model_buffer_capacity: int = 20_000

    real_ratio: float = 0.20
    batch_size: int = 128
    num_policy_updates: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    beta: float = 0.5
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 64

    eval_episodes: int = 2
    eval_interval: int = 50
    log_interval: int = 10
    device: str = "cpu"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Minimal educational MOBILE on a frozen Pendulum dataset."
    )
    defaults = Config()
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        arg_type = type(default)
        parser.add_argument(f"--{field.name}", type=arg_type, default=default)
    cfg = Config(**vars(parser.parse_args()))

    if cfg.dynamics_ensemble_size < 2:
        raise ValueError("MOBILE uncertainty requires at least two dynamics models")
    if not 0.0 <= cfg.real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1]")
    for name in (
        "num_offline_transitions",
        "dynamics_train_epochs",
        "dynamics_batch_size",
        "rollout_horizon",
        "rollout_batch_size",
        "rollout_interval",
        "model_buffer_capacity",
        "batch_size",
        "num_policy_updates",
        "eval_episodes",
        "eval_interval",
        "log_interval",
    ):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    return cfg


# -----------------------------------------------------------------------------
# 4. Seeding
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch; environments receive explicit seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


# -----------------------------------------------------------------------------
# 5. Offline dataset creation/loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineDataset:
    """The five standard transition fields plus episode IDs for safe splitting."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    def subset(self, indices: np.ndarray) -> "OfflineDataset":
        return OfflineDataset(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            episode_ids=self.episode_ids[indices],
        )


def check_continuous_env(env: gym.Env) -> tuple[int, int, np.ndarray, np.ndarray]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("this example requires a continuous Box observation space")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("this example requires a continuous Box action space")
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError("this example supports vector observations/actions only")
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("finite action bounds are required for the tanh actor")
    return env.observation_space.shape[0], env.action_space.shape[0], low, high


def pendulum_behavior_action(
    observation: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A noisy behavior policy used once to create the offline Pendulum data.

    Most actions come from a simple upright controller and some are uniform
    random actions.  This produces a small, imperfect dataset with more useful
    coverage than a purely random controller.  The learned MOBILE policy never
    asks this function—or the real environment—for additional training data.
    """

    if rng.random() < 0.30:
        return rng.uniform(action_low, action_high).astype(np.float32)
    cos_theta, sin_theta, angular_velocity = observation
    theta = math.atan2(float(sin_theta), float(cos_theta))
    torque = -2.0 * theta - 0.5 * float(angular_velocity) + rng.normal(0.0, 0.35)
    return np.clip(np.asarray([torque], dtype=np.float32), action_low, action_high)


def create_frozen_offline_dataset(cfg: Config) -> OfflineDataset:
    """Interact with the real environment once, then freeze all transitions."""

    env = gym.make(cfg.env_id)
    obs_dim, action_dim, action_low, action_high = check_continuous_env(env)
    if cfg.env_id != "Pendulum-v1":
        print(
            "Note: non-Pendulum env selected; offline behavior falls back to "
            "uniform random actions."
        )
    rng = np.random.default_rng(cfg.seed + 1_000)
    rows: dict[str, list[np.ndarray | float | bool | int]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": [],
        "episode_ids": [],
    }

    episode = 0
    observation, _ = env.reset(seed=cfg.seed + 2_000)
    while len(rows["actions"]) < cfg.num_offline_transitions:
        if cfg.env_id == "Pendulum-v1" and obs_dim == 3 and action_dim == 1:
            action = pendulum_behavior_action(observation, action_low, action_high, rng)
        else:
            action = rng.uniform(action_low, action_high).astype(np.float32)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        rows["observations"].append(np.asarray(observation, dtype=np.float32))
        rows["actions"].append(np.asarray(action, dtype=np.float32))
        rows["rewards"].append(float(reward))
        rows["next_observations"].append(
            np.asarray(next_observation, dtype=np.float32)
        )
        rows["dones"].append(done)
        rows["episode_ids"].append(episode)

        if done:
            episode += 1
            observation, _ = env.reset(seed=cfg.seed + 2_000 + episode)
        else:
            observation = next_observation
    env.close()

    # Copies plus frozen=True make the intent explicit: no real rows are added
    # or changed during policy optimization.
    return OfflineDataset(
        observations=np.asarray(rows["observations"], dtype=np.float32),
        actions=np.asarray(rows["actions"], dtype=np.float32),
        rewards=np.asarray(rows["rewards"], dtype=np.float32).reshape(-1, 1),
        next_observations=np.asarray(rows["next_observations"], dtype=np.float32),
        dones=np.asarray(rows["dones"], dtype=np.float32).reshape(-1, 1),
        episode_ids=np.asarray(rows["episode_ids"], dtype=np.int64),
    )


# -----------------------------------------------------------------------------
# 6. Real replay buffer and model replay buffer
# -----------------------------------------------------------------------------


class ReplayBuffer:
    """A minimal NumPy ring buffer with PyTorch batch conversion.

    ``D_real`` is initialized exactly once and never modified.  ``D_model`` is
    capped and overwritten in FIFO order as new imagined transitions arrive.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_offline_dataset(cls, dataset: OfflineDataset, seed: int) -> "ReplayBuffer":
        buffer = cls(
            len(dataset), dataset.observations.shape[1], dataset.actions.shape[1], seed
        )
        buffer.add_batch(
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.next_observations,
            dataset.dones,
        )
        return buffer

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for row in range(len(actions)):
            index = self.position
            self.observations[index] = observations[row]
            self.actions[index] = actions[row]
            self.rewards[index] = rewards[row]
            self.next_observations[index] = next_observations[row]
            self.dones[index] = dones[row]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def sample_observations(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = self.rng.integers(0, self.size, size=batch_size)
        return torch.as_tensor(self.observations[indices], device=device)


# 7. Train/validation split for the dynamics model.  This is kept beside the
# dataset code because the episode IDs are used only for this split.
def split_dynamics_data(
    dataset: OfflineDataset, seed: int, validation_fraction: float = 0.2
) -> tuple[OfflineDataset, OfflineDataset]:
    """Prefer an episode-disjoint split, matching the repository's convention."""

    rng = np.random.default_rng(seed)
    episodes = np.unique(dataset.episode_ids)
    if len(episodes) >= 2:
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(validation_fraction * len(shuffled))))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation_mask = np.isin(dataset.episode_ids, shuffled[:validation_count])
    else:
        # Very small smoke datasets may contain only one episode.
        order = rng.permutation(len(dataset))
        validation_count = max(1, min(len(dataset) - 1, int(0.2 * len(dataset))))
        validation_mask = np.zeros(len(dataset), dtype=bool)
        validation_mask[order[:validation_count]] = True
    return dataset.subset(np.flatnonzero(~validation_mask)), dataset.subset(
        np.flatnonzero(validation_mask)
    )


def print_dataset_summary(
    dataset: OfflineDataset, train: OfflineDataset, validation: OfflineDataset
) -> None:
    rewards = dataset.rewards[:, 0]
    print("\nFrozen offline dataset")
    print(f"  size: {len(dataset)}")
    print(f"  observation dimension: {dataset.observations.shape[1]}")
    print(f"  action dimension: {dataset.actions.shape[1]}")
    print(
        "  reward mean/std/min/max: "
        f"{rewards.mean():.3f} / {rewards.std():.3f} / "
        f"{rewards.min():.3f} / {rewards.max():.3f}"
    )
    print(f"  done fraction: {dataset.dones.mean():.4f}")
    print(f"  dynamics train/validation sizes: {len(train)} / {len(validation)}")
    print("  dataset is now frozen; policy training will not add real transitions.\n")


# -----------------------------------------------------------------------------
# 8. Normalization using real train data only
# -----------------------------------------------------------------------------


class DynamicsNormalizer:
    """Statistics fitted only on the real dynamics-training split.

    Validation data, model rollouts, and evaluation observations never affect
    these statistics.  The model predicts normalized [delta_state, reward].
    """

    def __init__(self, train: OfflineDataset, device: torch.device):
        inputs = np.concatenate([train.observations, train.actions], axis=1)
        targets = np.concatenate(
            [train.next_observations - train.observations, train.rewards], axis=1
        )
        self.input_mean = torch.as_tensor(inputs.mean(0), device=device)
        self.input_std = torch.as_tensor(
            np.maximum(inputs.std(0), 1e-6), device=device
        )
        self.target_mean = torch.as_tensor(targets.mean(0), device=device)
        self.target_std = torch.as_tensor(
            np.maximum(targets.std(0), 1e-6), device=device
        )

    def normalize_input(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        return (inputs - self.input_mean) / self.input_std

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def denormalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return targets * self.target_std + self.target_mean


# -----------------------------------------------------------------------------
# 9. Probabilistic ensemble dynamics model
# -----------------------------------------------------------------------------


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ProbabilisticDynamicsModel(nn.Module):
    """Predict a diagonal Gaussian over [next_state - state, reward]."""

    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int):
        super().__init__()
        self.network = mlp(input_dim, 2 * target_dim, hidden_dim)

    def forward(self, normalized_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.network(normalized_input).chunk(2, dim=-1)
        # Bounded log-variance prevents numerical explosions in this tiny demo.
        return mean, torch.clamp(logvar, min=-10.0, max=2.0)


# 10. Dynamics negative log-likelihood loss.
def gaussian_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Diagonal Gaussian NLL, omitting only the irrelevant constant.

    NLL = 0.5 * exp(-logvar) * (target - mean)^2 + 0.5 * logvar
    """

    per_dimension = 0.5 * torch.exp(-logvar) * (target - mean).square()
    per_dimension = per_dimension + 0.5 * logvar
    return per_dimension.mean()


class DynamicsEnsemble(nn.Module):
    """Independent probabilistic models used for rollouts and MOBILE's U(s,a).

    MOBILE needs an ensemble because it compares Bellman/value estimates caused
    by different learned dynamics models—not merely their predicted states.
    """

    def __init__(
        self,
        ensemble_size: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        normalizer: DynamicsNormalizer,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.normalizer = normalizer
        self.members = nn.ModuleList(
            [
                ProbabilisticDynamicsModel(
                    obs_dim + action_dim, obs_dim + 1, hidden_dim
                )
                for _ in range(ensemble_size)
            ]
        )

    def predict_member(
        self,
        member_index: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_input = self.normalizer.normalize_input(observations, actions)
        mean, logvar = self.members[member_index](normalized_input)
        normalized_prediction = (
            mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        )
        prediction = self.normalizer.denormalize_target(normalized_prediction)
        delta = prediction[:, : self.obs_dim]
        reward = prediction[:, self.obs_dim :]
        return observations + delta, reward


def arrays_for_dynamics(
    dataset: OfflineDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(dataset.observations, device=device)
    actions = torch.as_tensor(dataset.actions, device=device)
    targets = torch.as_tensor(
        np.concatenate(
            [
                dataset.next_observations - dataset.observations,
                dataset.rewards,
            ],
            axis=1,
        ),
        device=device,
    )
    return observations, actions, targets


@torch.no_grad()
def dynamics_validation_nll(
    ensemble: DynamicsEnsemble,
    validation: OfflineDataset,
    device: torch.device,
) -> float:
    observations, actions, targets = arrays_for_dynamics(validation, device)
    normalized_input = ensemble.normalizer.normalize_input(observations, actions)
    normalized_target = ensemble.normalizer.normalize_target(targets)
    losses = []
    for member in ensemble.members:
        mean, logvar = member(normalized_input)
        losses.append(float(gaussian_nll(mean, logvar, normalized_target).item()))
    return float(np.mean(losses))


# 11. Dynamics ensemble training.
def train_dynamics_ensemble(
    ensemble: DynamicsEnsemble,
    train: OfflineDataset,
    validation: OfflineDataset,
    cfg: Config,
    device: torch.device,
) -> tuple[float, float]:
    """Maximum-likelihood fit with a separate bootstrap for each member.

    Dynamics are trained before actor/critic updates because synthetic rollouts
    need a usable simulator.  The MOBILE paper trains seven models and keeps the
    best five; this small example trains and uses every configured member.
    """

    observations, actions, targets = arrays_for_dynamics(train, device)
    normalized_input = ensemble.normalizer.normalize_input(observations, actions)
    normalized_target = ensemble.normalizer.normalize_target(targets)
    optimizers = [
        torch.optim.Adam(member.parameters(), lr=cfg.dynamics_lr)
        for member in ensemble.members
    ]
    rng = np.random.default_rng(cfg.seed + 4_000)
    last_train_nll = math.nan
    last_validation_nll = math.nan

    for epoch in range(1, cfg.dynamics_train_epochs + 1):
        member_losses = []
        for member, optimizer in zip(ensemble.members, optimizers):
            # Bootstrap rows induce epistemic diversity across ensemble members.
            bootstrap = rng.integers(0, len(train), size=len(train))
            for start in range(0, len(train), cfg.dynamics_batch_size):
                indices = torch.as_tensor(
                    bootstrap[start : start + cfg.dynamics_batch_size], device=device
                )
                mean, logvar = member(normalized_input[indices])
                loss = gaussian_nll(mean, logvar, normalized_target[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                member_losses.append(float(loss.item()))
        last_train_nll = float(np.mean(member_losses))
        last_validation_nll = dynamics_validation_nll(ensemble, validation, device)
        print(
            f"Dynamics epoch {epoch:>2}/{cfg.dynamics_train_epochs}: "
            f"train NLL={last_train_nll:.4f}, "
            f"validation NLL={last_validation_nll:.4f}"
        )
    return last_train_nll, last_validation_nll


# -----------------------------------------------------------------------------
# 12. SAC actor and twin critics
# -----------------------------------------------------------------------------


class SquashedGaussianActor(nn.Module):
    """Stochastic Gaussian policy with tanh squashing into environment bounds."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
    ):
        super().__init__()
        action_dim = len(action_low)
        self.trunk = mlp(obs_dim, 2 * action_dim, hidden_dim)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2.0),
        )

    def _distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self.trunk(observations).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        pre_tanh = distribution.rsample()  # reparameterization trains the actor
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        correction = torch.log(self.action_scale * (1.0 - squashed.square()) + 1e-6)
        log_prob = (distribution.log_prob(pre_tanh) - correction).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self._distribution(observations).mean
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.network = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1))


@dataclass
class SACNetworks:
    actor: SquashedGaussianActor
    critic1: Critic
    critic2: Critic
    target_critic1: Critic
    target_critic2: Critic


def build_sac_networks(
    obs_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    hidden_dim: int,
    device: torch.device,
) -> SACNetworks:
    actor = SquashedGaussianActor(obs_dim, action_low, action_high, hidden_dim).to(device)
    critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    target_critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    target_critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    return SACNetworks(actor, critic1, critic2, target_critic1, target_critic2)


# SAC contributes a stable off-policy actor/critic learner.  Two critics reduce
# optimistic value bias by taking their minimum.  Slowly moving target critics
# keep bootstrapped targets from chasing the critics being updated.


# -----------------------------------------------------------------------------
# 13. Synthetic model rollout generation
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate_model_rollouts(
    ensemble: DynamicsEnsemble,
    actor: SquashedGaussianActor,
    real_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    cfg: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """Generate imagined short-horizon transitions from real starting states.

    Imagined transitions expand the data available to SAC, but they are risky:
    an actor can exploit learned-model errors.  That is exactly why MOBILE adds
    a conservative penalty when these rows later form critic targets.

    Simplification: synthetic terminal prediction is ignored, so every model
    transition has done=0.  Production code should learn terminations or apply
    environment-specific termination logic.
    """

    observations = real_buffer.sample_observations(cfg.rollout_batch_size, device)
    for _ in range(cfg.rollout_horizon):
        actions, _ = actor.sample(observations)
        next_observations = torch.empty_like(observations)
        rewards = torch.empty((len(observations), 1), device=device)
        member_indices = rng.integers(0, len(ensemble.members), size=len(observations))
        for member_index in np.unique(member_indices):
            mask_np = member_indices == member_index
            mask = torch.as_tensor(mask_np, device=device)
            next_obs, reward = ensemble.predict_member(
                int(member_index), observations[mask], actions[mask], sample=True
            )
            next_observations[mask] = next_obs
            rewards[mask] = reward
        dones = torch.zeros((len(observations), 1), device=device)
        model_buffer.add_batch(
            observations.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            next_observations.cpu().numpy(),
            dones.cpu().numpy(),
        )
        observations = next_observations


# -----------------------------------------------------------------------------
# 14. Model-Bellman Inconsistency computation: MOBILE's key idea
# -----------------------------------------------------------------------------


@torch.no_grad()
def model_bellman_inconsistency(
    observations: torch.Tensor,
    actions: torch.Tensor,
    ensemble: DynamicsEnsemble,
    actor: SquashedGaussianActor,
    target_critic1: Critic,
    target_critic2: Critic,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Compute U(s,a) = Std_i(T_hat_i^pi Q(s,a)).

    For each ensemble member, ask: "If this model supplied the next state, what
    soft Bellman continuation value would the current actor and target critics
    assign?"  The standard deviation of those answers is the Model-Bellman
    Inconsistency.

    This differs fundamentally from plain next-state disagreement: two models
    can predict numerically different states with similar values, or nearby
    states separated by a sharp Q boundary.  Therefore U depends jointly on the
    dynamics ensemble, current policy, and learned value functions.

    Following the paper's Eq. (13), reward is kept outside U.  The paper's
    printed Eq. (13) shows min(Q) without the entropy term; this educational
    version uses the prompt's practical soft-SAC form, min(Q) - alpha*log(pi),
    consistently with the critic target.  We also use one mean next state and
    one sampled action per member instead of a larger inner Monte Carlo
    expectation—an intentional teaching shortcut.
    """

    bellman_estimates = []
    for member_index in range(len(ensemble.members)):
        next_observations, _ = ensemble.predict_member(
            member_index, observations, actions, sample=False
        )
        next_actions, next_log_prob = actor.sample(next_observations)
        next_q = torch.minimum(
            target_critic1(next_observations, next_actions),
            target_critic2(next_observations, next_actions),
        )
        bellman_estimates.append(gamma * (next_q - alpha * next_log_prob))
    # unbiased=False matches the paper's 1/N population-standard-deviation form.
    return torch.stack(bellman_estimates, dim=0).std(dim=0, unbiased=False)


# -----------------------------------------------------------------------------
# 15. Critic update with separate real and model targets
# -----------------------------------------------------------------------------


@torch.no_grad()
def sac_target(
    batch: dict[str, torch.Tensor],
    actor: SquashedGaussianActor,
    target_critic1: Critic,
    target_critic2: Critic,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    next_actions, next_log_prob = actor.sample(batch["next_observations"])
    next_q = torch.minimum(
        target_critic1(batch["next_observations"], next_actions),
        target_critic2(batch["next_observations"], next_actions),
    )
    return batch["rewards"] + gamma * (1.0 - batch["dones"]) * (
        next_q - alpha * next_log_prob
    )


def update_critics(
    real_batch: dict[str, torch.Tensor],
    model_batch: dict[str, torch.Tensor] | None,
    networks: SACNetworks,
    ensemble: DynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> dict[str, float]:
    # Real samples are genuine frozen-environment transitions, so their SAC
    # targets are not penalized.
    real_targets = sac_target(
        real_batch,
        networks.actor,
        networks.target_critic1,
        networks.target_critic2,
        cfg.gamma,
        cfg.alpha,
    )
    observations = [real_batch["observations"]]
    actions = [real_batch["actions"]]
    targets = [real_targets]
    uncertainty_mean = 0.0
    model_before_mean = math.nan
    model_after_mean = math.nan

    if model_batch is not None:
        model_targets_before_penalty = sac_target(
            model_batch,
            networks.actor,
            networks.target_critic1,
            networks.target_critic2,
            cfg.gamma,
            cfg.alpha,
        )
        uncertainty = model_bellman_inconsistency(
            model_batch["observations"],
            model_batch["actions"],
            ensemble,
            networks.actor,
            networks.target_critic1,
            networks.target_critic2,
            cfg.gamma,
            cfg.alpha,
        )
        # The MOBILE penalty is applied only to model-generated targets because
        # those transitions may encode model error.  It is not applied to D_real.
        model_targets_after_penalty = model_targets_before_penalty - cfg.beta * uncertainty
        observations.append(model_batch["observations"])
        actions.append(model_batch["actions"])
        targets.append(model_targets_after_penalty)
        uncertainty_mean = float(uncertainty.mean().item())
        model_before_mean = float(model_targets_before_penalty.mean().item())
        model_after_mean = float(model_targets_after_penalty.mean().item())

    all_observations = torch.cat(observations, dim=0)
    all_actions = torch.cat(actions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    q1 = networks.critic1(all_observations, all_actions)
    q2 = networks.critic2(all_observations, all_actions)
    critic_loss = F.mse_loss(q1, all_targets) + F.mse_loss(q2, all_targets)
    optimizer.zero_grad()
    critic_loss.backward()
    optimizer.step()

    return {
        "critic_loss": float(critic_loss.item()),
        "uncertainty": uncertainty_mean,
        "real_target": float(real_targets.mean().item()),
        "model_target_before": model_before_mean,
        "model_target_after": model_after_mean,
    }


# 16. Actor update.
def update_actor(
    observations: torch.Tensor,
    networks: SACNetworks,
    optimizer: torch.optim.Optimizer,
    alpha: float,
) -> float:
    # SAC maximizes min(Q1,Q2) - alpha*log(pi); minimizing its negative gives:
    actions, log_prob = networks.actor.sample(observations)
    q = torch.minimum(
        networks.critic1(observations, actions),
        networks.critic2(observations, actions),
    )
    actor_loss = (alpha * log_prob - q).mean()
    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()
    return float(actor_loss.item())


# 17. Soft target update.
@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)


# -----------------------------------------------------------------------------
# 18. Policy evaluation (never inserted into either replay buffer)
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_policy(
    env_id: str,
    episodes: int,
    seed: int,
    device: torch.device,
    actor: SquashedGaussianActor | None,
) -> tuple[float, float]:
    """Return undiscounted mean return and mean length in fresh episodes."""

    env = gym.make(env_id)
    _, _, action_low, action_high = check_continuous_env(env)
    rng = np.random.default_rng(seed + 123)
    returns = []
    lengths = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        while True:
            if actor is None:
                action = rng.uniform(action_low, action_high).astype(np.float32)
            else:
                observation_tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = actor.deterministic(observation_tensor).cpu().numpy()[0]
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        returns.append(episode_return)
        lengths.append(episode_length)
    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


# Dynamics NLL measures how well the learned simulator fits held-out offline
# transitions.  Real-environment return measures policy behavior.  Better model
# likelihood can help policy learning, but the metrics are related—not identical.


# -----------------------------------------------------------------------------
# 19. Main MOBILE-style training loop
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    # Phase A: real data collection happens once and then stops.
    dataset = create_frozen_offline_dataset(cfg)
    dynamics_train, dynamics_validation = split_dynamics_data(
        dataset, cfg.seed + 3_000
    )
    print_dataset_summary(dataset, dynamics_train, dynamics_validation)

    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]
    probe_env = gym.make(cfg.env_id)
    _, _, action_low, action_high = check_continuous_env(probe_env)
    probe_env.close()

    # D_real stays frozen.  D_model contains only imagined transitions.
    real_buffer = ReplayBuffer.from_offline_dataset(dataset, cfg.seed + 5_000)
    model_buffer = ReplayBuffer(
        cfg.model_buffer_capacity, obs_dim, action_dim, cfg.seed + 6_000
    )

    # Phase B: fit dynamics before any policy update.
    normalizer = DynamicsNormalizer(dynamics_train, device)
    ensemble = DynamicsEnsemble(
        cfg.dynamics_ensemble_size,
        obs_dim,
        action_dim,
        cfg.hidden_dim,
        normalizer,
    ).to(device)
    final_train_nll, final_validation_nll = train_dynamics_ensemble(
        ensemble, dynamics_train, dynamics_validation, cfg, device
    )

    # Phase C: SAC learns only from D_real and D_model, never fresh real data.
    networks = build_sac_networks(
        obs_dim, action_dim, action_low, action_high, cfg.hidden_dim, device
    )
    actor_optimizer = torch.optim.Adam(networks.actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(
        list(networks.critic1.parameters()) + list(networks.critic2.parameters()),
        lr=cfg.critic_lr,
    )
    rollout_rng = np.random.default_rng(cfg.seed + 7_000)

    for update in range(cfg.num_policy_updates):
        if update % cfg.rollout_interval == 0:
            generate_model_rollouts(
                ensemble,
                networks.actor,
                real_buffer,
                model_buffer,
                cfg,
                device,
                rollout_rng,
            )

        # Preserve real/model identity so only model targets receive beta * U.
        if len(model_buffer) == 0:
            real_batch_size = cfg.batch_size
            model_batch_size = 0
        else:
            real_batch_size = int(cfg.batch_size * cfg.real_ratio)
            real_batch_size = min(max(real_batch_size, 1), cfg.batch_size)
            model_batch_size = cfg.batch_size - real_batch_size
        real_batch = real_buffer.sample(real_batch_size, device)
        model_batch = (
            model_buffer.sample(model_batch_size, device)
            if model_batch_size > 0
            else None
        )

        metrics = update_critics(
            real_batch, model_batch, networks, ensemble, critic_optimizer, cfg
        )
        actor_observations = [real_batch["observations"]]
        if model_batch is not None:
            actor_observations.append(model_batch["observations"])
        actor_loss = update_actor(
            torch.cat(actor_observations, dim=0),
            networks,
            actor_optimizer,
            cfg.alpha,
        )
        soft_update(networks.critic1, networks.target_critic1, cfg.tau)
        soft_update(networks.critic2, networks.target_critic2, cfg.tau)

        if update == 0 or (update + 1) % cfg.log_interval == 0:
            print(
                f"Update {update + 1:>5}: model_buffer={len(model_buffer):>6}, "
                f"critic_loss={metrics['critic_loss']:.4f}, "
                f"actor_loss={actor_loss:.4f}, U={metrics['uncertainty']:.4f}, "
                f"real_target={metrics['real_target']:.3f}, "
                f"model_target={metrics['model_target_before']:.3f} -> "
                f"{metrics['model_target_after']:.3f}"
            )

        # Evaluation uses fresh real episodes but never stores or trains on them.
        if (update + 1) % cfg.eval_interval == 0 and update + 1 < cfg.num_policy_updates:
            evaluation_return, evaluation_length = evaluate_policy(
                cfg.env_id,
                cfg.eval_episodes,
                cfg.seed + 80_000 + update,
                device,
                networks.actor,
            )
            print(
                f"Evaluation at update {update + 1}: "
                f"return={evaluation_return:.2f}, length={evaluation_length:.1f}"
            )

    # Final paired evaluation: random baseline and trained MOBILE-style actor.
    evaluation_seed = cfg.seed + 90_000
    random_return, random_length = evaluate_policy(
        cfg.env_id, cfg.eval_episodes, evaluation_seed, device, actor=None
    )
    mobile_return, mobile_length = evaluate_policy(
        cfg.env_id, cfg.eval_episodes, evaluation_seed, device, networks.actor
    )
    print("\nFinal metrics")
    print(f"  dynamics train NLL: {final_train_nll:.4f}")
    print(f"  dynamics validation NLL: {final_validation_nll:.4f}")
    print(f"  model buffer size: {len(model_buffer)}")
    print(f"  random policy return: {random_return:.2f}")
    print(f"  MOBILE policy return: {mobile_return:.2f}")
    print(f"  random average episode length: {random_length:.1f}")
    print(f"  MOBILE average episode length: {mobile_length:.1f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# 20. Example commands
# ---------------------------------------------------------------------
# Requested smoke test:
# python learning_examples/minimal_mobile_from_paper.py \
#   --num_policy_updates 5 \
#   --dynamics_train_epochs 2 \
#   --num_offline_transitions 1000 \
#   --rollout_batch_size 128 \
#   --eval_episodes 1 \
#   --device cpu
#
# Slightly longer educational run:
# python learning_examples/minimal_mobile_from_paper.py \
#   --num_policy_updates 500 --dynamics_train_epochs 20 --device cpu
#
# Simplifications versus the MOBILE paper:
# - Pendulum data are created locally instead of loading D4RL/NeoRL.
# - Small networks/ensembles and tiny training budgets replace paper settings.
# - All ensemble members are used; the paper trains 7 and selects the best 5.
# - U(s,a) uses one mean next state and one sampled next action per model rather
#   than a larger inner Monte Carlo expectation.
# - Synthetic terminal prediction is omitted.
# - Entropy temperature alpha is fixed instead of automatically tuned.
# - This is a transparent single-run demo, not a benchmark reproduction.


# ---------------------------------------------------------------------
# 21. How this could later map into the real repository
# ---------------------------------------------------------------------
# Educational MOBILE concept | likely repo area to modify later | notes
# Frozen D_real              | src/tracks/general/real_ecology_benchmark/dataset.py
#                            | TrajectoryDataset is analogous; preserve its
#                            | public/private information boundary.
# Dynamics ensemble          | src/tracks/general/real_ecology_benchmark/dynamics.py
#                            | Existing ensemble is auditable NumPy ridge, not
#                            | this neural Gaussian model.
# Public/hidden dynamics     | src/tracks/general/real_ecology_benchmark/public_models.py
#                            | Any real integration must respect hidden inputs.
# D_model / mixed replay     | no direct equivalent
#                            | Would be new integration work, not an extension
#                            | of an existing replay-buffer abstraction.
# Actor and twin critics     | methods/ogsrl.py and
#                            | methods/ensemble_value_disagreement.py are only
#                            | loose analogues; neither is neural SAC.
# Method lifecycle           | src/tracks/general/real_ecology_benchmark/pipeline.py
#                            | build_method()/run_method() call fit once, then
#                            | evaluate the fitted policy.
# Config                     | src/tracks/general/real_ecology_benchmark/config.py
#                            | Uses nested dataclasses plus YAML/CLI/manifest
#                            | layers; a future MOBILE config should be explicit.
# Training metrics           | src/tracks/general/real_ecology_benchmark/training_monitor.py
#                            | Local CSV/JSON history is the closest logger.
# Fresh evaluation           | src/tracks/general/real_ecology_benchmark/evaluator.py
#                            | Evaluation is separate and must not update policy.
# Existing MOPO-named method | src/tracks/general/real_ecology_benchmark/methods/mopo.py
#                            | It uses pessimistic particle MPC, not SAC replay;
#                            | it should not be mistaken for this MOBILE flow.
#
# These are likely integration points only.  MOBILE does not currently exist in
# the repository, and this learning file intentionally modifies none of them.
