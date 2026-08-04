# Offline dataset pipeline

**Classification:** offline RL/planning over a simulator-generated fixed
dataset; online rollouts are evaluation only.

Unless a second track is named, code citations in this file are under
`src/tracks/general/real_ecology_benchmark/`.

| Item | Code-defined behavior |
|---|---|
| Generating policy | `collector.MixedDangerZonePolicy`, one episode-level mixture component |
| Public fields | Collector output always has observations, actions, rewards, next observations, done, episode/timestep (`PUBLIC_FIELDS`), plus costs, opaque pop IDs, terminated/truncated (`PUBLIC_SANITIZED_FIELDS`) and the action-cost vector. Full set-point cells additionally expose rho/kappa/K and next-control fields; hidden cells withhold exactly those `PUBLIC_CONTROL_FIELDS` (`dataset.py`; `collector.py:collect_dataset()`). |
| Private fields | true/next state, demographic parameters, regimes, collapse entry, true reward, initially unsafe; optional true effective rate and penalty flag |
| Size | accepted manifests request 4,000 transitions, 25-step complete episodes, normally 160 episodes |
| Observation | nonnegative scalar lognormal survey |
| Action | integer 0–10 plus public costs/channels |
| Reward | logged `env.step()` reward computed from true next state |
| Boundary | terminated means extinction; collector-forced or env-horizon end is truncated |
| Split | episode-disjoint, usually 80/20, RNG `cfg.seed + split_seed_offset` |
| Beliefs | transition-aligned public feature/mean cache |
| Seeds | collection RNG starts at `cfg.seed=116`; per-episode env seeds are draws from it |

## Behavior policy and calibration

At reset, `MixedDangerZonePolicy.reset()` selects random, harvest-probe,
rescue-dwell, or threshold-probe with profile probabilities. `act()` uses
`true_state` when privileged; collection passes true `env.state` at line 303.
This policy is not realizable from survey history alone. `DEFAULT_PROFILE`,
family/action-count `PROFILES`, and `REAL_DEFAULT_PROFILE` are explicit in
`collector.py`. For real 11-action cells the default is `(0.20,0.45,0.10,0.25)`
with harvest tilt.

`calibration_summary()` checks healthy-start incident collapse against
`[0.15,0.24]`. This tunes behavior mixture/start coverage and therefore the
offline distribution; `collector.py` explicitly notes robust populations may
remain outside rather than forcing the band.

## Public/private boundary and truth-derived reward

`dataset.py:PUBLIC_FIELDS`, `PUBLIC_SANITIZED_FIELDS`, and `PRIVATE_FIELDS` are
the authoritative schemas. Access to `states`, demographic values, regime,
`entry`, `reward_true`, `r_eff_true`, or `safety_penalty_applied` would leak
truth. `assert_public_schema()` rejects those fields and hidden control metadata.
However, reward itself is truth-derived. Collector metadata says
`truth_derived_public_signal: "logged reward only; policy/filter transition excludes reward"`.
Given known action cost, alpha, K_ref, P and penalty status, the smooth reward
term is invertible for s'; hidden methods do not receive K_ref/P/penalty status,
but can statistically infer abundance/safety information from repeated rewards.

## Caches, locking, and splitting

`pipeline.ensure_dataset()` loads both files or acquires
`<public>.npz.lock` using exclusive create, writes temp files, then atomically
replaces. A stale lock is not removed automatically; a waiter times out after
one hour. Hidden `_validate_dataset_cell()` checks only expose mode, action
count, observation sigma, reward mode, and horizon. It does **not** compare
population, family, process noise, collapse penalty, safety mode/threshold,
collection seed/profile, or target rows; cell-specific paths and dataset hashes
are therefore load-bearing. Full-mode validation covers more fields but not all
action-effect table bytes.

`training_monitor.split_train_holdout()` shuffles episode IDs with
`cfg.seed + training.split_seed_offset`, preserving transition/cache alignment.
Holdout metrics are appended by `record_final_fit_metrics()` and are not used
for selection by `pipeline.py`; faithful methods instead own an ordered episode
split inside `faithful_fit.py`.

Accepted block seeds `[7001,7051,7101,7151,7201]` are set in
`experiments/accepted_general/configs/general_phase2e_full_sigma01_02.yaml` and
re-stamped by the manifest column `evaluation_seeds`.
`evaluator.py:ContinuousEvaluator.run()` expands each block seed as
`block_seed+local_episode`, producing 7001–7004, 7051–7054, …, 7201–7204.
Collection env seeds are RNG draws from seed 116, so
disjointness is overwhelmingly likely but not arithmetically guaranteed by a
namespace check. The recorded accepted dataset would have to be inspected to
prove no collision; the public dataset does not store collection episode seeds.

The fitted features are not raw state: hidden
`beliefs.cache_public_beliefs()` stores `BeliefState.public_features()` for
current and next public-observation beliefs, while full
`beliefs.cache_dataset_beliefs()` stores `BeliefState.features(K_ref,s_safe)`.
The exact feature construction lives in
`src/tracks/general/real_ecology_benchmark/types.py:BeliefState`.

## Runnable dataset audit

```bash
PYTHONPATH=src/tracks/general python - <<'PY'
import numpy as np, sys
from real_ecology_benchmark.dataset import load_public, dataset_sha256
p=sys.argv[1] if len(sys.argv)>1 else "outputs/general_phase2e/datasets/public.npz"
d=load_public(p)
print("transitions/episodes",len(d),d.num_episodes)
print("obs",np.min(d.observations),np.mean(d.observations),np.max(d.observations))
print("actions",np.bincount(d.actions.astype(int),minlength=len(d.action_costs)))
print("reward",np.min(d.rewards),np.mean(d.rewards),np.std(d.rewards))
print("terminated/truncated",np.sum(d.terminated),np.sum(d.truncated))
print("hash",dataset_sha256(d),d.metadata.get("dataset_sha256"))
PY
```

Unclear: whether this command’s default file exists locally; accepted `.npz`
files are external scratch inputs. Verify using a path from
`provenance/dataset_hashes.csv` and `DEEPRL_GENERAL_DATA_ROOT`.
