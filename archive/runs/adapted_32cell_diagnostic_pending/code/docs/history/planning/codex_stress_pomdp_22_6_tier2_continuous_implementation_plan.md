# Tier-2 continuous-state, noisy-observation implementation plan

Status: planning only. No implementation is authorized by this document.

## 1. Scope and design stance

This plan adds a Tier-2 benchmark beside the existing `stress_pomdp_116` implementation. It does not convert Tier-1 in place. The existing discrete environment, datasets, adapters, manifests, tests, and result artifacts remain reproducible.

The Tier-2 public process is:

- latent state `s_t >= 0`, continuous and unbounded;
- hidden episode context `m` (including `r_base` and the applicable `C`, `theta`, or switching regime state);
- discrete management action `a_t` from the existing 5- or 10-action table;
- direct authority `s_managed = s_t * (1 - h(a_t)) + delta(a_t)` before growth;
- observation `o_t = s_t * eta_t`, `eta_t ~ LogNormal(0, sigma_obs^2)`;
- operational reward from the current observation and action, plus a latent transition-entry penalty;
- public trajectories only for learning; latent truth exists only in an evaluator/calibration channel.

The architectural rule is to share contracts and infrastructure, not discrete representations. Tier-2 should reuse action records, Hydra/registry conventions, seed pairing, artifact conventions, and the high-level method identities. It should not reuse integer state interfaces, categorical next-bin networks, dense transition tensors, bin midpoint reconstruction, or finite-state value iteration.

The Tier-2 method set is deliberately limited to MOPO, RefPlan, BA-MCTS, PLUS, MOOR, Delphic, and OGSRL. The registered Tier-1 methods `combo`, standalone `cql`, `iql`, `romi`, and `hmmdp_baseline` are out of scope unless a later ablation explicitly adds them; their omission is not an implementation oversight.

The collector's information set is foundational: the primary dataset keeps the Tier-1 behavior-policy convention in which collection decisions use privileged true abundance `s_t`, while the learner receives only `o_t`. At `sigma_obs=0`, this source of action-outcome confounding disappears because `o_t=s_t`; as observation noise rises, residual confounding from the hidden true abundance is expected to strengthen. This is a falsifiable trend hypothesis for Delphic, not a guarantee that its return advantage must be monotone or that it must equal MOPO at zero noise.

## 2. What is reused, extended, or replaced

| Existing component | Tier-2 treatment | Reason |
|---|---|---|
| `src/environments/reward.py::ActionSpec` and the calibrated 5/10-action YAML records | Reuse unchanged values; preferably expose the parser from a neutral action module later | The action IDs and `(Delta r, Delta K, h, delta, cost)` contract are locked. |
| `src/registry.py` and Hydra composition | Reuse | Registry construction and action-count validation are representation-independent. |
| Hidden-parameter priors and equations in `src/environments/stress_pomdp_envs.py` | Re-express in new continuous classes | The biology is reusable; inheritance from `RickerEnv` is not, because it brings bins and `s_max`. |
| `BaseEnvironment`, `Dataset` in `src/interfaces/base_env.py` | Keep for Tier-1; add parallel Tier-2 protocols/types | Their types and documentation require integer observations, `num_states`, and flat transition tuples. |
| `BaseModel` in `src/interfaces/base_model.py` | Keep for Tier-1; add a belief-policy interface | `predict()` is defined as an `S`-way categorical distribution and every model exposes `num_states`. |
| `run_pipeline.py` orchestration | Reuse the phase structure, not its loader/validator | Its reward validation, metadata, model construction, and trainer branches assume discrete arrays. |
| `mixed_danger_zone_116` behavior-mixture idea | Reuse policy components after defining their information set | The current collector reads private continuous abundance; Tier-2 must explicitly decide whether that privileged access is intentional confounding. |
| MOPO bootstrap ensemble and batched rollout pattern | Reuse concept and acceleration structure | Embeddings, logits, cross-entropy, expected-bin uncertainty, clipping, and categorical sampling must be replaced. |
| RefPlan reflection/model-posterior and sequence scoring | Reuse concept | The current implementation is a tabular repo adaptation based on exact transition likelihoods and finite-state terminal values. |
| BA-MCTS posterior-state tree search | Reuse concept | Dense tabular members, integer tree keys, categorical observations, and VI leaves must be replaced. |
| PLUS candidate Ricker family and Bayesian model weighting | Reuse the intended structural bias | Current `[M,S,A,S]` tables and precomputed tabular `Q_k` are not usable. |
| MOOR single fitted Ricker model | Reuse the intended structural bias | Bin midpoint fitting, sub-bin Monte Carlo, and finite VI must be replaced. |
| `UnifiedEvaluator` seed pairing/artifact pattern | Reuse concept | Its history, abundance, coverage, entropy, and danger metrics are bin-valued. |
| `stress116_control_gap.py` oracle-versus-Ricker idea | Reuse concept | Bin midpoint reconstruction and a clairvoyant/noisy information mismatch must be removed. |
| Existing Slurm manifest/worker/report pattern | Extend with Tier-2-specific scripts | Tier-1 shell logic infers `num_states` from suffixes and is unsafe to reuse directly. |

## 3. Proposed repository layout

Names are provisional but deliberately parallel to Tier-1.

### 3.1 Interfaces and data

- `claude_build/src/interfaces/continuous_env.py`
  - Tier-2 environment protocol and reset/step result types.
- `claude_build/src/interfaces/belief_policy.py`
  - policy lifecycle: fit, reset episode, act from a `BeliefState`, observe transition, diagnostics, save/load.
- `claude_build/src/data/trajectory_dataset.py`
  - flat, vectorized public arrays plus `episode_id`, `timestep`, and episode offsets.
- `claude_build/src/data/trajectory_io.py`
  - schema/version validation, NPZ or Zarr serialization, metadata hashing, and public/private artifact separation.
- `claude_build/src/data/continuous_collector.py`
  - whole-trajectory collection and behavior-mixture policies.

The public training schema should contain only:

`observations`, `actions`, `rewards`, `next_observations`, `dones`, `episode_id`, `timestep`.

It should also contain immutable metadata such as schema version, environment key, action-table hash, `sigma_obs`, reward settings, seed, and trajectory offsets. No `s_t`, hidden parameter, regime, process-noise draw, or observation-noise draw belongs in the public artifact. A separately permissioned evaluator sidecar may contain truth for calibration and filter metrics; model-loading code must have no path to it. The public reward deliberately reveals one latent event bit when the `-20` entry penalty fires; document this as the only intended truth-derived signal available to the learner, rather than treating it as accidental leakage.

### 3.2 Environment

- `claude_build/src/environments/continuous/actions.py` (optional neutral home for the existing action parser; no value changes).
- `claude_build/src/environments/continuous/observation.py`
  - log-normal sample/log-likelihood, including exact branches for `sigma_obs=0` and the point mass at `s=0`.
- `claude_build/src/environments/continuous/reward.py`
  - operational and true-state reward channels, entry-event bookkeeping, and expected rollout reward helpers.
- `claude_build/src/environments/continuous/base.py`
  - RNG ownership, initial-state mixture, hidden episode lifecycle, action authority, horizon, truth diagnostics, and numerical checks.
- `claude_build/src/environments/continuous/ricker.py`
- `claude_build/src/environments/continuous/allee_ricker.py`
- `claude_build/src/environments/continuous/theta_logistic.py`
- `claude_build/src/environments/continuous/regime_switch.py`
- `claude_build/config/env/tier2/`
  - composable dynamics, action-table, observation-noise, initialization, reward, collection, and evaluation settings. Prefer composition/overrides over 24 copied YAML files.

### 3.3 Shared belief system

- `claude_build/src/beliefs/types.py`
  - particle tensors, normalized log weights, summaries, ESS, resampling diagnostics, and stable feature vectors.
- `claude_build/src/beliefs/filter.py`
  - common filter protocol and episode lifecycle.
- `claude_build/src/beliefs/particle_filter.py`
  - the first required backend over `(log s_t, episode latent m, z_t where applicable)`.
- `claude_build/src/beliefs/reference_proposal.py`
  - non-learned log-space persistence/random-walk proposal used to validate the known emission and particle machinery before training a latent proposal.
- `claude_build/src/beliefs/raw_observation.py`
  - ablation backend with the identical interface.
- `claude_build/src/beliefs/latent_transition.py`
  - learned, trajectory-level latent state-space proposal; it must not call the simulator's true equations.
- `claude_build/src/beliefs/training.py`
  - train the primary shared filter once per public dataset, checkpoint it, freeze it for all methods, and cache public offline-dataset belief trajectories.
- `claude_build/src/beliefs/method_proposals.py`
  - optional baseline-faithfulness ablations using method-appropriate transition proposals; these do not replace the primary shared-filter comparison.
- `claude_build/src/beliefs/recurrent_encoder.py`
  - optional second backend only after the particle filter is validated.

The particle filter should use a fixed known emission kernel and a learned transition proposal. Static episode latents need rejuvenation or ancestor-preserving updates; ordinary resampling with `sigma_p=0` will otherwise collapse the parameter particles early. The regime model additionally needs a discrete Markov component.

### 3.4 Continuous learned dynamics and planning

- `claude_build/src/models/continuous/dynamics_ensemble.py`
  - probabilistic ensemble over latent next state, conditioned on belief features/action; observation samples are produced by the fixed emission model.
- `claude_build/src/models/continuous/value_models.py`
  - belief-conditioned terminal/reward/cost critics shared where appropriate.
- `claude_build/src/planners/continuous_mpc.py`
  - batched action-sequence/particle MPC with common random numbers, expected observation reward, entry risk, and model disagreement.
- `claude_build/src/models/continuous/mopo_adapter.py`
- `claude_build/src/models/continuous/refplan_adapter.py`
- `claude_build/src/models/continuous/bamcts_adapter.py`
- `claude_build/src/models/continuous/plus_adapter.py`
- `claude_build/src/models/continuous/moor_adapter.py`
- `claude_build/src/models/delphic/`
  - compatible-world latent models, counterfactual value estimator, uncertainty cache, and Delphic-CQL adapter.
- `claude_build/src/models/ogsrl/`
  - guardian, safety/OOD cost models, guarded learned simulator, categorical constrained actor-critic, and adapter.

New adapters should not subclass the Tier-1 adapters. Sharing their class names would hide materially different algorithms and invite accidental use of tabular utilities.

### 3.5 Pipeline, analysis, and tests

- `claude_build/scripts/core/generate_tier2_dataset.py`
- `claude_build/scripts/core/train_tier2_filter.py`
- `claude_build/scripts/core/run_tier2_pipeline.py`
- `claude_build/scripts/analysis/tier2_calibrate.py`
- `claude_build/scripts/analysis/tier2_decision_gate.py`
- `claude_build/scripts/analysis/aggregate_tier2.py`
- `claude_build/scripts/experiments/make_tier2_manifest.py`
- Tier-2-specific Slurm workers and report scripts.
- Tests grouped as `test_tier2_envs.py`, `test_tier2_observation.py`, `test_tier2_dataset.py`, `test_tier2_filter.py`, `test_tier2_methods.py`, `test_tier2_evaluator.py`, and `test_tier2_gate.py`.

## 4. Environment interface and transition contract

The proposed public interface is conceptually:

```text
reset(seed, mode) -> ResetResult(observation: float, done: bool, public_info: dict)
step(action: int) -> StepResult(
    observation: float,
    reward: float,
    done: bool,
    truncated: bool,
    public_info: dict,
    evaluator_info: private/guarded diagnostics,
)
num_actions -> int
action_specs -> tuple[ActionSpec, ...]
observation_model -> LogNormalObservationModel
reward_model -> ContinuousRewardContract
```

No `num_states`, bin index, `s_max`, or public true abundance appears in this interface.

One step is ordered as follows:

1. At reset, sample hidden episode parameters, regime state, and `s_0` from the locked mixture; emit `o_0`.
2. The policy receives only public history/belief and selects `a_t`.
3. Compute the current observed abundance utility from the already emitted `o_t`.
4. If `s_t=0`, apply the selected extinction convention; otherwise apply direct authority before growth.
5. Compute the applicable continuous growth law in `float64`, add process noise only when its stress switch is enabled, and apply only the non-negativity floor.
6. Determine the latent safety-entry event from `(s_t, s_{t+1})` and the episode latch convention.
7. Compute `R_t` from `o_t`, action cost, and the entry event. Compute `R_t_true` in the private evaluator channel.
8. Advance the switching regime after the transition has used `z_t`.
9. Emit `o_{t+1}` from `s_{t+1}` and return it with the public reward.

The environment owns separate RNG streams for episode parameters, regime transitions, process noise, observation noise, and policy/collector randomness. This permits common-random-number pairing without making one component's sampling order alter another.

Numerical stability must not become a hidden `s_max`. Use log-domain computations where possible, explicit finite checks, and a run-invalid diagnostic on overflow. Do not silently clamp state or exponent to a scientifically meaningful ceiling. Calibration should make non-finite trajectories absent under the declared priors/actions.

## 5. Dataset, calibration, and decision gate

### 5.1 Collector

Port the four `mixed_danger_zone_116` components, but express thresholds in physical abundance units. For the primary dataset, lock decisions to privileged `s_t` as in Tier-1; the learner still receives only `o_t/history`. This creates the hidden action-outcome confounding needed by Delphic. An observation-only collector is a named causal-ablation dataset, not an interchangeable default. Each episode is appended intact; train/validation splits are by episode, never by transition. Frame/history builders use episode offsets rather than relying only on `done`.

For each `{Allee, theta, regime} x {5a,10a} x sigma_obs`, target about 75,000 transitions in one public dataset. Use the same physics, prior, action authority, initial mixture, and behavior-mixture parameters across noise levels; do not tune a different ecological system for every `sigma_obs`.

Calibration outputs should report:

- episode and transition counts;
- start distribution, including initially unsafe starts;
- incident entry/collapse rate among episodes starting above the safety floor;
- occupancy below the safety floor;
- action frequencies overall and near the floor;
- private behavior-propensity/action-frequency diagnostics by true-abundance band and hidden-parameter/regime bucket;
- observation quantiles and non-finite counts;
- effective support by action and belief region;
- regime occupancy/switch counts in private diagnostics.

The target collapse band should be applied to an unambiguous incident-collapse denominator. Cells outside the band are recalibrated using initialization/behavior-mixture settings, not per-noise changes to the locked action table or biological priors unless explicitly approved.

### 5.2 Gate

Use three controllers, not two:

1. **Belief oracle (hard-gate comparator):** sees the same noisy observations as the Ricker controller and knows the true dynamics family/prior, but not the realized state/episode parameters.
2. **Belief Ricker MPC:** sees the same observations and uses the misspecified Ricker family.
3. **Clairvoyant oracle (diagnostic upper bound):** knows true state and realized parameters; never used as the sole hard-gate comparator.

The hard gate should compare (1) and (2), with paired episode parameters, starts, observation-noise streams, and planner random numbers. This isolates structural misspecification under observation error. Comparing a true-state oracle directly with a noisy-observation Ricker controller would mix information advantage with model advantage and could pass even when the hidden structural model is not decision-relevant. Implement both belief controllers by plugging their respective transition models into the same particle-MPC engine used by the methods; do not build a second gate-only belief planner.

MPC should optimize expected operational return by integrating future observation noise, while entry risk is computed on latent rollout particles. Preserve the existing reward-gap and collapse-gap outputs and add action disagreement, controllability spread in physical units, and uncertainty intervals. Run each `sigma_obs` separately. A cell/noise combination that fails remains a reported failed gate; do not retune physics independently by noise after seeing method results.

## 6. Shared belief filter and raw-observation ablation

### 6.1 Required filter

The locked primary comparison still uses one shared state-belief filter checkpoint per public dataset, frozen and fed to every method. This prevents filter quality from becoming an unobserved method-specific advantage. Implement it as a staged ladder so the hardest unsupervised component is not the first blocker:

1. **Reference PF (engineering/MVP):** known log-normal emission, locked initial prior, and a weak log-space persistence/random-walk proposal. This validates weighting, resampling, edge cases, and `sigma_obs=0` without learning a latent SSM. It is a reference and smoke-test filter, not the final headline filter.
2. **Shared learned-proposal PF (primary):** replace the weak proposal with one trajectory-level latent transition proposal trained only from public trajectories, then freeze the resulting filter for all primary method comparisons.
3. **Method-appropriate proposal ablations:** Ricker proposals for MOOR/PLUS and learned-ensemble proposals for general learners quantify baseline faithfulness. PLUS additionally receives a Rao-Blackwellized/per-candidate filter-bank ablation. These are reported sensitivity analyses, because making them primary would confound algorithm differences with state-estimator differences and weaken the locked “shared filter feeding all methods” comparison.

The particle state and common machinery use:

- state represented internally as `log(s + epsilon)` plus an exact extinction atom;
- the locked initial mixture as the state prior;
- an episode-level latent context in the learned-proposal version and a discrete Markov component for regime switching;
- the exact log-normal observation likelihood;
- log weights, ESS-triggered systematic resampling, and static-latent rejuvenation;
- fixed-dimensional features such as posterior mean/std and quantiles of `log1p(s/K_ref)`, extinction/unsafe probability, latent-context moments, regime probability, ESS, and observation noise level;
- full particles for planners that can use them and stable summary features for neural policies.

Special likelihood branches are mandatory:

- `sigma_obs=0`: deterministic `o=s`, not a zero-variance Gaussian calculation;
- `s=0, o=0`: point mass with probability one;
- `s=0, o>0` and `s>0, o=0`: impossible under the declared observation model, handled without `log(0)` crashes.

The primary shared filter is trained once per dataset and frozen. Every method receives the same checkpoint and initial prior. Method-specific structural beliefs (for example PLUS's candidate-Ricker posterior or Delphic's compatible worlds) sit on top of, and are not silently folded into, the common state estimate. Cache the filter trajectory over each fixed offline dataset once and reuse it during method training. Do not claim cross-method caching at evaluation: policies choose different actions, so their observation/action histories and beliefs diverge immediately.

The learned episode latent may be identifiable only up to a transformation because true parameters are never labeled and simulator equations are withheld. State RMSE is therefore the shared headline filter metric. Method-specific physical diagnostics remain valid where the method explicitly estimates them: PLUS candidate-`r` posterior concentration, MOOR fitted-parameter recovery on the Ricker control, and regime-state posterior calibration may be reported separately.

Filter quality is a measured covariate, not background infrastructure. Report filter RMSE/log-RMSE beside every return delta and stratify or plot method advantage against filter error. Add a one-cell **oracle-state ablation** in which methods receive private true `s_t`; it is evaluator-only and estimates the performance ceiling removed by state-estimation error. It is never included in headline beats-baseline rates.

### 6.2 Raw ablation

The raw backend provides the same policy-facing type but uses normalized `o_t`, recent actions/rewards, and `sigma_obs`, with no filtering. Models must be retrained for this input; merely swapping the feature vector at evaluation would be an invalid ablation. Report belief-versus-raw for every method if compute permits, and at minimum for MOPO, PLUS, MOOR, Delphic, and OGSRL.

## 7. Method adaptations

### 7.1 MOPO

- Replace state/action embeddings and categorical next-bin logits with a continuous probabilistic latent transition ensemble.
- Predict latent next-state distributions and compose them with the fixed observation kernel; do not train a single Gaussian directly on `o_{t+1}` and call observation noise model uncertainty.
- Train members on episode-preserving bootstraps and trajectory windows.
- Use batched particle MPC. Score current observed reward, expected future observation reward, expected entry penalty, and ensemble disagreement in normalized physical units.
- Separate aleatoric emission variance from epistemic member disagreement so the pessimism coefficient does not automatically punish high `sigma_obs` twice.

### 7.2 RefPlan

- Replace `TabularDynamicsEnsemble` with continuous latent members and history likelihoods marginalized through the shared filter/emission model.
- Maintain a posterior over members from trajectory evidence.
- Sample belief-conditioned action sequences and latent/member rollouts; score mean return minus member/return uncertainty.
- Replace tabular terminal VI with a fitted belief-value model or a zero-terminal-value sensitivity check.

### 7.3 BA-MCTS

- Nodes contain a compact particle-belief summary plus depth, not an integer state.
- Each simulation samples a model member, latent state/parameter particle, next latent state, and next observation, then performs a Bayes update.
- Progressive widening or observation clustering is required because continuous observations create essentially one branch per sample.
- Replace posterior-averaged VI leaves with the same fitted belief-value model used by RefPlan or a short MPC rollout.
- Cache only after quantifying approximation error from belief compression; rounded member weights alone are not sufficient.

### 7.4 MOOR

- Fit one misspecified Ricker model to shared filtered state distributions. The preferred fit is a state-space likelihood/EM objective that integrates state uncertainty; least squares on posterior means is a declared cheaper ablation.
- Preserve action authority in the assumed transition.
- Plan in continuous belief space with sampled MPC or fitted Q iteration; do not resurrect a fixed abundance grid and dense VI as the primary implementation.
- Propagate observation noise through planning and expose fit uncertainty from restarts/bootstrap only as a diagnostic unless the baseline definition explicitly uses it.

### 7.5 PLUS

- Keep the finite grid of candidate Ricker `r_base` models and its uniform/configured prior.
- For each candidate, compute predictive observation likelihood by propagating candidate-conditioned state particles through the candidate Ricker model and the fixed emission kernel.
- Implement a Rao-Blackwellized/per-candidate state-filter bank for the baseline-faithfulness ablation. The primary shared-filter result may initialize/regularize those candidate beliefs from the common state posterior, but it must be disclosed as an approximation rather than called an exact PLUS port.
- Update the candidate posterior in log space. The shared state posterior is the common state estimate; candidate weights are PLUS-specific model belief.
- Replace tabular `Q_k(x,a)` with continuous candidate-wise fitted Q functions or cached candidate MPC value approximators, then act by posterior-weighted candidate value.
- Preserve Ricker structural misspecification: do not add candidate Allee, theta, or regime models.

### 7.6 Delphic

The paper's primary algorithm is not merely a transition ensemble. It trains multiple compatible latent-confounder worlds, each containing a trajectory posterior over `z`, a behavior-policy model `pi_b(a|state,z)`, and a behavior-policy value model `Q^{pi_b}(state,a,z)`. Worlds vary latent dimensionality, priors, and architectures; data bootstraps represent epistemic uncertainty. Counterfactual `Q^pi_w` variation across worlds is Delphic uncertainty, which is applied as a Bellman penalty to CQL in the paper.

The primary Tier-2 baseline should therefore be named **Delphic-CQL**:

- use frozen shared belief/history features as the observed state;
- fit the world models from whole trajectories;
- vary world assumptions deliberately, rather than treating ordinary random initializations as compatible worlds;
- estimate `u_Delta(b,a) = Var_w Q^pi_w(b,a)` and keep it distinct from within-world bootstrap disagreement;
- train a discrete-action CQL head with the Delphic Bellman penalty;
- update/cache uncertainty as the target policy changes, following the paper's delayed update idea.

A model-based Delphic-MOPO penalty is a useful secondary ablation, not a paper-faithful replacement for Delphic-CQL. The benchmark must also verify that the collection policy actually depends on information hidden from the learner; otherwise partial observability exists but Delphic's action-outcome confounding target may be weak or absent.

Use the noise sweep as a built-in confounding diagnostic under the locked privileged-`s` collector. At `sigma_obs=0`, true abundance is observed and the Delphic-specific penalty should be small or provide little incremental benefit; with noisier proxy observations, residual action-outcome confounding should generally grow. Predeclare tests for this trend in `u_Delta` and policy behavior. Do **not** require monotonically increasing return advantage or equality with MOPO: Delphic-CQL and MOPO have different backbones and may differ even when delphic uncertainty vanishes.

### 7.7 OGSRL

The paper's full instantiation is a guarded model-based constrained optimizer (GMB-CPO), not just `mean - lambda * OOD_score`. Tier-2 should implement:

- a guardian over low-dimensional belief features plus discrete action, using KDE/k-NN support scores and a held-out quantile threshold;
- a learned latent simulator shared with the OGSRL policy learner;
- an OOD visitation cost and a separate ecological safety cost;
- a categorical actor with reward, safety-cost, and OOD-cost critics and dual updates (a discrete-action GMB-CPO adaptation);
- safety cost based on posterior probability of entering/occupying the unsafe region, not an indicator of posterior mean;
- a hard fallback action only for numerical/infeasible-policy failures, with fallback frequency reported.

Add guarded CQL as a diagnostic because the paper evaluates it, but label the main A* baseline `OGSRL-GMB-CPO-discrete`. Do not claim the paper's containment theorem transfers unchanged: the benchmark uses learned beliefs, a learned low-dimensional support representation, discrete actions, and an approximate guardian.

For engineering sequencing, a Delphic-MOPO penalty and a guarded Lagrangian planner may be built first as explicitly named MVP adaptations to exercise the end-to-end interfaces. They are not eligible substitutes for the two A* headline baselines. Delphic-CQL and `OGSRL-GMB-CPO-discrete` remain required before final comparative claims, regardless of whether the MVP versions show favorable signal.

## 8. Unified evaluation and reporting

### 8.1 Protocol

- One frozen public training dataset per environment/action/noise cell by default.
- Five disjoint held-out seed blocks, 50 episodes each, maximum horizon 50, `gamma=0.95`.
- Identical episode parameter, start, regime, process-noise, and observation-noise streams across methods.
- Filter and policy reset at every episode; no posterior leakage across episodes.
- Policy receives only public observation/reward history and its belief.
- Predeclare per-method wall-clock and memory budgets plus CPU/GPU placement. Keep the common 250-episode headline protocol; if BA-MCTS needs a reduced scalability subset, label it separately rather than mixing unequal episode counts into the main table.

### 8.2 Per-episode outputs

- discounted operational return from `o_t`;
- discounted `R_true` in the private evaluator channel;
- first safety-entry indicator/time and number of entries under the chosen latch convention;
- fraction of latent timesteps below the safety threshold;
- observed abundance summaries, explicitly labeled as observations;
- true abundance summaries, private and explicitly labeled;
- filter RMSE plus log-RMSE/normalized RMSE, posterior interval coverage, ESS, and extinction/unsafe probability calibration;
- paired return deltas joined to the corresponding filter-error summary, so filter quality can be analyzed as a covariate;
- action frequencies near the safety boundary;
- method-specific uncertainty diagnostics without pretending their scales are comparable;
- wall time, peak CPU/GPU memory, and fallback/infeasibility counts.

### 8.3 Aggregation and claims

For each method, cell, and `sigma_obs`, compute paired seed-level differences against `max(PLUS, MOOR)` and the fraction of cells with positive paired mean difference. Report the beats-both rate with a bootstrap confidence interval, not only a count. Also report the two baseline returns separately so a weak-baseline artifact is visible.

Primary tables:

1. belief-filter methods across the four noise levels;
2. raw-observation ablation with paired deltas;
3. operational return, `R_true`, incident collapse, unsafe occupancy, and filter error;
4. beats-PLUS, beats-MOOR, and beats-both rates;
5. gate outcomes and failed/diagnostic cells;
6. one-cell oracle-state ceiling ablation;
7. runtime/memory, decision-time/filter-time/planner-time breakdown, and training-model counts.

Do not use state-space coverage as `visited/num_states`; use support/guardian diagnostics or observation/belief quantile coverage instead. Rename the old `policy_entropy`, which is predictive transition entropy, or replace it with actual action entropy.

## 9. Build order and stop criteria

### Phase 0 — freeze contracts and experiment matrix

- Settle Decision 6 (privileged behavior) and Decision 7 (meaning of shared filter) first, then the remaining numbered decisions in Section 12.
- Write a one-page public/private information contract and schema version.
- State explicitly that the entry penalty is the one intended truth-derived bit in public history.
- Define exact cell count, training seeds, gate policy, per-method wall budgets, and CPU/GPU placement.

Exit: no ambiguity remains about safety threshold, collapse/extinction semantics, collector information, filter fairness, or baseline variants.

### Phase 1 — continuous environment, observation, reward, and configs

- Implement parallel Tier-2 interfaces and all four dynamics.
- Reuse exact action values and verify authority-before-growth.
- Add deterministic tests for each equation, reset priors, regime timing, observation moments, reward timing, `sigma_obs=0`, `s=0`, no hidden-field leakage, and absence of clipping.

Exit: long random rollouts across declared priors have no non-finite values; empirical observation moments match theory; Tier-1 tests remain unchanged.

### Phase 2 — trajectory collector, calibration, and gate

- Add public trajectory artifacts and private evaluator sidecars.
- Port and lock the behavior mixture to privileged true abundance for primary datasets; add observation-only collection only as a causal ablation.
- Calibrate with small deterministic sweeps, then 75k-transition candidates.
- Run the cheap clairvoyant-versus-Ricker diagnostic and controllability checks now. Defer the final information-matched belief gate until the common PF/MPC machinery exists in Phase 3.

Exit: data schema/leakage tests pass, collapse/support targets are documented, behavior confounding diagnostics are present, and the provisional controllability/clairvoyant diagnostics are acceptable. No final gate claim is made yet.

### Phase 3 — shared particle filter and raw ablation

- Implement the known-emission reference PF first, including special cases, rejuvenation, diagnostics, and deterministic tests.
- Build the common particle-MPC engine with pluggable transition models; use it for the belief oracle and belief Ricker gate.
- Complete the information-matched hard gate and freeze admitted/diagnostic cells.
- Train the common latent proposal from public trajectories, freeze the primary shared-filter checkpoint, and cache offline-dataset belief trajectories once.
- Add method-appropriate proposal and PLUS filter-bank ablation hooks without making them primary.
- Implement the raw backend and evaluator-only state-error metrics.
- Add the one-cell oracle-state ceiling ablation contract.

Exit: `sigma_obs=0` state reconstruction is effectively exact; synthetic likelihood tests pass; controlled filter error responds sensibly to noise; repeated seeds are deterministic; no truth enters training; Allee/regime final gates pass at the approved primary noise levels.

### Phase 4 — MOPO, RefPlan, and BA-MCTS

- Plug the continuous learned dynamics ensemble into the already-tested particle-MPC spine first.
- Add RefPlan member posterior and fitted terminal value.
- Add BA-MCTS with progressive widening and belief compression.

Exit: each method overfits a tiny public dataset, beats a random policy in a smoke cell, honors episode boundaries, and produces finite batched planning results.

### Phase 5 — MOOR and PLUS

- Implement uncertainty-aware continuous MOOR fitting/planning.
- Implement candidate predictive likelihoods and continuous `Q_k` approximation for PLUS.
- Verify direct action physics and observation likelihood against hand calculations.

Exit: on a well-specified Ricker control, MOOR recovers plausible parameters and PLUS posterior concentrates; on stress cells they remain intentionally Ricker-misspecified.

### Phase 6 — Delphic and OGSRL

- **6A, interface MVP:** implement an explicitly labeled Delphic-MOPO penalty and guarded Lagrangian planner to validate data flow, uncertainty/cost logging, and evaluator integration.
- **6B, required headline baselines:** implement Delphic-CQL compatible worlds/uncertainty decomposition and the discrete GMB-CPO dual-constrained optimizer.
- Implement guardian calibration and guarded-CQL diagnostic.
- Use the privileged-collector noise sweep as the primary confounding diagnostic, with an optional injected-confounding control only if the trend is ambiguous.

Exit: Delphic uncertainty responds to residual hidden-state confounding rather than merely data bootstrap size; its zero-noise penalty is a negative-control diagnostic; guardian false-rejection/support metrics are calibrated; OGSRL satisfies modeled budgets before full evaluation. MVP adaptations cannot satisfy this exit criterion by themselves.

### Phase 7 — unified evaluator, manifests, and ablation tables

- Freeze tuning on calibration seeds before held-out runs.
- Generate manifests with shared dataset/filter checkpoint paths.
- Run staged smoke, one-cell pilot, all-cell belief matrix, then raw ablation.
- Split CPU-bound methods from GPU-bound learners using the existing worker pattern.
- Enforce predeclared wall budgets and record filter time separately from planner/model time.
- Cache only offline-dataset beliefs across methods; never reuse evaluation beliefs across policies.
- Aggregate paired results, confidence intervals, runtime, and failed runs.

Exit: every reported result is reproducible from a manifest row and artifact hashes; no held-out seed influenced calibration or hyperparameter selection.

## 10. Self-critique: highest-risk assumptions and recommended resolutions

| Risk or ambiguity | Why it can break the study | Recommended resolution | Alternative |
|---|---|---|---|
| **Shared filter circularity** | A particle filter over `(s,m)` needs transition dynamics, but dynamics are what the learners are meant to learn. Giving the simulator equations to the filter leaks the answer; giving each method its own primary filter destroys the locked shared comparison. | Validate a known-emission reference PF first, then train one public-data latent proposal and freeze it for the primary matrix. Treat its episode context as latent, not identified physical `C/theta/r`. | Add method-appropriate proposal results as baseline-faithfulness ablations, not replacements for the common primary filter. |
| **PLUS and MOOR inherit a learned front end** | A shared learned filter gives the mechanistic baselines learned-dynamics assistance, changing their scientific identity. | Declare the common filter a measurement-error correction layer, keep their planning dynamics strictly Ricker, and report raw plus method-appropriate Ricker-filter ablations. | Make method-specific filters primary; more internally faithful, but algorithm and filter effects can no longer be separated. |
| **Gate information mismatch** | True-state oracle versus noisy Ricker MPC can pass due only to clairvoyance. | Hard-gate with observation-matched belief controllers; retain clairvoyant oracle as an upper-bound diagnostic. | Keep the old comparison but add a second information-matched gap and require both. |
| **Safety threshold naming/value** | `C_safe` is overloaded with the regime model's safe threshold `C=90`, while Tier-1 collapse was about physical abundance 50. Theta has no biological `C`. | Rename the benchmark floor `s_safe_threshold` and set one physical value across models after approval. | Environment-specific safety floors; easier calibration but weakens cross-cell comparability. |
| **Floor-only dynamics versus old absorbing collapse** | Snapping `s<=threshold` to zero reintroduces a non-biological transition and violates “non-negativity floor only”; not snapping means “collapse” is an event and recovery may occur. | Do not snap to zero; latch the first healthy-to-unsafe entry for the one-time penalty, continue true dynamics, and reserve absorbing behavior for exact `s=0`. | Preserve Tier-1 snap-to-zero as a separately named catastrophe mode, not the default Tier-2 physics. |
| **Extinction plus stocking/revenue** | `s=0` cannot be absorbing if stocking is applied, and a negative harvest cost can generate revenue forever at zero. | Treat exact extinction as an absorbing terminal state: end the episode immediately, retain the reward formula on the transition that entered zero, and treat horizon 50 as a maximum. Padding for batched storage is masked and contributes no reward. | Continue at zero under the literal reward formula; formally simple but permits harvest revenue from an extinct population. |
| **Reward timing and hidden leakage** | The abundance term uses current `o_t`, while the entry penalty needs `s_{t+1}`. The penalty reveals a hidden event through reward. | Specify the step order in Section 4 and accept the penalty as an intentional public signal; do not also store a public collapse flag. | Compute penalty from observations, but that changes the locked reward and makes safety noise-dependent. |
| **Log-normal edge cases** | Standard log-density code is undefined at `sigma=0` and at zero state/observation. | Explicit deterministic and point-mass branches with unit tests. | Add arbitrary epsilons; simpler but changes the model near extinction. |
| **Deterministic dynamics and particle impoverishment** | With `sigma_p=0` and static episode parameters, resampling can irreversibly delete plausible parameter modes. | Use log-space particles, ESS resampling, Liu-West/regularized rejuvenation for static latents, and regime-specific discrete updates. | SMC2/particle MCMC; more principled and much more expensive. |
| **Unbounded-state numerics** | Ricker exponentials can overflow; theta-logistic can overshoot negative. A silent clip recreates `s_max`. | `float64`, log-domain evaluation, non-negativity floor, finite assertions, and calibration that excludes overflow. Mark overflow as an invalid cell/run. | A declared numerical emergency bound far above observed support, reported on every hit; practical but technically a maximum. |
| **Predicting observations directly** | A Gaussian model on `o_{t+1}` conflates measurement noise with process/model uncertainty and corrupts MOPO/Delphic penalties. | Predict latent next state and compose with the known emission kernel. | Direct heteroscedastic observation model as a raw-ablation-only learner. |
| **Behavior-policy confounding is underspecified** | Delphic requires hidden information to affect both logged actions and outcomes. If behavior acts only on public history, hidden episode parameters are latent dynamics but may not be action confounders in the paper's sense. | Define a privileged collector that uses true abundance (as Tier-1 effectively does), quantify confounding strength/action propensity variation, and disclose it. | Use observation-only collection and present Delphic as robustness to latent context rather than a matched hidden-confounding baseline. |
| **Overclaiming the noise-confounding trend** | Privileged true-`s` behavior makes residual confounding vanish at exact observation and generally strengthen as its proxy becomes noisier, but algorithm return differences also include backbone, support, and estimation effects. | Treat shrinking zero-noise `u_Delta` and a noise response as predeclared diagnostics; do not require monotone Delphic return advantage or equality with MOPO. | Add a controlled propensity/confounding-strength dataset if the natural noise sweep is inconclusive. |
| **Theta interpretation** | Tier-1 theta-5a was weak, but theta-10a passed the original numerical gate; declaring the whole family a negative control before the Tier-2 gate would prejudge the result. | Keep theta diagnostic/non-blocking and classify each theta cell after the predeclared gate. A failed-gate theta cell becomes a negative control; a passed cell remains decision-relevant. | Declare all theta cells negative controls, sacrificing potentially valid 10-action evidence. |
| **Delphic shortcut risk** | Calling ordinary dynamics-ensemble disagreement “delphic” would duplicate epistemic uncertainty and not implement the paper. | Implement compatible latent worlds plus behavior/value models and a Delphic-CQL penalty. | Implement Delphic-MOPO but label it an adaptation and retain a small Delphic-CQL validation. |
| **OGSRL shortcut risk** | An OOD penalty alone omits the paper's separate safety constraint and constrained optimizer. | Implement guardian cost plus ecological safety cost with dual-constrained model-based policy optimization. | Guarded CQL as a cheaper diagnostic, not the headline OGSRL result. |
| **Guardian on beliefs** | KDE on a high-dimensional particle vector is meaningless; theorem assumptions do not cover a learned belief representation. | Use a fixed low-dimensional belief feature map, held-out threshold calibration, and empirical containment metrics; make no transferred theorem claim. | k-NN on recurrent embeddings; potentially stronger but less interpretable. |
| **Safety on posterior mean** | A mean above the floor can hide substantial posterior collapse probability. | Define modeled safety cost using posterior probability of entry/occupancy below the floor. | CVaR of minimum latent abundance, more conservative and harder to tune. |
| **Continuous observation branching in BA-MCTS** | Naive trees create one child per sampled observation and never revisit nodes. | Progressive widening/observation clustering plus belief-summary hashing and fitted-value leaves. | POMCP-style root sampling without explicit observation child reuse. |
| **Internal planning discretization** | Reusing fixed grids/VI would make “no bins” mostly cosmetic and introduces an undeclared upper truncation. | Continuous sampled MPC or fitted Q/value approximation for MOOR/PLUS and learned methods. | Adaptive quadrature/grid only as a documented numerical approximation with convergence ablation. |
| **Initial low-start mixture and collapse denominator** | Episodes can start below the floor, so “collapse rate” can mean initial unsafe status or incident entry. | Report both; use healthy-start incident entry as the calibration/gate collapse rate and all-step occupancy as the safety burden. | Count any unsafe start as collapse; simpler but dominated by the fixed 20% mixture. |
| **Observation noise changes reward distribution** | `LogNormal(0,sigma^2)` is median-one but mean-greater-than-one; operational returns are not directly the same objective across noise levels. | Keep the locked observation model, compare methods within each noise level, and use `R_true` for cross-noise interpretation. | Mean-correct the noise to `LogNormal(-sigma^2/2,sigma^2)`, which would change the locked design. |
| **Offline hyperparameter selection** | Delphic, CQL, CPO duals, guardian thresholds, and planners have no reliable online validation objective. | Predeclare compact grids and select using calibration seeds plus model-fit/safety diagnostics; never held-out environment return. | Fitted Q evaluation, with sensitivity tables because it can be biased under confounding. |
| **Compute explosion and eval-time multiplier** | `3 x 2 x 4 x 7 x 2` gives 336 fits before tuning, while every evaluation step adds PF updates and particle planning. BA-MCTS is likely dominated by decision-time simulation, not training. | One 75k public dataset/cell, one primary shared checkpoint, cache only fixed offline-data beliefs, vectorize rollouts, split CPU/GPU workers, predeclare wall budgets, and record filter/planner time separately. Eval beliefs cannot be shared across policies. | Five training datasets or reduced BA-MCTS cells as separately declared robustness/scalability studies, not mixed into the headline protocol. |

## 11. Where the discrete port is specifically non-trivial

1. `BaseEnvironment` and `BaseModel` encode finite state in their abstract APIs, not merely in implementations. A compatibility shim would still expose fake `num_states` and invite clipping.
2. Dataset reward validation currently recomputes rewards from stored integer `(state,next_state)` tuples. Tier-2 entry penalties cannot be reconstructed from public observations alone without accepting the reward's hidden event signal.
3. MOPO's network, loss, sampling, uncertainty, and planner all use categorical bins. This is a full model/planner replacement, not an output-head change.
4. RefPlan and BA-MCTS allocate dense `[M,S,A,S]` dynamics and update member beliefs with exact observed transition probabilities. Continuous noisy observations require marginal likelihoods through a state belief.
5. BA-MCTS's tree identity and finite-VI leaves rely on recurrent integer states. Continuous observations require approximate belief identity and observation branching control.
6. PLUS materializes candidate transition and Q tables and treats exact bin observation as exact state knowledge. Tier-2 requires joint predictive observation likelihoods and continuous value approximation.
7. MOOR reconstructs bin midpoints, normalizes by `s_max`, fits least squares to proxies, and builds one finite MDP. All four steps change under latent continuous state.
8. Shared `value_iteration` is central to RefPlan leaves, BA-MCTS leaves, PLUS, and MOOR. Increasing its grid resolution would reintroduce Tier-1 and still require an arbitrary upper bound.
9. The evaluator's abundance, coverage, danger, catastrophic threshold, and entropy fields are bin-specific or mislabeled. They cannot be mechanically rescaled.
10. The control gate reconstructs abundance from a bin midpoint and compares controllers with different knowledge. Both transition and information contracts change.
11. Shell and manifest code derives state count from `_bw2/_bw5` suffixes and inserts `model.num_states`; Tier-2 needs a separate manifest schema.
12. Exact observations currently let PLUS/RefPlan/BA-MCTS update only model belief. Tier-2 introduces a state belief whose uncertainty must be propagated through every decision-time rollout.

## 12. Open decisions requiring approval before coding

1. **What numerical safety threshold should the benchmark use?** Recommended default: `s_safe_threshold=50` for all dynamics, preserving the Tier-1 physical threshold; reserve `C` for biological Allee/regime parameters.

2. **What exactly happens on first entry below the safety threshold?** Recommended default: apply the penalty once per episode on the first above-to-at/below transition, do not force state to zero, continue physical dynamics, and define collapse as that incident event.

3. **What happens at exact `s=0`?** Recommended default: exact extinction is an absorbing terminal state. End the episode on entry to zero, keep the locked reward formula for that final transition, and treat horizon 50 as a maximum. Any storage padding is masked and contributes no reward. This avoids both stocking revival and harvest revenue from an extinct population.

4. **Are initially unsafe episodes counted as collapsed?** Recommended default: no for incident-collapse rate; report `initially_unsafe` separately and include them in unsafe occupancy.

5. **Is the benchmark floor strict or inclusive?** Recommended default: use `s <= s_safe_threshold` consistently for reward, collapse, and occupancy; avoid mixing `<` and `<=`.

6. **What information does the offline behavior policy receive?** Foundational recommended default: lock primary collection to privileged true-abundance access to create genuine hidden action-outcome confounding, but not direct access to episode parameter labels; log/plot behavior propensities by latent diagnostic bucket privately. An observation-only collector is a named causal ablation. Settle this first because it determines whether Delphic is scientifically matched.

7. **What does “shared filter” mean for ecology baselines?** Recommended default: use a known-emission reference PF to validate the machinery, then one learned public-data filter checkpoint frozen for the primary all-method matrix. PLUS/MOOR retain Ricker planning/model beliefs. Report raw and method-appropriate Ricker-filter/per-candidate-PLUS ablations to quantify how much the common learned front end changes baseline fidelity.

8. **May the shared filter use simulator equations/true priors?** Recommended default: it may use the declared initial prior and exact observation kernel, but not true transition equations or realized hidden parameters. The reference PF may use a weak declared persistence/random-walk proposal; the primary proposal is learned from public trajectories.

9. **Should the filter's `m` be interpreted as the true named physical parameters?** Recommended default: no for the common learned latent. Use an episode-level latent context and make state filtering the shared headline. Still evaluate physical quantities where a method explicitly estimates them: PLUS `r`-candidate posterior, MOOR parameters on the Ricker control, and regime-state posterior calibration.

10. **May PLUS/MOOR use an internal finite abundance grid?** Recommended default: no for primary results. Use continuous fitted value functions/MPC; if a grid is retained as an alternative, require adaptive-range/convergence ablations and label it a planning approximation.

11. **Which Delphic variant is the headline baseline?** Recommended default: paper-faithful Delphic-CQL on shared belief features. Build Delphic-MOPO first only as an explicitly labeled engineering scaffold/secondary adaptation; do not decide whether to implement Delphic-CQL based on scaffold performance.

12. **Which OGSRL variant is the headline baseline?** Recommended default: `OGSRL-GMB-CPO-discrete` with separate OOD and ecological safety constraints. A guarded Lagrangian planner may be the engineering scaffold and guarded CQL a diagnostic, but neither substitutes for the headline baseline.

13. **How is OGSRL's safety budget defined?** Recommended default: discounted expected posterior probability of first unsafe entry, with a predeclared budget selected on calibration data; report true entry rate separately. A chance/CVaR constraint is an alternative if a stricter safety claim is desired.

14. **Which oracle controls the decision-relevance hard gate?** Recommended default: observation-matched belief oracle versus observation-matched Ricker MPC; report the clairvoyant true-state/parameter oracle only as an upper bound.

15. **Does theta remain diagnostic/non-blocking?** Recommended default: yes, matching the validated Tier-1 interpretation. Classify each theta cell by its predeclared Tier-2 gate: failed-gate cells are explicit negative controls where learned methods should not claim a structural-misspecification win; passed cells remain decision-relevant diagnostics. Do not label the whole theta family negative-control in advance.

16. **Must every noise level pass the hard gate?** Recommended default: Allee/regime must pass at `sigma_obs in {0,0.1,0.2}`; `0.4` may be a declared stress/failure regime rather than trigger physics retuning. If the claim requires all four, approve that stronger criterion before calibration.

17. **Is base Ricker a headline evaluation cell?** Recommended default: implement it as a control/smoke and baseline-recovery test, but keep the headline matrix at the six stress cells specified in the design.

18. **How many independent offline training datasets are used per cell?** Recommended default: one 75k-transition dataset per cell plus five disjoint held-out evaluation seed blocks. This matches “75k transitions/cell” and keeps the A* matrix feasible. Five independently trained datasets per cell should be a later robustness study.

19. **How are private true states retained for `R_true` and filter RMSE?** Recommended default: a separate evaluator-only sidecar with a distinct loader and schema; public training artifacts contain no truth. Deterministic regeneration from seeds is the stricter but slower alternative.

20. **How broad is the raw-observation ablation?** Recommended default: retrain all seven methods under raw input after the belief-filter matrix passes. If compute is constrained, predeclare the reduced set `{MOPO, PLUS, MOOR, Delphic, OGSRL}` rather than selecting after results.

21. **What constitutes “beats PLUS/MOOR”?** Recommended default: positive paired mean return difference against the stronger baseline within each cell/noise, with a paired bootstrap confidence interval reported; use the simple positive-rate as headline and confidence-aware wins as a sensitivity analysis.

22. **Can calibration vary by observation noise?** Recommended default: behavior policy naturally changes through noisy inputs, but biological priors, actions, threshold, initialization mixture, and behavior-mixture probabilities remain common across noise. Do not tune separate physics per noise level.

23. **What is the compute/failure policy for expensive evaluation?** Recommended default: predeclare wall-clock and memory budgets per method, preserve the common 250-episode headline protocol, split CPU/GPU workers, and report filter/planner timing. If BA-MCTS needs fewer cells or episodes, make that a separate scalability subset rather than an unequal headline comparison.

No coding should begin until Decisions 1–16 and 18–19 are fixed. Resolve them in priority order `6 -> 7 -> 1/3 -> remaining`; Decision 23 must be frozen before full-matrix manifests, and the others before their corresponding method/evaluation phase.
