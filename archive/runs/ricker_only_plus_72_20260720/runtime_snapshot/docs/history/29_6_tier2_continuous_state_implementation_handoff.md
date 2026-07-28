# 29_6 Tier-2 continuous-state noisy-observation implementation handoff

This is a next-chat-ready summary of the current Tier-2 implementation in
`discrete_action_cont_obser/`. It is written for follow-up work on improving the
current setting, methods, report, and code.

Status as of 2026-06-26:

- Standalone NumPy implementation exists under `discrete_action_cont_obser/`.
- It does not call into `claude_build/` or the old discrete implementation.
- The 384-row headline/ablation matrix has been run and aggregated.
- The report is `docs/22_6_Continuous_Observation_Experiment_Results.tex`.
- Local test suite passes: `PYTHONPATH=src python -m unittest discover -s tests -v`
  ran 32 tests successfully.

## 1. What was implemented

The implementation is the continuous-state, noisy-observation Tier-2 ecological
adaptive-management benchmark.

Core task:

- hidden latent abundance `s_t >= 0`;
- no bins, no finite `s_max`, no simulator-side clipping other than
  non-negativity;
- exact `s = 0` is terminal/absorbing;
- hidden episode parameters are drawn per episode and withheld from methods;
- true dynamics are non-Ricker stress families:
  - Allee-Ricker,
  - theta-logistic,
  - regime-switching;
- actions are the locked 5-action and 10-action management tables from the
  discrete benchmark;
- action authority is applied directly before growth:
  `s <- s * (1 - harvest_fraction) + stocking_delta`;
- observation is multiplicative log-normal:
  `o_t = s_t * eta_t`, with `eta_t ~ LogNormal(0, sigma_obs^2)`;
- observation-noise sweep:
  `sigma_obs in {0, 0.1, 0.2, 0.4}`;
- process noise is off in the headline setting:
  `sigma_p = 0`;
- public reward uses observed abundance:
  `alpha * o_t / (o_t + K_ref) - cost(a) - 20 * I[first entry into s <= 50]`;
- true-state reward is logged only for evaluation.

Main code map:

```text
discrete_action_cont_obser/
  configs/
    default.yaml             small smoke config
    full.yaml                final-budget config
  src/tier2_benchmark/
    actions.py               locked 5/10 action tables
    config.py                dataclass config and validation
    envs.py                  continuous Ricker/Allee/theta/regime simulator
    observation.py           exact log-normal emission kernel
    reward.py                operational and true reward
    collector.py             behavior policy and calibration profiles
    dataset.py               public dataset and private truth sidecar
    beliefs.py               raw/reference/learned/Ricker/true-family/oracle filters
    dynamics.py              bootstrap continuous dynamics ensemble
    planning.py              shared particle MPC
    gate.py                  three-controller decision gate
    evaluator.py             paired held-out evaluator
    manifest.py              384-row matrix generation and aggregation
    pipeline.py              end-to-end data/filter/method/eval orchestration
    cli.py                   command-line entry points
    methods/
      mopo.py
      refplan.py
      bamcts.py
      plus.py
      moor.py
      delphic.py
      ogsrl.py
      value.py
  scripts/
    run_gate_matrix.py
    run_manifest_row.py
    calibrate_profiles.py
    regenerate_failed_cells.py
    extract_report_tables.py
    capture_trajectories.py
    plot_trajectories.py
    slurm/
      run_gate_matrix.sh
      run_manifest_row.sh
      submit_full.sh
      capture_trajectories.sh
  tests/
    test_environment.py
    test_dataset_filter.py
    test_evaluator_gate.py
    test_methods.py
    test_calibration_profiles.py
  docs/
    architecture.md
    experiment_protocol.md
    methods.md
    reproducibility.md
    22_6_Continuous_Observation_New_Baselines.tex
    22_6_Continuous_Observation_Experiment_Results.tex
```

## 2. Information boundary and leakage discipline

The public method-training interface receives only:

- observation,
- action,
- public reward,
- next observation,
- done,
- episode ID,
- timestep.

The private truth sidecar contains latent state, hidden parameters, regime,
true reward, and collapse-entry information for evaluator/calibration only.
Training APIs do not accept the private file.

The only truth-derived public signal is the collapse-entry penalty bit, because
the public reward includes `-20` on first entry into `s <= safety_threshold`.
This is intentional: managers observe that a bad threshold event occurred through
the reward/incident signal, but they do not observe the hidden state itself.

Useful files:

- `src/tier2_benchmark/types.py`: public transition and policy/filter protocols.
- `src/tier2_benchmark/dataset.py`: public/private schema checks.
- `src/tier2_benchmark/evaluator.py`: policy receives sanitized public feedback.
- `tests/test_dataset_filter.py`: public/private round-trip and no-leak tests.
- `tests/test_evaluator_gate.py`: public-only `policy.observe()` test.

## 3. What the belief filter means

In Tier-2, the policy does not directly see true abundance `s_t`; it sees noisy
survey counts `o_t`. The belief filter is the state estimator:

```text
belief_t ~= P(s_t, latent episode context | o_0:t, a_0:t-1, r_0:t-1)
```

Implementation:

- represented as weighted particles;
- uses the exact known log-normal likelihood `p(o_t | s_t)`;
- predicts particles through a proposal model after each action;
- reweights particles by the new observation likelihood;
- resamples when effective sample size is low.

Important nuance: the learned filter has a generic latent episode context, not a
physically identified posterior over true `r_base`, `C`, `theta`, or `z`. State
RMSE is therefore meaningful; claims about correctly recovering physical hidden
parameters are not currently supported.

Filter modes:

- `learned`: primary shared known-emission particle filter, using a learned
  public-data log-space proposal;
- `raw`: ablation that treats observation as state;
- `ricker`: mechanistic Ricker proposal used for PLUS/MOOR baseline-fidelity
  rows;
- `true_family`: mechanistic true-family proposal for diagnostics/gate support;
- `oracle`: evaluator-only true-state belief ceiling, not allowed for training;
- `reference`: weak persistence proposal.

Role in the benchmark:

1. It makes the task a POMDP instead of a fully observed MDP.
2. It gives every method a comparable information front-end.
3. It prevents direct truth leakage.
4. It lets planners reason about posterior collapse risk, not just point
   abundance.
5. It is a major bottleneck at high observation noise.

Current learned-filter RMSE, averaged over methods:

```text
sigma_obs = 0.0  -> about   0
sigma_obs = 0.1  -> about  43
sigma_obs = 0.2  -> about  80
sigma_obs = 0.4  -> about 160
```

At `sigma_obs = 0.4`, the filter error is large enough to cap the learned
planners.

## 4. Implemented methods

General offline model-based / offline RL methods:

- `mopo`: bootstrap dynamics ensemble plus pessimistic particle MPC.
- `refplan`: posterior over ensemble members, mean-minus-uncertainty planning.
- `bamcts`: belief-rooted MCTS with continuous-state aggregation and pessimism.
- `delphic`: compatible-worlds CQL-style baseline; not a plain bootstrap
  ensemble.
- `ogsrl`: OOD guardian plus ecological safety constraint and deployment
  fallback.

Ecological baselines:

- `plus`: Ricker candidate-grid Bayesian controller, with per-candidate
  Rao-Blackwellized filter bank.
- `moor`: single fitted Ricker `(r, K)` model and fitted value iteration.

Important comparison semantics:

- PLUS and MOOR are intentionally Ricker-structured even when the true
  environment is Allee/theta/regime.
- This is the structural-misspecification stress test.
- In the headline matrix, PLUS/MOOR also receive the shared learned filter.
  Separate `ricker` filter rows quantify how much the shared front-end helps or
  hurts them.
- PLUS is almost filter-insensitive; MOOR is stronger with learned/raw filtering
  than with its own Ricker filter.

## 5. Experimental design: what each axis is testing

The matrix is:

```text
3 true dynamics families
  x {Allee, theta, regime}
2 action tables
  x {5-action, 10-action}
4 observation-noise levels
  x {0, 0.1, 0.2, 0.4}
7 methods
  x {MOPO, RefPlan, BA-MCTS, PLUS, MOOR, Delphic, OGSRL}
2 primary filter labels
  x {learned, raw}
+ Ricker-filter fidelity rows for PLUS/MOOR
= 384 rows
```

Meaning of the major axes:

### Structural misspecification stress

The true worlds are Allee/theta/regime, but PLUS and MOOR assume Ricker-family
dynamics. This asks:

> Can general learned dynamics beat strong ecological baselines when the
> baselines have the wrong structural model?

This stress test exists, but the current report mostly averages across the six
family/action cells. For a clearer paper, add family-stratified tables and a
dedicated section named something like “Ricker baselines on non-Ricker worlds.”

### Observation-noise stress

The four `sigma_obs` levels form a ladder:

- `0.0`: exact-observation sanity check;
- `0.1`: mild survey noise;
- `0.2`: moderate partial observability;
- `0.4`: heavy survey-noise stress test.

This asks:

> Does the learned-MBRL advantage survive when control depends on state
> estimation?

### Filter ablation

The learned-vs-raw-vs-Ricker filter rows ask:

> Are methods winning because they solve the noisy-observation problem, or are
> they just reacting to raw noisy observations?

### Safety evaluation

Collapse rate and unsafe occupancy ask:

> Does a policy preserve ecological safety, or does it obtain return by gambling
> near the threshold?

## 6. Data, calibration, gate, and evaluation

Full config:

- `configs/full.yaml`
- dataset seed: `116`;
- dataset transitions per cell: `75000`;
- dataset episode length: `25`;
- filter particles: `1024`;
- ensemble size: `15`;
- MPC horizon: `5`;
- MPC sequences: `64`;
- MPC particles: `16`;
- evaluation seeds: `[7001, 7051, 7101, 7151, 7201]`;
- episodes per seed: `50`;
- evaluation horizon: `50`;
- discount: `0.95`.

Calibration:

- target healthy-start incident-collapse band: `[0.15, 0.24]`;
- final rates:

```text
allee_10a   0.1835
allee_5a    0.2116
regime_10a  0.1960
regime_5a   0.2013
theta_10a   0.2094
theta_5a    0.2089
```

Collector profiles:

- `default`: preserved for already-passing `allee_10a` and `regime_10a`;
- `rescue_tilt`: used for `allee_5a` and `regime_5a`;
- `regulate`: used for `theta_5a` and `theta_10a`.

Decision gate:

- hard gate: Allee/regime at `sigma_obs <= 0.2`;
- `sigma_obs = 0.4` is diagnostic stress;
- theta is diagnostic/graded-control in the current report;
- all final cells pass their gate checks.

Gate-gap ranges:

```text
hard Allee/regime:
  reward gap   [2.88, 7.31]
  collapse gap [0.25, 0.80]

diagnostic theta:
  reward gap   [1.46, 5.03]
  collapse gap [0.15, 0.40]
```

## 7. Current empirical result

Use learned-filter rows for method ranking. Do not rank using aggregate
`model_return_mean`, because it mixes filter labels unevenly.

Learned-filter operational return:

```text
method    sigma0  sigma0.1  sigma0.2  sigma0.4  all
MOOR       6.97     6.94      7.22      7.80    7.23
MOPO       7.20     7.21      6.94      5.77    6.78
RefPlan    7.33     7.23      7.14      5.14    6.71
OGSRL      4.37     5.95      7.39      7.48    6.30
PLUS       5.64     5.74      5.93      6.56    5.97
BA-MCTS    6.01     5.87      5.66      4.18    5.43
Delphic    5.58     3.74      5.98      5.77    5.27
```

Learned-filter collapse rate:

```text
method    sigma0  sigma0.1  sigma0.2  sigma0.4  all
MOOR      0.053    0.055     0.043     0.023   0.043
OGSRL     0.103    0.057     0.038     0.053   0.063
MOPO      0.051    0.051     0.066     0.123   0.073
RefPlan   0.044    0.054     0.061     0.161   0.080
PLUS      0.116    0.119     0.111     0.087   0.108
BA-MCTS   0.114    0.122     0.131     0.185   0.138
Delphic   0.125    0.251     0.105     0.125   0.151
```

Learned-filter beats-both-baselines rate, where the comparator is
`max(PLUS, MOOR)` per cell:

```text
method    sigma0  sigma0.1  sigma0.2  sigma0.4
BA-MCTS    0.67     0.83      0.50      0.00
RefPlan    0.67     0.50      0.33      0.00
MOPO       0.50     0.33      0.33      0.00
OGSRL      0.00     0.17      0.50      0.17
Delphic    0.00     0.17      0.17      0.00
```

Filter ablation, learned minus raw, mean operational return over all cells:

```text
Delphic  +1.45
OGSRL    +1.40
BA-MCTS  +1.30
RefPlan  +0.56
MOPO     +0.28
PLUS     +0.09
MOOR     -0.05
```

PLUS/MOOR baseline-fidelity check:

```text
PLUS:
  learned = 5.97
  raw     = 5.88
  ricker  = 5.99

MOOR:
  learned = 7.23
  raw     = 7.28
  ricker  = 6.83
```

Interpretation:

- General learned methods do not universally beat the ecological baselines.
- At low-to-moderate noise, learned planners can beat the stronger baseline in
  a meaningful subset of cells.
- At high noise, the learned-planner advantage collapses.
- MOOR is the overall winner in the learned-filter matrix.
- OGSRL is the most robust safe learner at high noise, likely because the hard
  guardian/fallback becomes increasingly active.
- Delphic does not show the hoped-for monotone advantage with observation noise.
- BA-MCTS wins many low/mid-noise cells but has relatively high collapse risk.

## 8. Why the current learned methods lose at high noise

Likely causes, in priority order:

1. The learned filter is too weak at high observation noise.
   The proposal is an auditable linear log-space model, not a strong recurrent
   state-space model. At `sigma_obs = 0.4`, RMSE is about 160.

2. The learned dynamics ensemble is compact and intentionally simple.
   `dynamics.py` uses bootstrap ridge members over log-abundance features with
   action-state interactions. It does not preserve a physically meaningful
   per-episode latent parameter during rollout.

3. Rollout uncertainty is partly treated like per-step residual noise.
   The true headline process noise is zero, but learned ensemble rollouts sample
   residuals from fitted members. This can blur deterministic hidden-parameter
   uncertainty into repeated transition noise.

4. Learned planners have short effective horizons.
   MOPO/RefPlan/BA-MCTS use short particle planning/tree depths. MOOR/PLUS use
   fitted value-iteration-style longer-horizon structure, which can be an
   advantage near collapse thresholds.

5. MOOR's misspecified Ricker model is conservative in a useful way.
   At high noise, a simple robust mid-band controller can outperform a more
   flexible learner that compounds filter/model error.

6. Delphic's current compatible worlds are not yet a strong confounding model.
   The implemented uncertainty is posterior-ambiguity-conditioned random-feature
   world variation, not a fully paper-faithful latent-confounding identification
   pipeline.

## 9. Recommended next improvements

Highest priority:

1. Add a stronger belief/filter backend.
   Candidate: recurrent state-space model or neural proposal that still uses the
   exact log-normal emission likelihood. Keep the current particle-filter
   interface.

2. Add oracle/diagnostic ablations before changing too much.
   Useful rows:
   - learned model + oracle-state evaluation filter;
   - learned filter + true-family mechanistic proposal;
   - residual sampling on/off;
   - fixed ensemble member per rollout vs resampled member per step;
   - horizon 5 vs longer horizon or terminal value.

3. Improve the dynamics model.
   Candidate: small MLP ensemble or latent-context dynamics model that keeps an
   episode-level context fixed during rollout.

4. Improve planning horizon fairness.
   Add terminal value estimates to MOPO/RefPlan/BA-MCTS or use CEM/beam search
   with longer horizons.

5. Separate theta in the report.
   Theta is calibrated and decision-relevant, but it behaves differently and may
   deserve a graded-control/negative-control framing rather than being silently
   pooled into every headline.

6. Make the structural-misspecification story explicit.
   Add family-stratified tables:
   - Allee only,
   - theta only,
   - regime only,
   - 5-action vs 10-action splits.

7. Rework Delphic if it is part of the scientific claim.
   Validate whether its `u_delta` tracks confounding strength as `sigma_obs`
   increases. If not, either improve the compatible-world fitting or describe it
   as an exploratory baseline.

8. Calibrate OGSRL per noise level.
   Current guardian/fallback helps high-noise safety but is too conservative at
   exact observation.

9. Improve BA-MCTS leaves.
   Replace zero-value leaves with fitted belief-value estimates and add a
   threshold-aware terminal penalty.

10. Add independent dataset seeds.
    Current evaluation has paired held-out seeds, but only one dataset seed
    (`116`). A robustness study needs multiple offline datasets.

## 10. Report/doc improvements needed

The current result doc is numerically useful but conceptually hard to read.
Recommended edits:

- Add an “experimental logic” section explaining:
  - structural misspecification stress;
  - observation-noise stress;
  - belief-filter ablation;
  - safety evaluation.
- Add family-stratified results instead of relying only on six-cell averages.
- Be explicit that `sigma_obs = 0.4` is a diagnostic stress setting, not a
  gate-enforced primary condition.
- Keep warning that `model_return_mean` in `aggregate.json` is filter-mixed and
  should not be used for ranking.
- Say “generic episode context” for the learned filter, not posterior over true
  physical parameters.
- Mention that trajectory plots are illustrative only: five captured paired
  episodes, not the statistical result.

## 11. Useful commands

From `discrete_action_cont_obser/`:

```bash
# Tests
PYTHONPATH=src python -m unittest discover -s tests -v

# Small smoke
PYTHONPATH=src python -m tier2_benchmark.cli smoke --config configs/default.yaml

# Generate one dataset / run one method
PYTHONPATH=src python -m tier2_benchmark.cli generate --config configs/full.yaml --transitions 75000
PYTHONPATH=src python -m tier2_benchmark.cli run --config configs/full.yaml --method mopo --filter learned

# Gate matrix
PYTHONPATH=src python scripts/run_gate_matrix.py --config configs/full.yaml

# Full manifest support
PYTHONPATH=src python - <<'PY'
from tier2_benchmark.manifest import make_manifest
print(make_manifest("outputs/manifest.csv"))
PY

# Extract report tables from completed artifacts
python scripts/extract_report_tables.py

# Calibration profile checks
PYTHONPATH=src python scripts/calibrate_profiles.py --check-committed
PYTHONPATH=src python scripts/calibrate_profiles.py --verify
```

Slurm:

```bash
sbatch --parsable scripts/slurm/run_gate_matrix.sh
sbatch --parsable scripts/slurm/submit_full.sh
sbatch --parsable scripts/slurm/capture_trajectories.sh
```

The implementation is CPU/NumPy. GPU jobs are not required unless future neural
filter/dynamics backends are added.

## 12. Best next-chat prompt

Paste something like this into the next chat:

```text
We are improving the Tier-2 continuous-state noisy-observation ecological offline
MBRL benchmark in discrete_action_cont_obser/. First read:

- docs/29_6_tier2_continuous_state_implementation_handoff.md
- docs/architecture.md
- docs/experiment_protocol.md
- docs/methods.md
- docs/22_6_Continuous_Observation_Experiment_Results.tex

Do not rewrite the whole benchmark. The current implementation passes tests and
has a completed 384-row result matrix. Focus on improving the current setting and
code.

Main scientific issue: general learned MBRL does not beat PLUS/MOOR overall;
MOOR wins at high observation noise. I want you to propose and/or implement the
highest-value improvement. Prioritize:

1. stronger belief filter / recurrent proposal;
2. latent-context dynamics model that keeps episode-level uncertainty fixed;
3. oracle/residual-off/fixed-member/long-horizon ablations;
4. planner terminal value or longer-horizon planning;
5. family-stratified reporting and clearer structural-misspecification story.

Before coding, inspect the relevant current files and tell me exactly which
change you recommend first, why, expected compute, and what result would validate
it.
```

