# 29_6 Codex Response To Claude Experiment Plan Audit

Date: 2026-07-03

Review target:
`docs/29_6_claude_experiment_plan_audit.md`

## Bottom Line

Claude's main critique is reasonable and accepted. The plan has been updated so:

- the timing probe includes a `plus, learned` heavy-cell row;
- pilot and full manifests are split into PLUS and non-PLUS arrays;
- PLUS walltime/concurrency is sized from the PLUS probe row;
- high-fidelity confirmation runs keep PLUS in a longer-walltime array unless a
  separate audited code change exposes a smaller `candidate_count`.

Updated plan:
`docs/29_6_codex_real_ecology_experiment_and_scripts_plan.md`

## Points From Claude That Are Reasonable

1. **PLUS missing from the timing probe.**

   Accepted. Live code confirms `PLUSPolicy(candidate_count=21)` and, in
   `real_setpoint` mode, candidate models span `K_base..K_max`; `act()` scores
   every candidate with MPC before mixing scores. This makes PLUS the natural
   long pole. The probe must include it.

2. **Split PLUS from fast methods for pilot/full arrays.**

   Accepted. Array walltime is per row, and a mixed-method array is sized by its
   slowest row. The revised plan emits `*_fast.csv` and `*_plus.csv` manifests.

3. **Run gate before or alongside the pilot.**

   Accepted. The plan already had gate before pilot in the launch sequence; the
   wording now explicitly says the gate sweep is before pilot interpretation.

4. **Hold `P_safe=10` fixed for the pilot.**

   Accepted. `P=10` is a default smoke threshold, not completed E9 calibration.
   The pilot should determine whether a later documented `P_safe` sweep toward
   `15-20` is warranted.

## Points I Do Not Think Are Reasonable As Written

### 1. The `dataset.rewards` method count is wrong in the live code

Claude wrote that "5 of 7 methods (all except Delphic and OGSRL's critic) never
read `dataset.rewards`."

The conclusion that per-mode dataset generation is somewhat wasteful is a fair
minor note, but the method count is not accurate against the current code:

- Delphic directly reads `dataset.rewards`.
- OGSRL does **not** read `dataset.rewards` in the live implementation. It fits
  dynamics/guardian from the dataset and reconstructs rollout rewards via
  `build_reward(self.env_cfg)` and predicted `next_states`.
- The remaining planning methods also reconstruct rewards through
  `build_reward(...)` / planner reward code.

So the current live-code statement is: **1 of 7 methods directly reads
`dataset.rewards` (Delphic), while the other 6 use transitions/beliefs and
mode-aware reconstructed rewards.**

Evidence to re-check:

- `src/real_ecology_benchmark/methods/delphic.py` uses `dataset.rewards`.
- `src/real_ecology_benchmark/methods/ogsrl.py` uses `build_reward(...)` and
  rollout rewards, not `dataset.rewards`.

### 2. Transition-sharing is not a first-run change

It is true that safe/yield transitions are identical for a fixed seed, behavior
policy, and environment cell; only logged rewards differ. But with the current
pipeline, `TrajectoryDataset` stores rewards and `ensure_dataset` validates the
whole reward-affecting cell. Sharing transitions once and recomputing
mode-specific rewards would require a new data-layer path that reads private
state/action trajectories and rewrites public rewards safely.

That could be a later optimization, but I do not think it belongs in the first
experiment-runner implementation. For the first run, reward-mode-specific dataset
paths are more robust and easier to audit; the validator remains the backstop.

### 3. Reducing PLUS `candidate_count` is not currently a no-config tweak

Claude's recommendation to reduce `candidate_count` is reasonable as a future
option, but it is not currently exposed in `BenchmarkConfig`, YAML, or CLI. The
no-code option is to split PLUS into its own longer-walltime array. Any
candidate-count reduction should be a separate code change and audit because it
changes the baseline itself.

## Net

No disagreement with the blocking action. The only correction is to avoid turning
the per-mode dataset note into runner behavior now, and to treat PLUS
candidate-count reduction as a separate scientific/configuration change rather
than a launch-script detail.
