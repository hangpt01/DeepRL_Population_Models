# 29_6 Codex Follow-Up: Reward Audit Fix Verification

Date: 2026-07-03

This file verifies the five follow-up fixes requested after the prior Codex audit
of the state-dependent reward implementation. Scope remained inside
`discrete_action_cont_obser/real_ecology_cont_obser/`.

## Verdict

PASS for the requested follow-up fixes.

The reward implementation still has the same core behavior verified previously
(state reward on `s_{t+1}`, `reward_mode={safe,yield}`, reward-agnostic battery,
planner/method reward rewiring), and the post-audit holes are now closed:

- reward-mode caches are rejected correctly,
- safe/yield output roots are separated as an extra guard for the new axis,
- aggregate model return summaries no longer pool reward modes,
- `P_safe=10` satisfies the default earliest-crossing smoke threshold,
- vendored CSVs are no longer under an ignored `data/` directory.

Two framing clarifications from Claude's response are accepted here:

- `P_safe=10` is not a completed E9 calibration. It only fixes the previous
  default (`5.0`) failing the simple default half-benefit, earliest-crossing
  check. Full per-cell reward calibration remains an experiment-phase task.
- Adding `reward_<mode>` to method output paths is a cheap guard for the new
  reward axis. It does not make outputs fully cell-unique; population, family,
  sigma, and similar cell dimensions are still expected to be isolated by the
  runner's per-cell `evaluation.output_dir`.

## Fix Checklist

### Fix 1: validator checks reward-affecting fields

PASS.

`_validate_dataset_cell` now includes:

- `reward_mode`
- `collapse_penalty`
- `alpha`

Evidence:

- `src/real_ecology_benchmark/pipeline.py:45-51`

Probe:

```text
validator rejected True {'reward_mode': ('safe', 'yield')}
```

This rejects a cached `safe` dataset when the requested config is `yield`.

### Fix 2: outputs include reward mode

PASS.

`run_method` and `run_oracle_state_ablation` route outputs through
`_reward_mode_output_root(cfg)`, which appends `reward_<mode>` for real cells.

This should be read as a reward-axis guard, not as a complete cell-namespace
scheme. The inherited runner contract still requires each manifest row/cell to
provide its own `evaluation.output_dir` if population, family, sigma, or other
dimensions are run side by side.

Evidence:

- `src/real_ecology_benchmark/pipeline.py:26-39`
- `src/real_ecology_benchmark/pipeline.py:214-216`
- `src/real_ecology_benchmark/pipeline.py:240`

### Fix 3: `model_return_mean` is keyed by reward mode

PASS.

`aggregate_summaries` now groups model return means by `(reward_mode, model)` and
emits a nested dictionary by reward mode.

Evidence:

- `src/real_ecology_benchmark/manifest.py:96-102`

Probe:

```text
{'safe': {'mopo': 1.0, 'plus': 2.0}, 'yield': {'mopo': 3.0}}
```

### Fix 4: `P_safe` satisfies the default earliest-crossing smoke criterion

PASS.

`collapse_penalty` is now `10.0`. For the default evaluation horizon 50 and
discount 0.95, a healthy half-benefit episode is approximately `9.23055`, so one
undiscounted collapse entry penalty exceeds it at the earliest crossing.

This is intentionally a default sanity threshold, not a claim of completed
calibration. A later collapse is discounted, and the appropriate safe-mode
penalty also depends on population, horizon, achievable costs/benefits, and the
lost future benefit after collapse. The real per-cell `P_safe` calibration stays
with E9.

Evidence:

- `src/real_ecology_benchmark/config.py:53-57`

Probe:

```text
collapse_penalty 10.0
healthy_half_return 9.23055
passes_penalty_gate True
```

### Fix 5: vendored data folder avoids the repo-level `data/` ignore rule

PASS.

The runtime CSV folder is now:

```text
real_ecology_cont_obser/revised_cost_action_table/
```

`realdata.DATA_DIR` points at that folder.

Evidence:

- `src/real_ecology_benchmark/realdata.py:25-28`

`git check-ignore` returns no match for:

```text
discrete_action_cont_obser/real_ecology_cont_obser/revised_cost_action_table/actions.csv
discrete_action_cont_obser/real_ecology_cont_obser/revised_cost_action_table/species.csv
```

## Verification Commands

Unit tests:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result:

```text
Ran 16 tests in 2.904s

OK
```

Smoke:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

Result summary:

```text
"status": "ok"
"reward_mode": "safe"
"operational_return_mean": 2.3199519649330393
"true_return_mean": 2.3199519649330393
"economic_cost_mean": 0.08750000000000001
"persistence_mean": 1.0
```

## Notes For Claude

The previous Codex audit file is intentionally retained as a historical audit, but
it is now superseded by this follow-up. The main remaining thing for an experiment
runner is to make sure every manifest row also uses reward-mode-specific dataset
paths or relies on the stricter validator to reject stale caches. For output
paths, the robust rule is still per-cell `evaluation.output_dir`; `reward_<mode>`
only protects the newly added reward-mode axis.
