# Practical 3–4 hour code-reading plan

## 0:00–0:15 — orientation

Files: `01_BIG_PICTURE_FLOW.md`, `src/tracks/general/real_ecology_benchmark/{cli,pipeline}.py`,
`methods/__init__.py`.

Inspect: `cmd_run()`, `run_method()`, `build_method()`, `METHODS`.

Question/outcome: identify the sole offline fit and the immediate online
evaluation; know why this is not deep-RL training.

```bash
rg -n 'def cmd_run|def run_method|policy.fit|evaluator.run|METHODS =' \
  src/tracks/general/real_ecology_benchmark
```

## 0:15–0:45 — scientific environment

Files: `envs.py`, `controls.py`, `actions.py`, `reward.py`,
`observation.py`, `configs/ecology/{species,actions,action_effects_long}.csv`.

Inspect: `transition_value()`, `step()`, `advance_public_controls()`,
`real_action_table()`, reward methods.

Question/outcome: derive all four maps, action effects, observation and exact
scoring reward.

```bash
PYTHONPATH=src/tracks/general python -m unittest \
  tests.general.real.test_real_ecology -v
```

## 0:45–1:10 — fixed offline data

Files: `collector.py`, `dataset.py`, `pipeline.py`.

Inspect: `MixedDangerZonePolicy.act()`, `collect_dataset()`,
`TrajectoryDataset.validate()`, `ensure_dataset()`,
`_validate_dataset_cell()`.

Question/outcome: distinguish public truth-derived reward from private state and
locate cache-validation gaps.

```bash
rg -n 'true_state if|truth_derived_public_signal|PRIVATE_FIELDS|expected = ' \
  src/tracks/general/real_ecology_benchmark/{collector,dataset,pipeline}.py
```

## 1:10–1:40 — beliefs and privacy

Files: `config.py`, `beliefs.py`, `public_surrogate.py`,
`methods/base.py`, `privacy.py`.

Inspect: `MethodContext`, `PublicObservationFilter`,
`cache_public_beliefs()`, `fit_public_surrogate()`, BasePolicy privacy guard.

Question/outcome: enumerate exactly what hidden methods know and why oracle is
evaluation-only.

```bash
PYTHONPATH=src/tracks/general python -m unittest \
  tests.general.real.test_general_privacy tests.general.real.test_faithful_privacy -v
```

## 1:40–2:20 — two methods deeply

Read one adapted method:
`faithful_fit.py`, `faithful_pomdp.py`, `planners/pbvi.py`,
`methods/plus_faithful.py`; and one general method:
`methods/ensemble_value_disagreement.py` or `refplan.py`.

Inspect: `_trajectory_objective()`, `fit_mechanistic_model()`,
`PointBasedPlanner.action_values()`, EVD `_fit_member()/act()` or RefPlan
`fit()/act()/observe()`.

Question/outcome: name the fitted object, deployed selector, uncertainty term,
and reward channel.

```bash
rg -n 'objective.backward|surrogate.predict|dataset.rewards|def act' \
  src/tracks/general/real_ecology_benchmark/{faithful_fit.py,faithful_pomdp.py,methods}
```

## 2:20–2:45 — pipeline end to end

Files: numbered `pipeline.py` and `training_monitor.py`.

Inspect: `run_method()` lines 378–557, `split_train_holdout()`,
`record_final_fit_metrics()`.

Question/outcome: trace command → cache → fit → evaluator → artifacts and every
seed offset.

```bash
nl -ba src/tracks/general/real_ecology_benchmark/pipeline.py | sed -n '378,557p'
```

## 2:45–3:10 — evaluation and accepted table

Files: `evaluator.py`, `manifest.py`,
`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` and receipt.

Inspect: `ContinuousEvaluator.run/summarize()`, `aggregate_summaries()`.

Question/outcome: reconstruct operational return, sample SD, fallbacks, and the
144-row factorial coverage.

```bash
python - <<'PY'
import csv,collections
r=list(csv.DictReader(open("results/accepted/MATCHED_P10_144_METHOD_CELLS.csv")))
print(len(r),collections.Counter(x["method"] for x in r))
print({(x["population"],x["environment"],x["sigma_obs"]) for x in r})
PY
```

## 3:10–3:35 — configs, tracks, provenance

Files: accepted configs/manifests, `scripts/general/run_real_manifest_row.py`,
`provenance/*`, `Makefile`.

Inspect: `apply_row_config()`, cache/receipt gates, integrity script.

Question/outcome: know which layer is authoritative and what external inputs
exact parity needs.

```bash
diff -rq --exclude=__pycache__ src/tracks/ecological src/tracks/general
python scripts/verify_integrity.py
```

## 3:35–4:00 — scientific challenge

Files: `14_SCIENTIFIC_VALIDITY_RISKS.md`,
`src/diagnostics/replay_analysis/{metrics,constants}.py`,
`scripts/diagnostics/followups/run_s2.py`.

Inspect: M7/M13/M14, `max_reachable_abundance()`,
`belief_mdp_policy_candidates()`, regret sign.

Question/outcome: separate reproducibility from validity and identify the first
experiments needed to support a four-family uncertainty claim.

```bash
python scripts/verify_constants.py
make verify-self-tests verify-negative-gate
```

