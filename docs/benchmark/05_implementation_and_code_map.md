# Implementation and Code Map

This file explains the engineering choices and where each concept lives in the
code. The scientific rationale is in `01_problem_setting_and_design.tex`; this
file avoids restating it.

## Engineering Rationale

`data_mode` is orthogonal to `control_mode` because the data source and the
action semantics are different axes. Real and dummy data both use
`setpoint_cumulative`; they differ only in whether populations/actions/effects
come from CSVs or from `dummydata.py`. Synthetic settings retain their own
control identifiers for regression and stress experiments.

The public/private data boundary exists to keep the partial-observation problem
honest. Public data contains survey observations, actions, rewards as logged
targets, next observations, done flags, timesteps, episode IDs, and public
control fields. Private sidecars contain true abundance and hidden simulator
state for evaluation and audit only. Training APIs do not accept private files.

Dataset-cache validation keys include the environment fields that change action
tables, rewards, safety metrics, observation noise, data source, and control
semantics. This prevents accidentally training a method on a public dataset from
one ecological cell while evaluating it as another.

The backend abstraction records both requested and effective compute backends.
It refuses to label a NumPy run as GPU work unless the workload is actually
implemented on the accelerated path. This keeps timing and fairness claims
auditable.

## Code Map

| Concept | Files |
| --- | --- |
| config loading and validation | `../../src/real_ecology_benchmark/config.py` |
| real data loader | `../../src/real_ecology_benchmark/realdata.py` |
| dummy data source | `../../src/real_ecology_benchmark/dummydata.py` |
| action tables and action hashes | `../../src/real_ecology_benchmark/actions.py` |
| public controls | `../../src/real_ecology_benchmark/controls.py` |
| environment transition | `../../src/real_ecology_benchmark/envs.py` |
| reward channels and safety indicator | `../../src/real_ecology_benchmark/reward.py` |
| public/private dataset schema | `../../src/real_ecology_benchmark/dataset.py` |
| dataset collection | `../../src/real_ecology_benchmark/collector.py` |
| observation model | `../../src/real_ecology_benchmark/observation.py` |
| belief filters and mechanistic proposals | `../../src/real_ecology_benchmark/beliefs.py` |
| learned dynamics ensemble | `../../src/real_ecology_benchmark/dynamics.py` |
| shared particle MPC | `../../src/real_ecology_benchmark/planning.py` |
| method implementations | `../../src/real_ecology_benchmark/methods/` |
| pipeline and CLI | `../../src/real_ecology_benchmark/pipeline.py`, `../../src/real_ecology_benchmark/cli.py` |
| evaluation and gate | `../../src/real_ecology_benchmark/evaluator.py`, `../../src/real_ecology_benchmark/gate.py` |
| manifests | `../../src/real_ecology_benchmark/manifest.py`, `../../scripts/` |
| Slurm entry points | `../../scripts/slurm/` |
| compute backend matrix | `../../src/real_ecology_benchmark/backend.py` |
| tests | `../../tests/real/`, `../../tests/dummy/`, `../../tests/synthetic/` |

## Package Layout

The runnable package is `real_ecology_benchmark`. The root-level `src/` and
`real_ecology_data/` directories are siblings; that invariant is load-bearing
because `realdata.DATA_DIR` resolves the CSV folder relative to the package root.

Configs live in `../../configs/`. Real configs use `real_*`, dummy configs use
`dummy_setpoint_*`, and synthetic configs use `synthetic_*` plus the
`cumulative_controls_12h` stress setting.

Verified against: src/real_ecology_benchmark/config.py, src/real_ecology_benchmark/realdata.py, src/real_ecology_benchmark/dummydata.py, src/real_ecology_benchmark/actions.py, src/real_ecology_benchmark/controls.py, src/real_ecology_benchmark/envs.py, src/real_ecology_benchmark/reward.py, src/real_ecology_benchmark/dataset.py, src/real_ecology_benchmark/collector.py, src/real_ecology_benchmark/beliefs.py, src/real_ecology_benchmark/dynamics.py, src/real_ecology_benchmark/planning.py, src/real_ecology_benchmark/pipeline.py, src/real_ecology_benchmark/cli.py, src/real_ecology_benchmark/backend.py, tests/ @ be96c36
