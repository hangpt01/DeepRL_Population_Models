# Experiment Protocol and Reproducibility

Run commands from the repository root.

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/synthetic_default.yaml
```

The same pipeline serves real, dummy, and synthetic settings:

```text
generate -> run --method <method> -> gate -> aggregate
```

## Config Inventory

| Config group | Purpose |
| --- | --- |
| `configs/real_*` | real ecology cells backed by `real_ecology_data/` |
| `configs/dummy_setpoint_*` | small in-code set-point/cumulative experiments |
| `configs/synthetic_*` | one-step synthetic-control defaults and full settings |
| `configs/cumulative_controls_12h.yaml` | cumulative-control synthetic stress run |

The CLI can override the data source for compatible configs with `--data-mode`.
For example, a real smoke config can be redirected to dummy mode when the
population/action requirements are also compatible.

## Row Flow

1. Load and validate the YAML config.
2. Resolve real, dummy, or synthetic environment defaults.
3. Resolve the action table and action hash.
4. Load or generate a public dataset and private sidecar.
5. Fit or load the requested public belief/filter cache.
6. Fit the selected policy.
7. Evaluate with paired seeds and write per-episode plus summary metrics.
8. Optionally run the gate and aggregate manifest outputs.

## Methods

The method registry contains seven identifiers:

```text
mopo, refplan, bamcts, plus, moor, delphic, ogsrl
```

See `04_algorithm_adaptations_and_claims.tex` before interpreting those names.

## Outputs

Real/set-point output paths are namespaced by effective backend, data mode, and
reward mode so CPU/GPU, real/dummy, and safe/yield rows do not overwrite each
other. A typical path contains components like:

```text
backend_numpy/data_real/reward_safe/<method>/<filter>/
```

Public datasets are compressed NumPy files; private truth is written separately.
Evaluation writes `episodes.csv`, `summary.json`, and optional training-history
artifacts.

## Manifests and Slurm

Manifest helpers in `../../scripts/` create experiment rows. Slurm wrappers in
`../../scripts/slurm/` execute one row at a time and are intended to be
restartable. Real matrix generation records population, family, observation
noise, reward mode, filter mode, method, and backend fields so aggregation can
avoid pooling incompatible rows.

## Seeds and Atomicity

The environment uses separated NumPy seed streams for parameters, regime,
process noise, observation noise, and initial state. Dataset writes, learned
proposal writes, and belief-cache writes are atomic. Manifest workers use locks
around shared dataset creation so concurrent rows do not corrupt shared
artifacts.

## Provenance Bundle

The frozen real-ecology P-safe run is stored at
`../../real_ecology_runs/psafe_overnight_20260705/`. Do not modify it. Aggregate
new analyses with the live package against that frozen data tree when needed.

Verified against: docs/experiment_protocol.md, docs/reproducibility.md, Makefile, README.md, scripts/, scripts/slurm/, src/real_ecology_benchmark/cli.py, src/real_ecology_benchmark/pipeline.py, src/real_ecology_benchmark/manifest.py, real_ecology_runs/psafe_overnight_20260705/analysis/PAPER_RESULT_PACKAGE.md @ be96c36
