# Reproducibility, seeds, caches, and provenance

There are no generic neural-network checkpoints in this repository. The
reproducible fitted state is carried by dataset/belief/surrogate/faithful-fit
caches, in-memory fitted policy objects, method artifacts, hashes, and receipts
(`src/tracks/general/real_ecology_benchmark/pipeline.py:run_method()` and
`faithful_artifacts.py`).

## Seed tree

```text
cfg.seed = 116
├─ collector RNG; draws each collection episode's env seed
├─ surrogate episode split = 20116
├─ belief-cache episode streams = 30116 + episode index
├─ policy/model construction = 40116
└─ train/holdout episode split = 50116

evaluation block seed + local episode
├─ environment = seed
├─ filter = seed + 10000
└─ policy reset = seed + 20000
```

Inside `ContinuousEcologyEnv._spawn_rngs()`, `SeedSequence(seed).spawn(5)`
creates separate parameters, regime, process, observation, and initial streams.
Faithful fitting makes fixed NumPy random banks from its candidate seed, then
uses deterministic torch float64 LBFGS (`faithful_fit.py`); it does not call a
global torch random sampler during the objective.

NumPy versus CuPy is resolved by `backend.py`. Strict mode prevents fallback;
non-strict fallback is recorded. BLAS/LAPACK implementation, thread count, CPU,
CuPy/CUDA version, and reductions can alter last bits. The accepted computation
profile is `configs/cluster/accepted-8452Y.yaml`; Python dict ordering is stable
in supported Python but is not the main numerical risk.

## Cache layers

| cache | key/check | staleness behavior |
|---|---|---|
| public/private dataset | caller path; partial metadata validation; public hash inside metadata | hidden validation omits population/family/P/safety/process noise |
| learned full filter | dataset stem `.learned_filter.npz`; exact env dict compatibility | public dataset hash not explicit |
| belief cache | dataset stem + expose/filter; native full adds bins | loaded with no compatibility check |
| public surrogate | dataset stem + version + split seed; validates public hash | mismatch refits and atomically replaces |
| faithful fit | `fit_cache_key()` includes transition hash, observation protocol, context, model/fit config, candidate/split/seed, torch runtime | accepted runner requires registered cache hits |

Per run, `pipeline.py` writes `episodes.csv`, `summary.json`,
`offline_beliefs.npz`, optional training CSV/JSON/PNG, and method-specific
faithful model/planner artifacts.

`provenance/frozen_tracks.sha256` covers frozen packages;
`scientific_inputs.sha256` covers canonical scripts, ecology CSVs and
experiments; `copied_artifacts.sha256` covers copied outputs;
`dataset_hashes.csv` maps accepted datasets; `frozen_snapshots.json` records
source snapshots. `scripts/verify_integrity.py` checks exact ledger coverage and
the accepted CSV hash.

## What is and is not reproducible

A parity-approved cell can be replayed bit-for-bit on a known compatible stack;
`VERIFY_REPORT.md` records exact seven-field parity for ecological A6 and
general B1. Repository code alone is insufficient for all accepted rows:
external public/private trajectory `.npz`, adapted fit caches/receipts, and
their directory layout are required. These live under a scratch project
selected by `SCRATCH_PROJECT`, `DEEPRL_GENERAL_DATA_ROOT`, and related variables
documented in `configs/paths.example.yaml`.

`recomputed_fits=0` means replay loaded the registered demographic candidates
instead of performing a new optimization; it protects parity and outcome-blind
reuse. `run_diagnostic_replay.py:recomputed_fit_count()` derives it from fit
diagnostics and `enforce_acceptance()` requires zero.

The parity fields are return mean, return SD, unsafe fraction mean, persistence
mean, collapse-entry mean, minimum-population mean, and economic-cost mean
(`summarize7()`), with PASS at maximum absolute difference <= `1e-9`.

## Commands

```bash
# full verification (needs external accepted data)
make verify SCRATCH_PROJECT=/path/to/Claude_DeepRL_Population_Models

# accepted ecological replay
make verify-cell VERIFY_CELL=A6 VERIFY_SIDE=moor \
  SCRATCH_PROJECT=/path/to/Claude_DeepRL_Population_Models

# accepted general replay
make verify-cell-general VERIFY_GENERAL_CELL=B1 \
  VERIFY_GENERAL_METHOD=ensemble_value_disagreement_pessimism

# tests
make test-ecological
make test-general

# standalone smoke
PYTHONPATH=src/tracks/general python -m real_ecology_benchmark smoke \
  --config configs/tracks/general/real_smoke.yaml
```

Bit-level nondeterminism remains possible outside the accepted software/hardware
profile, especially for threaded BLAS or CuPy. A summary-level parity pass is
stronger than approximate replication but does not prove every intermediate
array is identical.
