# Ecological offline MBRL benchmark

This repository's active project is a self-contained continuous-state,
noisy-observation ecological adaptive-management benchmark. The runnable package
is `src/real_ecology_benchmark/`, with its real data tables in
`real_ecology_data/`; it has no runtime dependency on any other directory in
this repository.

The current top-level package is `real_ecology_benchmark`. It supports:

- `data_mode: real`, the paper-facing real-ecology table setting backed by
  `real_ecology_data/`;
- `data_mode: dummy`, a small in-code synthetic profile for quick experiments;
- shared `control_mode: setpoint_cumulative` semantics in both modes: action-specific
  set-point `r` and cumulative public `K`;
- continuous-state synthetic experiments for one-step and cumulative-control
  settings.

Real, dummy, and synthetic experiments run through the same
`real_ecology_benchmark` package, so they share the dataset format, methods,
evaluator, gate, and aggregation code.

Core setting:

- continuous unbounded latent abundance `s >= 0`, with exact extinction terminal;
- hidden episode parameters and optional switching regime;
- fixed 5- and 10-action management tables;
- direct harvest/stocking authority before growth;
- log-normal observations at `sigma_obs in {0, 0.1, 0.2, 0.4}`;
- observed-abundance reward and one-time latent safety-entry penalty;
- trajectory-preserving public datasets and evaluator-only truth sidecars;
- shared known-emission particle filtering, raw, method-Ricker, and oracle-state ablations;
- MOPO, RefPlan, BA-MCTS, PLUS, MOOR, Delphic-CQL, and OGSRL implementations;
- paired evaluation, decision-relevance gate, calibration, manifests, and aggregation.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
ecology-benchmark smoke --config configs/synthetic_default.yaml
ecology-benchmark generate --config configs/synthetic_default.yaml --transitions 75000
ecology-benchmark run --config configs/synthetic_default.yaml --method mopo
ecology-benchmark gate --config configs/synthetic_default.yaml
python scripts/run_gate_matrix.py --config configs/synthetic_full.yaml
```

Without installation:

```bash
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml --data-mode dummy
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests/real -v
PYTHONPATH=src python -m unittest discover -s tests/dummy -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/synthetic_default.yaml
```

The full `unittest discover -s tests` command includes real, dummy, and synthetic suites.
The synthetic default config is intentionally small enough for a laptop smoke test. `configs/synthetic_full.yaml` contains the final-budget synthetic defaults. Generated public data and private truth are separate files; method training accepts only the public file.

The Slurm worker requires passing hard-gate artifacts for Allee/regime at primary noise levels by default. Set `REQUIRE_GATE=false` only for explicit diagnostic/stress runs.

## Repository layout

- `src/real_ecology_benchmark/`: active benchmark package.
- `configs/`: real, dummy, and synthetic YAML experiment defaults.
- `tests/`: real, dummy, and synthetic regression suites.
- `scripts/`: manifest, plotting, repair, and Slurm helpers.
- `real_ecology_data/`: authoritative real-ecology CSV tables.
- `real_ecology_runs/`: tracked provenance bundle for the P-safe overnight run.
- `docs/benchmark/`: canonical reader-facing benchmark documentation.
- `docs/history/` and `docs/real_ecology_history/`: historical planning/audit notes.
- `baseline_original/`: historical two-state solver preserved for attribution.

## Scientific guardrails

- No finite abundance bins, dense finite-MDP tensors, `s_max`, or silent state clipping.
- The primary behavior policy acts on private true abundance; this is deliberate hidden action-outcome confounding. An observation-only collector is available as an ablation.
- The shared primary filter never calls simulator equations. Mechanistic proposal/filter-bank variants are labeled baseline-fidelity ablations.
- The `-20` entry penalty is the only truth-derived bit exposed through public history.
- Delphic uncertainty is variation over posterior-ambiguity-conditioned compatible latent worlds, not ordinary bootstrap disagreement; `sigma_obs=0` is an explicit negative control.
- OGSRL has separate OOD and ecological safety constraints, posterior-particle risk estimates, and a hard deployment guardian/fallback.
- The policy implementations are auditable NumPy adaptations of algorithmic
  ideas in a linear/mechanistic model class, not reproductions of the published
  deep-RL systems.

See `docs/benchmark/` for the current problem setting, method adaptations, run
protocol, results, and limitations.

## Historical baseline

`baseline_original/` preserves the earlier universal two-state, n-action
adaptive-management solver. It is not part of the active continuous-state
benchmark runtime, but remains in the repository as historical/reference code.
