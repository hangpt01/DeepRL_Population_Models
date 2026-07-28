# Claude Audit Request: Paper-Faithful PLUS/MOOR Implementation

Please audit the implementation and executed smoke independently. Do not trust this
summary as evidence: inspect the frozen v4 source, Slurm accounting, logs, public
datasets, fitted artifacts, and tests directly.

## Read first

1. `docs/fix_implement_ecology_baseline/ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md`
2. `docs/fix_implement_ecology_baseline/PAPER_FAITHFUL_PLUS_MOOR_IMPLEMENTATION_PLAN.md`
3. `docs/fix_implement_ecology_baseline/CLAUDE_AUDIT_PAPER_FAITHFUL_PLUS_MOOR_PLAN.md`
4. `docs/fix_implement_ecology_baseline/SERVER_IMPLEMENTATION_AND_SMOKE_HANDOFF_PAPER_FAITHFUL_PLUS_MOOR.md`
5. `real_ecology_runs/paper_faithful_smoke_20260717_v4/RUN_HANDOFF.md`

## What Codex implemented

### New mechanistic implementation

- `src/real_ecology_benchmark/faithful_ecology.py`
  - Explicit normalized Ricker, Allee-Ricker, theta-logistic, and two-state
    regime-switching depensation equations.
  - Per-action nonnegative growth, mortality, capacity increment, and stocking
    parameters `(g,h,d,u)`.
  - Deterministic bounded capacity update, process noise, public log-normal survey
    likelihood, and explicit zero/stocking behavior.
  - No imports from real tables, action tables, `EnvironmentConfig`, evaluator, or
    private sidecars.

- `src/real_ecology_benchmark/faithful_fit.py`
  - Complete-episode ordered trajectory fitting rather than exchangeable transition
    regression.
  - PyTorch float64 automatic differentiation with `torch.optim.LBFGS`, deterministic
    starts, fixed common random numbers, and bounded transforms.
  - Shared fitted reset distribution, capacity bounds, process scale, all action
    effects, family-specific parameters, action coverage/sparsity reporting, fit and
    holdout episode IDs, normalized holdout survey error, objective traces, gradient
    norms, and an L-BFGS curvature-condition diagnostic.
  - PLUS bank currently uses independently fitted episode-bootstrap MAP candidates.
    Uniform prior is the registered default and candidate numerical diversity is
    enforced.

- `src/real_ecology_benchmark/faithful_pomdp.py`
  - The same fitted candidate object drives transition construction, observation
    likelihood, belief updates, likelihood evidence, capacity evolution, and planner
    rewards.
  - Monte Carlo transition discretization with sparse barycentric projection.
  - Separate abundance/regime belief per candidate and deterministic candidate
    capacity context.
  - Shared public reward surrogate receives previous/current/following public survey,
    timestep, action/cost, and opaque population token.

- `src/real_ecology_benchmark/planners/base.py`
- `src/real_ecology_benchmark/planners/pbvi.py`
  - Seeded finite-horizon reachable-belief/context PBVI approximation.
  - Previous observation and timestep remain in the planning context.
  - No QMDP or stationary `t=0` shortcut.

- `src/real_ecology_benchmark/faithful_artifacts.py`
  - Versioned no-pickle model, fit, candidate-bank, complete discretized POMDP,
    planner, hash, and privacy artifacts.
  - Saves `faithful_fit.npz/json`, candidate files, candidate bank for PLUS,
    `pomdp_model_*.npz/json`, `pbvi_policy_diagnostics.npz`, planner provenance, and
    privacy audit.

### New policies

- `src/real_ecology_benchmark/methods/plus_faithful.py`
  - Registered ID: `plus_faithful_pbvi`.
  - Fits one or more candidates independently for each of four forms from public
    ordered histories.
  - Uniform initial model posterior by default.
  - Candidate parameters remain fixed online; abundance/regime beliefs, deterministic
    capacity contexts, and candidate posterior update from public action/observation
    history only.
  - Action is selected from posterior-weighted candidate PBVI values.

- `src/real_ecology_benchmark/methods/moor_faithful.py`
  - Registered ID: `moor_faithful_ricker_misspec_pbvi`.
  - Fits one controlled-Ricker model in every private simulator family.
  - Does not inspect the private family; misspecification is explicit in the method ID.

### Additive integration

- `src/real_ecology_benchmark/config.py`
  - Added faithful model, fit, and planner configurations and validation.
- `src/real_ecology_benchmark/methods/base.py`
  - Added a no-op artifact hook inherited unchanged by old policies.
- `src/real_ecology_benchmark/methods/__init__.py`
  - Added the two PBVI IDs and `FAITHFUL_METHODS`; old mappings remain.
- `src/real_ecology_benchmark/pipeline.py`
  - New IDs are hidden-only, require `faithful_internal`, receive `MethodContext`,
    skip the legacy pre-fit belief cache, and let the faithful fitter own the one
    registered 80/20 episode split.
  - Saves faithful artifacts after evaluation.
- `src/real_ecology_benchmark/cli.py`
  - Added the faithful filter route.
- `pyproject.toml`
  - Added optional `paper-faithful-fit` PyTorch dependency.
- `.gitignore`
  - Ignores the isolated `.venv-paper-faithful/` environment.

Existing `plus_native.py`, `moor_native.py`, `native_fit.py`, and `native_solver.py`
were not edited as part of this addition. The frozen 20260716 hidden-r/K comparison
was not modified.

## Configs, scripts, and tests added

- `configs/paper_faithful_hidden.yaml`
- `configs/paper_faithful_hidden_smoke.yaml`
- `scripts/make_paper_faithful_manifest.py`
- `scripts/prepare_paper_faithful_smoke.py`
- `scripts/run_paper_faithful_acceptance.py`
- `scripts/run_paper_faithful_solver_probe.py`
- `scripts/slurm/run_paper_faithful_row.sh`
- `docs/fix_implement_ecology_baseline/paper_faithful_fit_requirements.lock`
- `tests/real/faithful_fixtures.py`
- `tests/real/test_faithful_equations.py`
- `tests/real/test_faithful_planners.py`
- `tests/real/test_faithful_artifacts.py`
- `tests/real/test_faithful_privacy.py`
- `tests/real/test_moor_faithful.py`
- `tests/real/test_plus_faithful.py`

The legacy all-method tests were scoped to skip hidden-only faithful IDs; the faithful
methods have dedicated tests instead.

## Dependency environment

An isolated environment was created at `.venv-paper-faithful/` with CPU PyTorch.
The exact package record is:

`docs/fix_implement_ecology_baseline/paper_faithful_fit_requirements.lock`

Important versions include Python 3.10.14, NumPy 2.2.6, PyYAML 6.0.3,
PyTorch 2.13.0+cpu, pytest 9.1.1, and Ruff 0.12.12.

## Verification performed before submission

```bash
PYTHONPATH=src .venv-paper-faithful/bin/python -m pytest -q
.venv-paper-faithful/bin/python -m ruff check \
  src/real_ecology_benchmark/faithful_ecology.py \
  src/real_ecology_benchmark/faithful_fit.py \
  src/real_ecology_benchmark/faithful_pomdp.py \
  src/real_ecology_benchmark/faithful_artifacts.py \
  src/real_ecology_benchmark/planners \
  src/real_ecology_benchmark/methods/moor_faithful.py \
  src/real_ecology_benchmark/methods/plus_faithful.py
```

Observed: 126 tests passed; Ruff passed.

The privacy/data-dependence tests patch table readers, action resolution, and
`NativeSolver.build` to fail. Fitting still succeeds. Perturbing sanitized public
transitions changes fitted parameter hashes.

## Canonical job and artifacts

Use only:

`real_ecology_runs/paper_faithful_smoke_20260717_v4/`

- Accepted Slurm array: `58351686`
- Array row 0: `plus_faithful_pbvi`
- Array row 1: `moor_faithful_ricker_misspec_pbvi`
- Both: `COMPLETED`, exit `0:0`
- Elapsed: 17 s and 11 s
- Batch peak RSS: 220,332 KiB and 222,940 KiB
- Stderr: zero bytes
- Cell: Amur tiger, private simulator Ricker, safe reward, sigma=0.1
- Public target/actual rows: 160/160
- Public dataset SHA-256:
  `1e382875fc707fb205871fadbdb6eb5e894eab2b949b3a32867afae2151a5b9e`
- Frozen source commit: `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`
- Frozen snapshot tree SHA-256:
  `a7836e85690316681296dbc5fe095fc195d2d1ee8108ad5a2b5e8ec740af386f`
- Manifest SHA-256:
  `4a14a5d6912175ee1badda63d30d2f0fb40ca54793efc89ecebe117adff38e03`

Key files:

- `real_ecology_runs/paper_faithful_smoke_20260717_v4/code/`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/manifests/registration.json`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/manifests/smoke.csv`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/provenance/solver_probe.json`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/acceptance.json`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/completion.json`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/logs/`
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/evaluation/`

The PLUS smoke contains four candidates total, one per form, and four matching POMDP
artifacts. MOOR contains one Ricker candidate/POMDP. Every candidate records 8 fit
episodes and 2 reporting-only holdout episodes.

## Audit commands

```bash
sacct -j 58351686 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,AllocCPUS -P

.venv-paper-faithful/bin/python \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/code/scripts/run_paper_faithful_acceptance.py \
  --root real_ecology_runs/paper_faithful_smoke_20260717_v4 \
  --expected-rows 2

sha256sum \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/acceptance.json \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/manifests/registration.json \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/manifests/smoke.csv \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/provenance/solver_probe.json
```

## Superseded jobs

Do not use these for acceptance:

- `58351609`: exposed a duplicate outer/inner holdout split.
- `58351632`: corrected split, then superseded by tightened parameter bounds and
  candidate-diversity enforcement.
- `58351647`: correct fit semantics, then superseded by complete POMDP/planner
  artifacts and richer fit diagnostics.

Their roots contain `SUPERSEDED.md`. They are retained only for forensic history.

## Important limitations and deliberate stop

This was an interface/serialization smoke, not a new headline experiment. Codex did
not submit the 32-row blinded canary or a full sweep.

Before headline execution, audit and resolve:

1. The total CPU-hour ceiling and fixed full-versus-balanced-core rule are unset.
2. The full config uses 16 PLUS candidates, but the real smoke used four total.
3. Candidate construction currently uses deterministic episode-bootstrap MAP fits.
   The plan's richer local-curvature/low-discrepancy proposal-and-rescore stage is not
   implemented; decide whether it is mandatory before calling the bank fully faithful.
4. Start traces, action coverage, holdout error, process scale, and an L-BFGS curvature
   diagnostic are present. Full profile intervals and the complete residual/noise
   calibration suite remain incomplete.
5. DESPOT, SARSOP, and `pomdpsol` are absent. No `_despot` or `_sarsop` ID is
   registered. The implemented MOOR path is explicitly `_pbvi`.
6. The required 4,000-target one-cell smoke has not run yet.

Please report findings first, ordered by severity, with exact frozen `file:line`
references and artifact paths. Distinguish implementation correctness, privacy/fairness,
paper faithfulness, and experiment-execution validity.
