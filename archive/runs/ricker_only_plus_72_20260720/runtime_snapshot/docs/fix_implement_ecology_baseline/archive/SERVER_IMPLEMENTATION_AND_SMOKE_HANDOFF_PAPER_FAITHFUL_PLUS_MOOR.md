# Paper-Faithful PLUS/MOOR Implementation and Smoke Handoff

## Verdict

The additive hidden-only PBVI implementation is integrated and passes its registered
160-transition interface smoke. It should be classified as **offline parametric
fitting of mechanistic models plus online Bayesian model selection**, not as the old
polynomial native approximation and not as exact reproduction of every source-paper
application detail.

The accepted smoke is
`real_ecology_runs/paper_faithful_smoke_20260717_v4/`, Slurm job `58351686`.

## Production changes

- `faithful_ecology.py`: fixed Ricker, Allee-Ricker, theta-logistic, and hidden-regime
  equations with fitted `(g,h,d,u)`, bounded capacity, process noise, and survey model.
- `faithful_fit.py`: ordered complete-episode Monte Carlo survey objective, PyTorch
  float64 autodiff L-BFGS, deterministic starts/common random numbers, action coverage,
  bounded transforms, holdout diagnostics, start traces, and curvature diagnostics.
- `faithful_pomdp.py`: candidate-consistent MC/barycentric transitions, exact public
  survey likelihood, deterministic capacity context, abundance/regime Bayes update,
  evidence, and shared public-history reward adapter.
- `planners/pbvi.py`: seeded finite-horizon reachable-belief/context backups carrying
  previous survey and timestep; no stationary shortcut and no QMDP fallback.
- `methods/plus_faithful.py`: fixed candidate bank, uniform/registered prior, separate
  candidate beliefs, public-only online likelihood update, posterior-weighted values.
- `methods/moor_faithful.py`: one controlled-Ricker fit and one consistent model used
  for filtering and planning.
- `faithful_artifacts.py`: no-pickle model, fit, complete discretized POMDP, planner,
  candidate-bank, hash, and privacy artifacts.
- Config, CLI, registry, pipeline, manifest scripts, frozen-smoke preparation, Slurm
  runner, acceptance checker, optional PyTorch extra, and isolated dependency lock were
  added without changing the existing native method IDs.

The existing `plus_native`, `moor_native`, `native_fit`, and `native_solver` modules
were not modified by this paper-faithful addition. The completed 20260716 hidden-r/K
run was not changed.

## Verification

- Full repository tests after the post-audit hardening: 128 passed.
- Ruff: clean on all new implementation, scripts, and tests.
- Table/privacy probe: public-data perturbation changes the fit; patched table readers
  and `NativeSolver.build` are never called; hidden artifacts pass the forbidden-name
  scan.
- Slurm `58351686`: two of two rows `COMPLETED`, exit `0:0`, zero stderr bytes.
- Both rows used the same 160-row public dataset hash.
- PLUS smoke artifacts contain four distinct forms/hashes and four POMDP artifacts;
  MOOR contains one Ricker/POMDP artifact.
- External probe records no DESPOT/SARSOP binary, so no external-solver ID is enabled.

## Post-audit hardening

Claude's implementation audit correctly identified two immediate gaps. Live source now
contains both halves of the ordered-episode contract: reversing transitions within an
episode changes the diagnostic objective, while permuting complete episodes leaves it
unchanged. The new external verifier
`scripts/hash_paper_faithful_snapshot.py` also publishes the exact snapshot-tree recipe.
Applied to the untouched v4 `code/` tree, it reproduces the registered digest
`a7836e85690316681296dbc5fe095fc195d2d1ee8108ad5a2b5e8ec740af386f` over 118 files.

Future PLUS artifacts and manifests explicitly label candidate construction as
`episode_bootstrap_map_v1`, and the method implementation version is
`plus_faithful_bootstrap_map_cross_form_v1`. These changes are live-source hardening for
the next frozen snapshot. Future MOOR manifest rows correctly record one fitted model
and no candidate prior. The completed v4 smoke tree and artifacts were not edited.

## Remaining gates

This is not authorization to present a new paper result. The accepted v4 smoke used one
candidate per form, so it demonstrates cross-form posterior updating but no
within-family parameter uncertainty. The next 4,000-target one-cell smoke must use four
candidates per form (16 total) and a newly frozen snapshot. The accepted registered
variant is deterministic complete-episode bootstrap MAP, not the plan's original
local-curvature/low-discrepancy proposal-and-rescore construction. The blinded 32-row
runtime canary has not run, and the CPU ceiling is not registered. Profile intervals and
the complete residual/noise-calibration diagnostic suite also remain before a headline
sweep.

DESPOT remains unavailable pending a pinned source/binary, complete license review,
and repeated invocation gate. The implemented MOOR method is therefore explicitly
named `_pbvi`; SARSOP remains deferred.
