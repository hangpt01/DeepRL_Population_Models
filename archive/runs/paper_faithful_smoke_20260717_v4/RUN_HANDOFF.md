# Paper-Faithful PLUS/MOOR Smoke Handoff

## Accepted run

- Root: `real_ecology_runs/paper_faithful_smoke_20260717_v4/`
- Slurm array: `58351686` (`0` PLUS, `1` MOOR)
- Status: both `COMPLETED`, exit `0:0`; elapsed 17 s / 11 s; peak batch RSS
  220,332 KiB / 222,940 KiB; stderr is empty.
- Scope: interface/serialization smoke only, 160 public target transitions, one Amur
  tiger / private-Ricker / safe / sigma=0.1 cell. It is not a headline result.
- Shared public dataset SHA-256:
  `1e382875fc707fb205871fadbdb6eb5e894eab2b949b3a32867afae2151a5b9e`.

The exact frozen source is `code/`; provenance is in
`manifests/registration.json`; the registered rows are `manifests/smoke.csv`.
`acceptance.json` is return-blind and verifies finite fits, planner invocation,
privacy records, and one POMDP artifact per fitted candidate.

## What ran

- `plus_faithful_pbvi`: four smoke candidates, one fitted independently for each
  registered form; uniform initial posterior; separate abundance beliefs; public
  observation likelihood updates; posterior-weighted candidate PBVI values.
- `moor_faithful_ricker_misspec_pbvi`: one ordered-episode fitted controlled-Ricker
  model with the same PBVI approximation, explicitly misspecified outside Ricker.
- Both methods received only hidden-mode `MethodContext` plus the sanitized public
  dataset. Their fit artifacts record 8 fit episodes and 2 reporting-only holdout
  episodes. They share no table-built solver/filter.

## Audit commands

```bash
sacct -j 58351686 --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,AllocCPUS -P
.venv-paper-faithful/bin/python \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/code/scripts/run_paper_faithful_acceptance.py \
  --root real_ecology_runs/paper_faithful_smoke_20260717_v4 --expected-rows 2
sha256sum real_ecology_runs/paper_faithful_smoke_20260717_v4/{acceptance.json,manifests/registration.json,manifests/smoke.csv,provenance/solver_probe.json}
```

## Deliberate stop

No blinded canary or full sweep was submitted. Before either:

1. Set the preregistered total CPU-hour ceiling and fixed full-versus-core rule.
2. Decide whether episode-bootstrap MAP candidates are accepted for the initial PLUS
   bank or whether the planned curvature/low-discrepancy proposal stage is mandatory.
3. Finish/profile the registered identifiability diagnostics (profile intervals and
   residual calibration are not yet complete).
4. DESPOT is unavailable and unlicensed in this environment; therefore no `_despot`
   method ID is registered. SARSOP remains deferred. PBVI is the tested path.
5. Run the 4,000-target one-cell registered smoke before the 32-row blinded canary.

The roots without `_v4` and with `_v2`/`_v3` are retained but contain explicit
`SUPERSEDED.md` files and are not acceptance inputs.
