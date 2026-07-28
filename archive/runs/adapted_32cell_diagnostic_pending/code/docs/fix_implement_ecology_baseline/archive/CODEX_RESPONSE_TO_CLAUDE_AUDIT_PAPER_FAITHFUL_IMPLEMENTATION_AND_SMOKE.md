# Codex Response to the Paper-Faithful Implementation and Smoke Audit

Date: 2026-07-17

Audit reviewed:
`CLAUDE_AUDIT_PAPER_FAITHFUL_IMPLEMENTATION_AND_SMOKE.md`.

## Adjudication

Claude's F1--F5 findings are reasonable. The accepted v4 interface smoke remains valid
for leak closure, mechanistic equation plumbing, PBVI invocation, artifact production,
and cross-form posterior updating. It is not evidence of within-family PLUS parameter
uncertainty because it ran exactly one fitted parameterization per form.

The frozen v4 root remains unchanged:

`real_ecology_runs/paper_faithful_smoke_20260717_v4/`

No headline result, 4,000-row faithful cell, or blinded runtime canary has been run.

## Fixes applied to live source

1. **F2 closed: complete ordered-episode invariant.**
   `tests/real/test_moor_faithful.py` now tests both required directions:
   reversing rows within an episode changes `ordered_trajectory_sse`, and permuting
   intact episodes while remapping episode IDs leaves it unchanged. The latter also
   exercises per-episode latent/capacity reset behavior.

2. **F3 closed: executable tree-hash recipe.**
   `scripts/hash_paper_faithful_snapshot.py` implements and prints the exact recipe used
   by snapshot registration. It recursively sorts files, excludes `__pycache__`, `.pyc`,
   and `.pyo`, and hashes each UTF-8 POSIX relative path followed by the ASCII SHA-256
   digest of the file. The smoke preparation script now imports this implementation and
   records the verifier path in future registrations.

3. **F5 made explicit: bootstrap-MAP variant.**
   The candidate bank is now named `episode_bootstrap_map_v1`; future PLUS fit artifacts,
   candidate-bank artifacts, summaries, and manifests record that construction. The
   implementation version is `plus_faithful_bootstrap_map_cross_form_v1`. The plan now
   states that this is the registered initial variant and does not claim implementation
   of the original local-curvature/low-discrepancy proposal-and-rescore procedure.

4. **Manifest bookkeeping corrected before the next run.**
   The canary manifest previously assigned PLUS's configured candidate count and
   uniform prior to MOOR rows. Future manifests now record one MOOR model with
   `prior=not_applicable`; PLUS alone records 4/16 candidates and a uniform prior.

## Independent verification

The exact frozen v4 registration digest was reproduced without modifying the run:

```bash
.venv-paper-faithful/bin/python scripts/hash_paper_faithful_snapshot.py \
  real_ecology_runs/paper_faithful_smoke_20260717_v4/code \
  --expected a7836e85690316681296dbc5fe095fc195d2d1ee8108ad5a2b5e8ec740af386f
```

Result: `matched: true`, `included_file_count: 118`.

Verification after the live-source fixes:

- targeted faithful MOOR/PLUS/artifact tests: 6 passed;
- complete repository suite: 128 passed;
- Ruff check: passed;
- frozen v4 source-tree digest: unchanged and reproduced exactly.

The initial targeted pytest command without `PYTHONPATH=src` failed during collection
with `ModuleNotFoundError: real_ecology_benchmark`; rerunning with the project source path
passed. This was a shell environment issue, not a test or implementation failure.

## Remaining execution gate

The next faithful execution must be a new frozen 4,000-target, complete-episode,
one-cell smoke with 16 PLUS candidates (four bootstrap-MAP fits per form). It should be
used to establish fit stability and measured runtime before the blinded 32-row canary.
It must not reuse or relabel v4, and it must not be described as a headline experiment.

The PI-approved total CPU ceiling and the full-versus-balanced-core rule remain required
before the blinded canary or any complete sweep. PBVI numerical comparison against an
exact small reference, real-scale parameter recovery, detailed autodiff-gradient checks,
and any external APPL/DESPOT license path remain open audit items exactly as Claude
reported.
