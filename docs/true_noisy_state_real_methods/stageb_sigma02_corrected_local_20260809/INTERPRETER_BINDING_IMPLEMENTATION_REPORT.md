# Interpreter-binding registration correction

Date: 2026-08-11 (Australia/Melbourne)

## Scope and controlling state

This bounded tracked correction starts from
`ddf5fc11ddf4fde781c833f549097176eea80dfa`. It does not change either frozen method source
track, artifact serialization, fitted-object replay semantics, accepted artifacts or any
earlier evidence namespace. The prior 36/36 fit-only evidence remains integration evidence;
the deficient frozen registration and its keys must remain unchanged and must not be reused.

## Registration contract

`corrected_stageb_registration_v2` requires `stageb_interpreter_bindings`. The section has
exactly two role records:

- `ecological_paper_faithful`: ecological PLUS/MOOR, logical tasks 0, 1, 6 and 7, configured
  `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python`,
  resolved `/apps/miniforge3/24.3.0-0/miniforge3/bin/python3.10`, Python 3.10.14 and
  NumPy 2.2.6;
- `general_registered`: general RefPlan/OGSRL/BA-MCTS/EVD, logical tasks 2, 3, 4, 5, 8, 9,
  10 and 11, configured `/usr/bin/python`, resolved `/usr/bin/python3.9`, Python 3.9.25 and
  NumPy 1.23.5.

The full Python strings are frozen verbatim. Both `arm-o-gate` and
`finalize-inspection-only` explicitly bind to `general_registered`, matching the successful
pre-execution environment. Validation requires exact role ordering and identity, complete
non-overlapping method/task coverage, task-manifest agreement, absolute existing paths and
the registered symlink resolution.

## Runtime and launch enforcement

All four driver subcommands validate the applicable task or command binding immediately
after registration load and task selection, before driver inputs, artifacts, evaluators or
publication. Every task, parity, gate and finalizer terminal receipt/publication records the
complete expected and observed identity. Gate and finalizer also authenticate upstream task
identity receipts.

The four scientific Slurm roles no longer accept interpreter environment overrides. They
freeze the bundle, derive the task/command role, select its registered absolute interpreter
and invoke the driver, which repeats the full check. The prepared pre-execution package must
record the same binding and resolved launch map in `SCIENTIFIC_CHAIN_ENVIRONMENT.json`.

## Tests and stop boundary

Twenty-five focused regressions cover the V1/missing-section rejection, exact coverage,
task-role mismatch, five independent identity-field failures, both correct roles, all four
subcommand ordering gates, task/command resolution, environment non-override, receipt
completeness and Slurm derivation. The corrected suite now collects 227 tests. Final local
and M3 gate results are recorded only in their checksum-covered evidence packages.

The final development gate passed:

- corrected suite: 227/227;
- audited I2A: 38/38 with unittest and 38/38 with pytest;
- audited fast-track: 15/15 with unittest and 15/15 with pytest;
- Ruff lint/format: 25 Python files clean;
- strict JSON: 11/11 files parsed;
- shell syntax: 5/5 files passed;
- source/test manifest: complete standard coverage plus normalized self-hash;
- frozen tracks: ecological 52-file hash
  `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` and general
  55-file hash `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd`.

No scientific Arm O/T job, evaluator, truth/private access, runtime `next_states` access or
return calculation is authorized by this implementation report.
