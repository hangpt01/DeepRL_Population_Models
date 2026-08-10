# Corrected Stage B sigma=0.2 local infrastructure

**LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE**

This is the trusted external implementation constructed from Revision 3.1, frozen I1 and
zero-SD contracts, audited I2A adapters, audited fast-track wrappers, frozen method source,
and the independent Stage B audit. It does not import or execute either failed corrected
attempt.

The package implements:

- issued, non-serializable registration and Arm-T process capabilities plus an authenticated
  cross-process gate receipt that reissues the process capability only after full validation;
- strict component-specific canonical schemas and lossless external serializers for complete
  real frozen fitted-object graphs, with gate-side reload and prediction/action replay;
- exactly one cell/source-bound Arm-O surrogate shared by RefPlan, OGSRL and BA-MCTS;
- complete Ricker/PBVI byte identity and demonstrated ecological O/T interfaces;
- registration-bound, filesystem-dereferenced task evidence, including strict fitted-component
  reload plus schema-dispatched residual, complete posterior/dispersion/action, boundary,
  paired RNG and return receipts;
- recursive public allowlists and closed context-preserving exact-state wrappers;
- unique task-local full-tree validation and last-operation atomic publication;
- configurable M3 Slurm dependency templates.

Registration JSON files are deliberately incomplete templates. They cannot authorize a
scientific return. M3 deployment must fill and freeze real commit, configuration, dataset,
artifact and task identities and then pass the Arm O parity/artifact gate.

Synthetic registration fixtures derive and commit-verify the current repository `HEAD` at
test runtime. They contain no static commit SHA and fail closed on unavailable, unsuccessful,
malformed, multi-line or non-commit Git results. The production registration validator remains
the independent equality gate between frozen registration bytes and checked-out `HEAD`.

Run the local corrected tests with:

```sh
LC_ALL=C PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  .venv-stageb-local/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/private/tmp/stageb-corrected-tests \
  docs/true_noisy_state_real_methods/stageb_sigma02_corrected_local_20260809/tests
```

No real dataset, truth archive, accepted output, scientific return, MPS/GPU path or Slurm
endpoint is used by this package or its tests.
