# Corrected Stage B sigma=0.2 local infrastructure

**LOCAL ARM64 DEVELOPMENT TEST — NOT SCIENTIFIC EVIDENCE**

This is the trusted external implementation constructed from Revision 3.1, frozen I1 and
zero-SD contracts, audited I2A adapters, audited fast-track wrappers, frozen method source,
and the independent Stage B audit. It does not import or execute either failed corrected
attempt.

The package implements:

- one registration-bound external `driver.py` with the exact `arm-o`, `arm-o-gate`, `arm-t`,
  and `finalize-inspection-only` interfaces consumed by the existing Slurm templates;
- issued, non-serializable registration and Arm-T process capabilities plus an authenticated
  cross-process gate receipt that reissues the process capability only after full validation;
- strict component-specific canonical schemas and lossless external serializers for complete
  real frozen fitted-object graphs, with gate-side reload and prediction/action replay;
- source-backed component applicability and V3 component-keyed deterministic prediction
  fixtures, with exact native feature widths and a separate fixture-manifest hash;
- exactly one cell/source-bound Arm-O surrogate shared by RefPlan, OGSRL and BA-MCTS;
- complete Ricker/PBVI byte identity and demonstrated ecological O/T interfaces;
- registration-bound, filesystem-dereferenced task evidence, including strict fitted-component
  reload plus schema-dispatched residual, complete posterior/dispersion/action, boundary,
  paired RNG and return receipts;
- recursive public allowlists and closed context-preserving exact-state wrappers;
- unique task-local full-tree validation and last-operation atomic publication;
- configurable M3 Slurm dependency templates.

The driver has no fit entry point. It loads the sealed V4 fitted-object publications, keeps
Arm O on the public-observation route, enables the context-preserving exact-current-state
overlay only after signed-gate verification, and atomically publishes complete task evidence.
Its bytes are a mandatory production-registration identity under
`corrected_stageb_code_configuration_hashes_v2`.

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

The authoritative applicability map and V2-to-V3 receipt migration are documented in
`ARTIFACT_CONTRACT_V3.md`. Distinct fitted `preprocessing`, `feature_transformations`, and
PBVI alpha-vector objects do not exist in the inspected method graphs; MOOR also has no
candidate prior. The contract excludes these absent components instead of inventing stand-ins.

The bounded producer/independent fitted-object replay evidence is specified in
`REPLAY_DIAGNOSTIC_CONTRACT.md`. Its separate atomically published artifact localizes exact
post-call graph differences while preserving the original strict parity requirement.
