# Review-fix verification report

Date: 2026-07-28 (Australia/Melbourne)

Verdict: **PASS**. The independent-review blockers and standalone-publication
findings are fixed at the repository layer. The frozen scientific tracks remain
byte-identical to the accepted-result code.

## Clean installation

A fresh environment was made with Python 3.10.14:

```text
/apps/miniforge3/24.3.0-0/miniforge3/bin/python -m venv \
  /tmp/deeprl-clean-install-B1
/tmp/deeprl-clean-install-B1/bin/pip install -r requirements.txt
/tmp/deeprl-clean-install-B1/bin/pip install -e '.[analysis]'
```

Result: **PASS**. The requirements installation completed successfully and
`torch.__version__` is `2.13.0+cpu`. The analysis extra supplied pandas and
matplotlib for the verification entry points.

## Full verification

Command:

```text
MPLCONFIGDIR=/tmp/deeprl-mpl-review make verify \
  PYTHON=/tmp/deeprl-clean-install-B1/bin/python \
  SCRATCH_PROJECT=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models
```

Result: **PASS**.

- Diagnostic self-tests: all 36 passed (exceeding the requested 32).
- Constants self-check: passed.
- Integrity: 107 frozen-track files, 116 repository scientific inputs,
  18 copied result artifacts, and the accepted CSV passed.
- Accepted A6-MOOR replay: passed at `max_abs_diff=0.0`.

The separate publication targets also passed:

- `make verify-standalone-s2`: repository-only S2 planning smoke passed.
- `make verify-cell-general`: general B1/EVD replay passed at
  `max_abs_diff=0.0`.

### Enforced negative gate

Direct proof through the public wrapper:

```text
set +e
/tmp/deeprl-clean-install-B1/bin/python \
  scripts/verify_accepted_cell.py --gate-self-test-corrupt
status=$?
printf 'NEGATIVE_GATE_EXIT=%s\n' "$status"
```

Output:

```text
verdict=INVESTIGATE
max_abs_diff=9.999999999177334e-07
NEGATIVE_GATE_EXIT=1
```

Thus the wrapper propagates a non-zero child status. `make verify` runs the same
corrupted-comparison check and fails if the child unexpectedly returns zero.

### Accepted ecological cell

Cell A6, Egyptian vulture, Ricker, sigma 0.1, accepted MOOR:

```text
parity=PASS
max_abs_diff=0.0
recomputed_fits=0
reuse_ok=true
elapsed_s=1541.1
```

`recomputed_fits` is no longer a literal. It is measured from
`policy.fit_diagnostics` (`fit_cache_hit` for MOOR and `fit_cache_misses` for
PLUS), and the gate requires it to equal zero. All seven accepted comparison
fields had absolute difference `0.0`.

### General cell

Command:

```text
make verify-cell-general PYTHON=/tmp/deeprl-clean-install-B1/bin/python
```

Result for B1, Crab-eating fox, regime, sigma 0.2, EVD:

```text
parity=PASS
max_abs_diff=0.0
dataset_hash_matches_accepted=true
elapsed_seconds=0.454
```

All seven comparison fields had absolute difference `0.0`; the scientific
implementation loaded from `src/tracks/general`.

## Test harnesses

- `make test-ecological`: **123 passed**.
- `make test-general`: **155 passed**.
- Corrected-canary pytest harness: **3 passed per track** after explicit
  canonical-path correction.

## Frozen-tree and import integrity

Both `git diff --stat -- src/tracks` and `git diff --name-only -- src/tracks`
produced no output. `scripts/verify_integrity.py` verified all 107 entries in
`provenance/frozen_tracks.sha256`; the ledger itself retained SHA-256
`319cd42884da74cbdd54b228ecf4fbb11688e40293d376a333bf1b3be6ffcaaf`.

A grep for removed flat test namespaces and flat-script search-path imports
returned no matches. Same-directory imports such as `from
run_real_manifest_row import ...` occur only inside the canonical
`scripts/ecological/` and `scripts/general/` trees.

## Provenance and scope

`provenance/scientific_inputs.sha256` is enforced with exact discovered coverage
for canonical track scripts, `configs/ecology/*.csv`, and `experiments/`.
Copied small results retain canonical scratch source paths and SHA-256 values in
`provenance/copied_artifacts.sha256`.

No file under `src/tracks/**` was modified. The known inert F1 constant issue is
commented in the repository-layer analysis constants only. The MIT license
remains explicitly provisional pending owner confirmation.
