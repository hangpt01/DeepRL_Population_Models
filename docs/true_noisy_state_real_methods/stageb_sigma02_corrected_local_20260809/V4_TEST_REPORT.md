# Artifact Contract V4 test report

Date: 2026-08-10 (Australia/Melbourne)

All validation used `LC_ALL=C` and `PYTHONDONTWRITEBYTECODE=1`. Python test caching was
disabled. No scientific data, evaluator, fit, return, driver, registration, key, or Slurm
path was opened.

| Check | Final result |
|---|---:|
| Corrected Stage B package | 165/165 passed |
| Audited I2A suite | 38/38 pytest and 38/38 unittest passed |
| Audited fast-track suite | 15/15 pytest and 15/15 unittest passed |
| Ruff lint and format | clean; 21 Python files formatted |
| Strict JSON parsing | 10/10 files parsed |
| Shell syntax | 5/5 files passed `bash -n` |
| Candidate manifest | complete coverage and normalized self-check passed |
| Frozen ecological source | 52 files; `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e` |
| Frozen general source | 55 files; `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd` |

The V4-specific suite includes exact near-zero float64 preservation, mortality-dominant and
mutual-exclusion cases, legacy-state rejection, channel and candidate identity constraints,
one-ULP/cache-hash mismatches, regime row sums, method-scoped 0.019 behavior, exact ecological
diagnostics, V3/V4 mutual rejection, unchanged general floors and information boundaries,
and synthetic plus actual frozen source-class projections across all twelve task paths.

An initial inherited-suite command used `/usr/bin/python` for pytest where pytest is not
installed, and an initial fast-track command omitted its required `I2B_TEST_TMPDIR`.
Those invocation errors did not change repository content and are not counted as test
results. The registered cache-free commands were then run with the required interpreter and
temporary-directory variables and produced the passing results above.
