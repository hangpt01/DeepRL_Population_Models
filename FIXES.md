# Independent-review fixes

Date: 2026-07-28 (Australia/Melbourne)

This file maps the findings in `REVIEW.md` to the follow-up changes. All changes
are in the repository layer. The 107 files under `src/tracks/**` were not edited.

| Finding | Status | Resolution and evidence |
|---|---|---|
| B1 | DONE | Added the PyTorch CPU extra index as line 1 of `requirements.txt`; aligned `README.md` and `REPRODUCE.md`. A new Python 3.10.14 venv installed the complete requirements successfully and imported `torch==2.13.0+cpu`. |
| B2 | DONE | The diagnostic replay now exits non-zero unless parity is `PASS`, cache reuse is true, and measured recomputed fits are zero. `recomputed_fits` is derived from `policy.fit_diagnostics`. The wrapper propagates the child status. `make verify` includes a deliberately corrupted comparison that must exit non-zero. |
| S1 | DONE | Added shared `DEEPRL_*` repository/scratch path resolution and repointed all 15 diagnostic entry points to canonical frozen tracks. `finalize_followups.py` reads `results/accepted/`. A repository-only S2 planning run passes. `CLEANUP_MANIFEST.md` now states the actual parameterization boundary. |
| S2 | DONE | Added the hash-recorded 576-row general manifest and config under `experiments/accepted_general/`, plus `make verify-cell-general`. Tier-B B1/EVD reproduces with `max_abs_diff=0.0`. |
| S3 | DONE | Added test-package shims and corrected the general privacy fixture import. `make test-general` passes 155 tests. |
| S4 | DONE | Installation instructions now install `.[analysis]` after the exact requirements, providing pandas and matplotlib before `make verify`. |
| S5 | DONE | Removed stale flat script/test duplicates. Canonical code is under `scripts/ecological`, `scripts/general`, `tests/ecological`, and `tests/general`. Remaining harness imports and snapshot preparation point explicitly to those track directories; both corrected-canary pytest files pass. |
| S6 | DONE | Added `provenance/scientific_inputs.sha256` for canonical track scripts, ecology CSV inputs, and experiments. `scripts/verify_integrity.py` enforces exact coverage and hashes in `make verify`. |
| S7 | DONE | Documented `module load miniforge3/24.3.0-0` and `/apps/miniforge3/24.3.0-0/miniforge3/bin/python` for Python 3.10.14. |
| N1 | DONE | Documented the `src/tracks/real_ecology_data` compatibility symlink and added diagnostic startup existence checks. |
| N2 | DONE | Added `DEEPRL_P10_PACKAGE` and related standalone path keys to `configs/paths.example.yaml`. |
| N3 | NOT-STARTED | Optional review suggestion; not required for this follow-up. |
| N4 | DONE | Added `.review/` to `.gitignore`. |
| N5 | DONE | Documented byte-identical flat YAML compatibility copies in `configs/README.md`; canonical ownership remains track-specific. |
| N6 | NOT-STARTED | Optional review suggestion; not required for this follow-up. |
| N7 | NOT-STARTED | Optional review suggestion; not required for this follow-up. |
| N8 | DONE | Corrected documentation: the authored `archive/` material is tracked; only bulky local archive artifacts are ignored. |
| N9 | DONE | `top_up_followups.py` now handles only the pending H12 files, skips identical recorded artifacts, rejects conflicts, and appends verified source/destination hashes to provenance. |
| F1 | DOCUMENTED | Added a clear comment at the analysis-constant definition describing the inert vulture `r_lgm` discrepancy. Values remain unchanged; correction and regression coverage are deferred. |

The MIT text remains provisional in `LICENSE_STATUS.md`.
