# Handoff: Reduce Duplication After `claude_build` Removal

Date: 2026-07-09
Author: Claude (audit + dedup plan for codex)
Scope: `discrete_action_cont_obser/`
Repo root: `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`

> Status: instructions only. Do each task as its own commit, after the working
> tree is clean/committed. Get user approval before deleting `tier2_benchmark`
> (Task 2). Task 1 involves migrating real content first — do not blind-`rm`.

## Context: what is already done

- `claude_build/` removed cleanly (commits `320fa63`, `f385b0f`); its docs
  preserved under `docs/claude_build_report/`. Verified: 0 tracked, gone from
  disk, `.gitignore` trimmed.
- The `data_mode: real|dummy` refactor is committed on `main`.

## The remaining duplication: three copies of one package

| Copy | Git | Role | Verdict |
|---|---|---|---|
| `src/real_ecology_benchmark/` | tracked | **canonical** superset (real + dummy + retains synthetic `tier2_one_step`/`cumulative_capped`) | keep |
| `src/tier2_benchmark/` | tracked (28 files) | original synthetic fork parent | retire (Task 2) |
| `real_ecology_cont_obser/` | **untracked**, ~994M | stale pre-promotion copy + own artifacts + un-migrated scripts/docs | retire (Task 1) |

`src/real_ecology_benchmark/` and `src/tier2_benchmark/` share 20 module names
(5 byte-identical: `dataset.py`, `dynamics.py`, `__main__.py`, `observation.py`,
`rollout.py`; ~15 near-identical). This is the bulk of the remaining duplication.

---

## Task 1 — Retire the stale `real_ecology_cont_obser/` duplicate

Biggest, lowest-risk win, but **not** a blind delete: `src/`, `configs/`, and
`tests/` were promoted during the refactor, but `scripts/` and `docs/` were
**not**. Reconcile those first.

### 1a. Migrate or retire the real-specific scripts

These exist only in the old folder (top-level `scripts/` has the tier2/tier3
synthetic scripts, not these):

```
real_ecology_cont_obser/scripts/make_real_experiment_manifests.py
real_ecology_cont_obser/scripts/run_real_gate_row.py
real_ecology_cont_obser/scripts/run_real_manifest_row.py
real_ecology_cont_obser/scripts/shard_manifest.py
real_ecology_cont_obser/scripts/summarize_real_outputs.py
real_ecology_cont_obser/scripts/slurm/{run_real_aggregate.sh, run_real_gate_row.sh,
                                        run_real_row_gpu.sh, run_real_row.sh}
```

They drove the P-safe overnight run, so treat as **keep unless confirmed dead**.

- Decide with the user: still needed for future real runs, or one-off/obsolete?
- If kept: move to top-level `scripts/` (and `scripts/slurm/`). **They were written
  to run from inside `real_ecology_cont_obser/`** — before trusting them, check and
  fix any hardcoded paths / `PYTHONPATH` / `cd` assumptions:
  ```bash
  rg -n "real_ecology_cont_obser|PYTHONPATH|sys.path|cd " real_ecology_cont_obser/scripts
  ```
  Update those to the promoted `src/` layout, and smoke-run at least
  `run_real_manifest_row.py` to confirm it still works.

### 1b. Preserve the unique design/audit docs

`real_ecology_cont_obser/docs/` has **26 files** of real-ecology design, audit,
and handoff history (the `29_6_*`, `AUDIT_*`, `CODEX_*`, `HANDOFF_*`,
`CODE_REVIEW_*` notes). Do not delete these with the folder.

- Diff against `discrete_action_cont_obser/docs/` to find which are already there:
  ```bash
  for f in real_ecology_cont_obser/docs/*; do
    b=$(basename "$f"); [ -e "docs/$b" ] && echo "DUP: $b" || echo "UNIQUE: $b"
  done
  ```
- Move the `UNIQUE` ones into `docs/` (or `docs/real_ecology_history/`). These are
  provenance for the real-ecology results; keep them tracked.

### 1c. Confirm configs/tests are truly superseded

Same filenames exist top-level; confirm the promoted copies are canonical (or
newer) before dropping the old ones:

```bash
for f in real_default real_experiment real_probe real_smoke; do
  diff -u real_ecology_cont_obser/configs/$f.yaml configs/$f.yaml && echo "SAME: $f" \
    || echo "DIFF: $f  (review before dropping the old one)"
done
diff -u real_ecology_cont_obser/tests/test_real_ecology.py tests/real/test_real_ecology.py \
  && echo "tests SAME" || echo "tests DIFFER (tests/real is canonical; confirm)"
```

If any `DIFF`, reconcile intentionally — the promoted copy should win, but verify
no fix lives only in the old one.

### 1d. Archive artifacts only if provenance matters

`real_ecology_cont_obser/{outputs, log}` are generated. The authoritative P-safe
provenance bundle already lives at `real_ecology_runs/psafe_overnight_20260705/`,
so these are likely disposable — but confirm they are not the only copy of
anything paper-facing before removing.

### 1e. Delete and commit

The whole folder is untracked (`git ls-files real_ecology_cont_obser` -> 0), so
this is a plain filesystem delete after 1a–1d:

```bash
# after scripts migrated, unique docs moved, configs/tests confirmed superseded:
rg -n "real_ecology_cont_obser" discrete_action_cont_obser/{src,tests,configs,scripts} \
   Makefile README.md    # must return nothing
rm -rf real_ecology_cont_obser
git add -A
git commit -m "Remove superseded real_ecology_cont_obser duplicate; migrate real scripts/docs"
```

### Task 1 — Do NOT

- Do not `rm -rf real_ecology_cont_obser` before 1a/1b — you would lose the
  real-specific scripts and 26 design/audit docs (they are untracked, so **not
  recoverable from git**).
- Do not assume configs/tests are identical — diff them.

---

## Task 2 — Retire `src/tier2_benchmark/` and unify on one package

`src/real_ecology_benchmark/` is a **superset**: it keeps
`CONTROL_MODES = {tier2_one_step, cumulative_capped, real_setpoint}`, so synthetic
runs can execute under it. That makes `tier2_benchmark` a duplicate to retire —
but this is deliberate and needs verification, not a blind delete.

### Blockers to clear first

- **7 legacy tests import `tier2_benchmark`:** `tests/common.py`,
  `tests/test_methods.py`, `tests/test_environment.py`, `tests/test_tier3_controls.py`,
  `tests/test_dataset_filter.py`, `tests/test_evaluator_gate.py`,
  `tests/test_calibration_profiles.py`.
- **3 configs are tier2:** `configs/default.yaml`, `configs/full.yaml`,
  `configs/tier3_12h.yaml`.
- **`pyproject.toml`** exposes the `tier2 = "tier2_benchmark.cli:main"` console
  script and `packages.find` discovers both packages.

### Steps

1. Repoint the 7 tests' imports `tier2_benchmark` -> `real_ecology_benchmark`
   (put them under `tests/legacy_synthetic/` if you want them separated from
   `tests/real` and `tests/dummy`).
2. Repoint the 3 configs if they name a package/module; keep them as the
   synthetic reference cells.
3. **Run them under `real_ecology_benchmark` — this is the compatibility test:**
   ```bash
   PYTHONPATH=src python -m unittest discover -s tests -v
   PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/default.yaml
   ```
4. Only if green: `git rm -r src/tier2_benchmark` (28 tracked files), drop the
   `tier2` console script from `pyproject.toml`, update `Makefile`/`README.md`.
5. Verify no residue: `rg -n "tier2_benchmark" src tests configs scripts Makefile pyproject.toml`.

### Reproducibility caveat (read before deleting)

- The fork **diverged**: ~15 of the 20 shared modules differ between the two
  packages. Running the legacy synthetic tests under `real_ecology_benchmark` may
  surface behavioral drift. If a test fails, either reconcile the diff or accept
  that the exact old Tier-2/Tier-3 semantics are legacy.
- **Dummy mode is not a semantic replacement** for tier2: dummy is
  set-`r`/cumulative-`K`, whereas `tier2_one_step` sets both and
  `cumulative_capped` accumulates both. If you may need to reproduce the old
  additive/both-cumulative synthetic runs exactly, do **not** rely on the
  diverged fork — keep the git tag/commit `f36764c` as the reproducible archive
  and delete `tier2_benchmark` from the live tree only.

---

## Task 3 — Housekeeping (optional, low priority, do last)

- **Package name:** after unification, `real_ecology_benchmark` also hosting
  dummy + synthetic is a mild misnomer. A later *pure* rename (e.g.
  `cont_obser_benchmark`) is cosmetic and churny — only if desired, and never
  bundled with Task 1/2.
- **The 5 byte-identical modules** become moot once `tier2_benchmark` is gone; do
  not invest in factoring a shared lib while a package is still slated for
  deletion.
- **Generated artifacts** (`outputs/`, `logs/`, and the old copy's `outputs/`) are
  disk bloat, not code duplication — a separate cleanup pass, git-tracked-file
  check first (some may be tracked despite `.gitignore`).

---

## Sequencing and approval

1. **Task 1** first — independent, safe once scripts/docs are migrated; biggest
   footprint reduction.
2. **Task 2** next — needs the test/config repoint + a green run to justify it;
   get user sign-off given the reproducibility caveat.
3. **Task 3** last, only if wanted.

Each task = its own commit. Confirm the working tree is committed before starting,
and (as with `claude_build`) never delete untracked content that has not been
migrated or archived.
