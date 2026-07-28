# Handoff: Repo Polish (commit, test-discovery gap, legibility, git hygiene)

Date: 2026-07-09
Author: Claude (post-implementation audit of the readability refactor)
Scope: `discrete_action_cont_obser/` (+ a little repo-root git hygiene)

> The readability refactor itself is **verified correct** — see "Verified" below.
> Nothing here fixes a bug in it. These are follow-ups, ordered by priority.
> Each task = its own commit.

## Verified (no action needed)

I re-ran and probed the implemented `setpoint_cumulative` refactor:

- `real_setpoint` survives in `src/` **only** as the `LEGACY_REAL_SETPOINT`
  constant; all 30 semantic sites use `is_setpoint_cumulative(...)`.
- The three proxy sites are handled exactly as the audit required:
  `config.py` `load_config` keeps its `→ "real"` data-mode default;
  `pipeline.py` normalizes **both** recorded and expected control modes (it added
  an `expected_value()` helper); `collector.py` renamed `is_real` → `is_setpoint`.
- Functional probes: a legacy `control_mode="real_setpoint"` config normalizes to
  `setpoint_cumulative` in memory; a legacy cache (old label, **no** `data_mode`
  key) still validates against a canonical config; `synthetic` + set-point is
  correctly rejected, with reader-facing wording in the error message.
- `docs/architecture.md` has the axes + characteristic-name tables and the alias
  note. Remaining "tier" strings are only the deliberately-kept code identifiers
  (`tier2_one_step`, `cumulative_capped`).

---

## Task 1 — Commit the work (BLOCKER)

Nothing is committed: **51 modified + 10 untracked** files. The renames currently
appear as `D old` + `?? new` pairs.

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
git add -A          # lets git detect the renames as renames, preserving history
git status          # confirm renames show as "renamed:", not delete+add
```

Prefer the plan's three-commit split (path-scoped `git add`):
1. `Document benchmark axes and characteristic names` (docs only)
2. `Rename setpoint cumulative control mode` (src + configs + tests)
3. `Retire Tier names from live surfaces` (file renames + references)

If splitting the entangled worktree is impractical, one commit is acceptable —
but do **not** leave this uncommitted. Everything below assumes a clean tree.

---

## Task 2 — Fix the silent test-discovery gap (HIGH VALUE, cheap)

**The documented "full suite" never runs the real or dummy tests.**

Evidence:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v | grep -ciE "dummy|real_ecology"
# -> 0        (of 46 collected tests, none are real/dummy)
```
Cause: `tests/real/` and `tests/dummy/` have no `__init__.py`, and on Python 3.9
`unittest` discovery does not recurse into non-package directories. So
`make test` and every historical "46 tests OK" report exercised only the loose
synthetic/shared tests — **not** the 38 real + 9 dummy tests that guard the
paper-facing setting and the whole `data_mode` refactor.

Proven fix (verified in a scratch copy — `Ran 93 tests / OK`):
```bash
touch tests/real/__init__.py tests/dummy/__init__.py
PYTHONPATH=src python -m unittest discover -s tests   # -> Ran 93 tests, OK
```

Also:
- Update the `Makefile` `test` target so one command runs all 93; keep
  `test-real` / `test-dummy` as focused targets.
- Update `README.md` so the "full suite" command is the one that really is full.
- `pyproject.toml` declares `[tool.pytest.ini_options] testpaths = ["tests"]`, but
  **pytest is not installed** in this environment (`No module named pytest`).
  Either add pytest to the dev extra or drop that config block so it does not
  imply a working path that nobody runs.

Acceptance: `PYTHONPATH=src python -m unittest discover -s tests` reports
**93 tests, OK**, and the verbose output contains real and dummy test names.

---

## Task 3 — Rename the misleading synthetic configs

`configs/default.yaml` declares `data_mode: synthetic`, but the CLI's `--config`
default is `configs/real_default.yaml`. **The file named "default" is not the
default.** Same trap for `full.yaml` (the synthetic full-budget config).

Rename to match the existing `real_*` / `dummy_setpoint_*` convention:

```
configs/default.yaml -> configs/synthetic_default.yaml
configs/full.yaml    -> configs/synthetic_full.yaml
```

**Reference-integrity: ~15 sites must change in the same commit.** Sweep first:
```bash
rg -n "configs/(default|full)\.yaml" README.md Makefile scripts/ docs/
```
Known references:
- `scripts/run_gate_matrix.py:20`, `scripts/run_manifest_row.py:26`,
  `scripts/regenerate_failed_cells.py:30` (argparse defaults → `configs/full.yaml`)
- `scripts/calibrate_profiles.py:85`, `scripts/capture_trajectories.py:74`
  (`load_config(ROOT / "configs/full.yaml")`)
- `README.md` lines ~37-41, 52, 56
- `Makefile` lines ~13, 22, 25

Acceptance: the 93-test suite passes; all four smokes still run; `rg` finds no
stale `configs/default.yaml` / `configs/full.yaml` reference.

---

## Task 4 — Make the test tree mirror `data_mode`

Today `tests/real/` and `tests/dummy/` exist, but the synthetic + shared tests sit
loose at `tests/` root. Move them into `tests/synthetic/` so the tree reads as the
three data modes:

```
tests/real/       test_real_ecology.py
tests/dummy/      test_dummy_setpoint.py
tests/synthetic/  test_methods.py test_environment.py test_cumulative_controls.py
                  test_dataset_filter.py test_evaluator_gate.py
                  test_calibration_profiles.py common.py
```

`tests/common.py` is imported **only** by `test_methods.py`,
`test_evaluator_gate.py`, and `test_dataset_filter.py` — all synthetic — so it
moves into `tests/synthetic/` with them.

**Import gotcha:** once `tests/synthetic/` is a package (`__init__.py`, per Task 2),
discovery imports these as `synthetic.test_methods`, and a bare `import common`
will no longer resolve (it would look for a top-level `common`). Change those
three files to a package-relative import:
```python
from .common import ...      # instead of: import common / from common import ...
```

Add `tests/synthetic/__init__.py` alongside the two from Task 2.

Acceptance: `discover -s tests` still collects **93 tests, OK**, and
`tests/{real,dummy,synthetic}` is the only structure under `tests/`.

---

## Task 5 — Declutter `docs/` root so it reads as current truth

`docs/` root holds **22 files**, but only ~9 are current reference material; the
rest are process history (plans, audits, handoffs, dated design notes). A
supervisor opening `docs/` should see the current spec, not the argument that
produced it. You already have the precedent: `docs/real_ecology_history/` and
`docs/claude_build_report/`.

**Keep at `docs/` root (live reference + current specs):**
```
architecture.md
methods.md
experiment_protocol.md
reproducibility.md
22_6_Continuous_Observation_Experiment_Results.tex
22_6_Continuous_Observation_New_Baselines.tex
29_6_Real_Ecology_Setting_Implementation_Plan.tex
29_6_Real_Ecological_Data_Actions_and_Costs.tex
29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md
```

**Move to `docs/history/` (process/provenance):**
```
PLAN_single_real_data_repo_with_dummy_setpoint_2026-07-09.md
PLAN_single_real_data_repo_with_dummy_setpoint_REVISED_2026-07-09.md
PLAN_readability_fixes_axes_and_setpoint_rename_2026-07-09.md
AUDIT_PLAN_single_real_data_repo_2026-07-09.md
AUDIT_readability_plan_2026-07-09.md
HANDOFF_repo_cleanup_discrete_action_cont_obser_2026-07-09.md
HANDOFF_claude_build_removal_and_open_issues_2026-07-09.md
HANDOFF_reduce_duplication_2026-07-09.md
HANDOFF_repo_polish_2026-07-09.md          (this file)
HANDOFF_real_ecology_data_setting.md       (confirm: spec or handoff?)
29_6_Improve_Continuous_Observations.tex
29_6_tier2_continuous_state_implementation_handoff.md
29_6_tier3_12h_experiment_plan.md
29_6_tier3_final_implementation_plan.md
```

Two judgement calls to confirm with the user rather than assume:
`HANDOFF_real_ecology_data_setting.md` (may be a live spec despite the name), and
whether the three `29_6_Real_*` specs are still authoritative or superseded by
`architecture.md`.

Use `git mv` (these are tracked) so history follows. Then add a two-line
`docs/README.md` saying: root = current reference; `history/` = how we got here.

Acceptance: `docs/` root lists only the live set; no link in `README.md` or
`architecture.md` points at a moved file (`rg -n "docs/(PLAN|AUDIT|HANDOFF)" .`).

---

## Task 6 — Git hygiene: stop tracking generated files

Confirmed tracked-despite-`.gitignore`:

| Path | Tracked files | Action |
| --- | ---: | --- |
| `baseline_original/**/__pycache__/` | 39 | all 39 tracked pycache files live here; `git rm -r --cached` |
| repo-root `logs/` | 1,178 | `.gitignore` has `*.err`/`*.out` but **not** `logs/`; add it + `git rm -r --cached logs` |
| repo-root `outputs/` | 12 | `.gitignore` has `outputs/` yet these are tracked; `git rm -r --cached outputs` |
| `real_ecology_runs/` | 173 | **leave tracked** — this is the intentional paper-provenance bundle (code 82, analysis 51, manifests 18, canaries 12, run scripts) |

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
git rm -r --cached logs outputs
git rm -r --cached $(git ls-files | grep __pycache__ | sed 's|/__pycache__/.*|/__pycache__|' | sort -u)
# add to .gitignore:  logs/
git commit -m "Stop tracking generated logs, outputs, and __pycache__"
```

Note: `git rm --cached` untracks without deleting from disk. Verify
`git ls-files | grep -c __pycache__` returns 0 afterwards.

**Related, separate decision:** all tracked `__pycache__` sits under
`baseline_original/`, which is now **orphaned** (only the removed `claude_build`
used it, via `HmMDPAdapter`). Deciding its fate — archive or delete — would remove
those 39 files as a side effect. Ask the user; do not delete unprompted.

---

## Suggested order

1. **Task 1** (commit — protects everything).
2. **Task 2** (discovery gap — it changes what "verified" means; do it before
   relying on any further green run).
3. **Tasks 3 + 4 + 5** as one "repo legibility" commit, or three small ones (all
   are renames/moves with reference updates; verify with the 93-test suite and the
   four smokes after each).
4. **Task 6** as its own git-hygiene commit.

Deferred / not in scope here: renaming the `real_ecology_benchmark` package (it now
hosts real + dummy + synthetic — cosmetic, touches every import; do it just before
publication), collapsing the duplicate `real-ecology` / `ecology-benchmark` console
scripts, deciding the fate of the orphaned `baseline_original/`, and the repo-root
"one folder" consolidation (`outputs/`, `logs/`, `docs/`, `slurm-*.out`).
