# Handoff: Promote the Benchmark to Repo Root (stop being a standalone subrepo)

Date: 2026-07-09
Author: Claude (audit + corrected plan for codex)
Repo root: `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`

Supersedes the ad-hoc to-do list in codex's standalone audit. That list was
factually correct — every "standalone fingerprint" it named is real, and root has
no conflicting `src/`, `configs/`, `tests/`, `scripts/`, `pyproject.toml`, or
`Makefile` — but it under-specified five things and missed two. Corrections are
called out inline and summarised at the end.

> Each task = its own commit. Get user sign-off on the dispositions in Task 2
> before moving anything.

---

## Verified starting state

- `git log` head: `066b7c3 Stop tracking Slurm job logs`. Working tree has exactly
  one uncommitted change: `M .gitignore`.
- Baseline that must still hold after the move:
  `PYTHONPATH=src python -m unittest discover -s tests` → **93 tests, OK**, and the
  four smokes (`synthetic_default`, `real_smoke`, `dummy_setpoint_smoke`,
  `cumulative_controls_12h`) all pass.
- `discrete_action_cont_obser/logs/` and `outputs/` are **untracked** already
  (1,996 log files on disk, correctly ignored). `real_ecology_runs/` is **173
  tracked** provenance files.

## The one invariant that must not break

`realdata.py` computes:
```python
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent   # -> repo root after the move
DATA_DIR     = PACKAGE_ROOT / "real_ecology_data"
```
**`src/` and `real_ecology_data/` must remain siblings.** If `real_ecology_data/`
lands anywhere else, `DATA_DIR` silently resolves to a non-existent path — the
exact off-by-one class of bug that bit the last promotion. A test already asserts
`DATA_DIR / "species.csv"` exists; keep it green.

**Correction to codex's list, item 5:** "update path assumptions" is a *verify*
step, not a rewrite. Nothing needs editing:
- `scripts/*.py` use `ROOT = Path(__file__).resolve().parents[1]` → becomes repo
  root automatically once `scripts/` sits at root.
- `Makefile`'s `PYTHONPATH=src` works identically from root.
- `realdata.py`'s **logic** is already correct; only its stale **comment** (line 25,
  naming `discrete_action_cont_obser/real_ecology_data`) needs updating.

---

## Task 0 — Pre-flight

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
git add .gitignore && git commit -m "Update gitignore"   # clear the stray change
git status --short                                        # must be empty
cd discrete_action_cont_obser && PYTHONPATH=src python -m unittest discover -s tests   # 93 OK
```
Do not start with a dirty tree.

---

## Task 1 — Merge `.gitignore` before deleting the nested one

**Correction to codex's list, item 4:** do *not* just delete
`discrete_action_cont_obser/.gitignore`. It contains five entries the root
`.gitignore` lacks. Dropping them means generated `.npz` datasets and build
artifacts stop being ignored and start showing up as committable.

Add to the root `.gitignore` first:
```
data/
*.npz
build/
dist/
*.egg-info/
```
(`__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `.venv/`,
`outputs/`, `checkpoints/` are already covered at root.)

Then the nested `.gitignore` can be deleted in Task 3.

---

## Task 2 — Decide the dispositions (needs user sign-off)

| Path | Tracked | Disposition |
| --- | ---: | --- |
| `src/`, `configs/`, `tests/`, `scripts/` | — | `git mv` to repo root |
| `real_ecology_data/` | — | `git mv` to repo root — **must stay sibling of `src/`** |
| `pyproject.toml`, `Makefile`, `requirements.txt` | — | `git mv` to repo root (root has none) |
| **`LICENSE`** | — | **codex's list missed this.** Root has no LICENSE → `git mv` to root |
| `real_ecology_runs/` | 173 | `git mv` to repo root, unchanged (paper provenance) |
| `README.md` (nested) | — | its content becomes the new root README (Task 5) |
| `.gitignore` (nested) | — | delete *after* Task 1 merge |
| `logs/`, `outputs/` (nested) | 0 | generated + untracked → `rm -rf`, do not move |
| `docs/` (nested, 55 tracked) | 55 | merge into root `docs/` — see below |

Two decisions to confirm with the user:

- **`baseline_original/`** (102 tracked) — the root README describes *this*
  (the published "universal 2-state n-action solver"). It became **orphaned** when
  `claude_build` was removed (only `HmMDPAdapter` used it). Keep as
  attribution/history, or archive? The root-README rewrite (Task 5) depends on
  this answer.
- **Root `docs/` clutter** — `phase116_*`, `stress_pomdp_*`, `codex_*`,
  `system_prompt.md`, `REPO_BRIEF.md`, `next_chat_*` are `claude_build`-era process
  notes. Propose moving them into `docs/history/` alongside the benchmark's own
  history. Confirm before moving.

### docs/ merge — there is a real collision

**Correction to codex's list, item 2:** it says "merge deliberately" without
flagging that `22_6_Continuous_Observation_New_Baselines.tex` exists in **both**
trees **and they differ**. I diffed them:

- root copy: still says `Tier-2 report: ...` / `(Tier-2: from discrete abundance bins ...)`
- nested copy: says `continuous-state synthetic benchmark report: ...`

The **nested copy wins** — it carries the post-rename wording from the
`Retire Tier names` commit. Overwrite the root copy with it.

Proposed merged layout:
```
docs/                      <- current reference (nested live docs land here)
docs/README.md             <- nested docs/README.md (root docs has none)
docs/history/              <- nested docs/history/ (14 files) + root process notes
docs/real_ecology_history/ <- moved as-is
docs/claude_build_report/  <- already at root, stays
docs/plots_selected/       <- moved as-is
```

---

## Task 3 — Pure `git mv` promotion (ONE commit, no content edits)

**Correction to codex's list:** it never says to isolate the move. Do the rename
in its own commit with **zero content changes**, so git detects renames and
`git log --follow` keeps working.

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
D=discrete_action_cont_obser
git mv $D/src $D/configs $D/tests $D/scripts $D/real_ecology_data \
       $D/real_ecology_runs $D/pyproject.toml $D/Makefile \
       $D/requirements.txt $D/LICENSE .
# docs: merge per Task 2 (nested .tex overwrites root copy)
git mv $D/docs/* docs/            # resolve the one collision explicitly
git rm $D/.gitignore
rm -rf $D/logs $D/outputs         # untracked generated
rmdir $D 2>/dev/null || ls -A $D  # must be empty; investigate anything left

git status   # renames must show as "renamed:", not delete+add
git commit -m "Promote ecology benchmark to repo root"
```

Sanity check *before* committing: `ls src real_ecology_data` — they must be
siblings at root (the invariant above).

---

## Task 4 — Update the standalone wording (second commit)

Only three live files claim standalone status:
- `README.md` (nested → root, line 3): "This is a **standalone implementation** …
  It has no runtime dependency on the parent repository or `claude_build`."
- `src/real_ecology_benchmark/cli.py:1`: `"""Command-line interface for the
  standalone benchmark."""`
- `src/real_ecology_benchmark/realdata.py:25` (comment): names
  `discrete_action_cont_obser/real_ecology_data`.

**Correction to codex's list, item 6:** don't simply delete the standalone claim.
"No runtime dependency on the parent repo" is a genuine engineering property — it
is what lets this be extracted later as a paper artifact. Promotion does not
destroy it (the package stays `src/real_ecology_benchmark/` with its sibling
`real_ecology_data/`). Reword rather than remove:

> "The benchmark is a **self-contained package**: `src/real_ecology_benchmark/`
> plus its data tables in `real_ecology_data/`. It has no dependency on any other
> directory in this repository."

Also drop any `cd discrete_action_cont_obser` from README/Makefile instructions —
commands now run from the repo root unchanged (`PYTHONPATH=src ...`).

---

## Task 5 — Rewrite the root README (third commit)

The root `README.md` currently describes the paper *"A universal 2-state n-action
adaptive management solver"* — i.e. `baseline_original/`, not the current work.

- Make the ecology benchmark the first thing a reader sees (start from the nested
  README's content, reworded per Task 4).
- Relocate the old 2-state-solver README text into `baseline_original/README.md`
  (or `docs/history/`), depending on the Task 2 decision.

This is the highest-visibility change for supervisors — the root README should
answer "what is this repo?" with the current benchmark.

---

## Task 6 — Sweep the leftovers

```bash
rm -f slurm-55471800.out          # stray root file
rm -rf .pytest_cache             # generated
```
(`.gitignore` already covers `*.out` and `.pytest_cache/`; these predate it.)

---

## Task 7 — Verify (from the repo root now)

```bash
PYTHONPATH=src python -m unittest discover -s tests -v      # must be 93 OK
for c in synthetic_default real_smoke dummy_setpoint_smoke cumulative_controls_12h; do
  PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/$c.yaml
done
bash -n scripts/slurm/*.sh
python -c "from pathlib import Path; import sys; sys.path.insert(0,'src'); \
  from real_ecology_benchmark import realdata; \
  assert (realdata.DATA_DIR/'species.csv').exists(), realdata.DATA_DIR; print('DATA_DIR ok', realdata.DATA_DIR)"
```

Then two sweeps — **enumerate by pattern, not by the paths you happen to remember**
(this is the lesson from the last cleanup, where a path list missed 1,332 tracked
log files):
```bash
# 1. no live standalone/subrepo wording or paths
rg -n "standalone|cd discrete_action_cont_obser|discrete_action_cont_obser" \
   README.md Makefile pyproject.toml src/ scripts/ configs/ tests/ docs/*.md

# 2. no tracked generated artifacts
git ls-files | grep -E '\.(err|out|npz|pt|log)$|__pycache__|\.pytest_cache'
```

Expected residual hits for sweep 1 (leave them alone):
- `real_ecology_runs/psafe_overnight_20260705/code/**` — frozen provenance snapshot,
  duplicated old code **by design**. Do not refactor it.
- `docs/history/**`, `docs/real_ecology_history/**` — historical text.

---

## Summary of corrections to codex's original list

1. **Path assumptions:** verify only — `ROOT`/`PACKAGE_ROOT` are relative and
   survive. Only `realdata.py`'s comment is stale. (Its item 5 overstated this.)
2. **`.gitignore`:** merge 5 entries into root *before* deleting the nested file.
   (Its item 4 said "delete".)
3. **docs merge:** there is a real collision, and the copies **differ** — the nested
   `22_6_Continuous_Observation_New_Baselines.tex` wins. (Its item 2 hand-waved.)
4. **Missing dispositions:** `LICENSE` (nested; root has none) and
   `real_ecology_runs/` (173 tracked) were never assigned destinations.
5. **Commit hygiene:** isolate the pure `git mv` in one commit for rename
   detection. (Not mentioned.)
6. **Keep the self-containment claim,** reworded — it is a real property, not a
   leftover. (Its item 6 would have deleted it.)
7. **`baseline_original/` decision** is entangled with the root-README rewrite and
   must be made explicitly. (Implied but not surfaced.)

## Do NOT

- Do not move `real_ecology_data/` anywhere other than a sibling of `src/`.
- Do not refactor `real_ecology_runs/psafe_overnight_20260705/code/` — archived
  run evidence, intentionally duplicated.
- Do not mix content edits into the `git mv` commit.
- Do not delete the nested `.gitignore` before merging its five unique entries.
- Do not begin with a dirty working tree.
