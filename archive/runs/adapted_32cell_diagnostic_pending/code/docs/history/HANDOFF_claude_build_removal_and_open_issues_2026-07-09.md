# Handoff: Open Issues + Guide To Remove `claude_build`

Date: 2026-07-09
Author: Claude (audit of codex's implementation)
Repo root: `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`
(same tree as `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`)

This document has two parts:

- **Part A** — open issues in the current `data_mode` implementation, verified by
  running the code.
- **Part B** — a safe, step-by-step guide for codex to remove `claude_build/`.

> Status: instructions only. Get user approval before any `git rm`/`rm`. Do all
> destructive work as its own commit(s), separate from the code refactor.

---

## Part A — Open Issues (audit of the implemented refactor)

The implementation is **functionally correct** — I verified it, not just the
summary:

- Real tests 38/38, dummy tests 7/7, full mixed `tests/` discovery 46/46; real +
  dummy smokes run.
- Independent semantic probe (dummy mode): repeating a growth action holds
  `rho` constant (set-point, no accumulation); repeating a capacity action
  accumulates `kappa` and clips `K_eff` at `K_max`.
- `realdata.DATA_DIR` fix, `data_mode` seams (`resolve_actions`,
  `dummy_environment`, `beliefs.MechanisticProposal`, cache validation +
  `data_{mode}` output namespacing) are all correctly routed.

The following are still open. None block using the benchmark today, but they
should be closed before/around the cleanup.

### A1 — BLOCKER for cleanup sequencing: the refactor is not committed yet

Everything the refactor added is **untracked in git**:

```
git ls-files discrete_action_cont_obser/src/real_ecology_benchmark   -> 0
git ls-files discrete_action_cont_obser/tests/real tests/dummy        -> 0
git ls-files discrete_action_cont_obser/configs/dummy_setpoint_*.yaml  -> 0
```

**Do not run any destructive cleanup (Part B, or removing the duplicate in A2)
until this work is committed.** Commit the good refactor first so it is
recoverable, then do each removal as its own commit. This is the single most
important sequencing rule here.

### A2 — Divergent duplicate package copy (drift hazard)

The package was **copied, not moved**. Both of these now exist and have diverged:

- new (canonical): `discrete_action_cont_obser/src/real_ecology_benchmark/`
- old (stale):      `discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/`

Eight `.py` files differ (`actions, beliefs, cli, config, evaluator, __init__,
pipeline, realdata`), and the old copy is missing `dummydata.py` and the
`data_mode` edits. Risk: someone edits or imports the stale copy.

Also, the old folder still holds subdirs that were **not** migrated:

```
real_ecology_cont_obser/{scripts, docs, log, outputs, configs, tests, README.md}
```

None of `real_ecology_cont_obser/` is git-tracked (0 files), so removal is a
plain filesystem delete — but first **check for unique content worth keeping**,
especially `real_ecology_cont_obser/scripts/` and `real_ecology_cont_obser/docs/`
(the `src/`, `configs/`, `tests/` were already promoted).

Suggested resolution (its own commit, after A1):
1. `diff -rq real_ecology_cont_obser/scripts <wherever scripts should live>` and
   migrate anything still needed (e.g. into a top-level `scripts/`).
2. Confirm `real_ecology_cont_obser/docs/` has no doc not already under
   `discrete_action_cont_obser/docs/`.
3. Archive `real_ecology_cont_obser/outputs` + `log` only if provenance matters,
   then `rm -rf real_ecology_cont_obser/`.
4. `rg -n "real_ecology_cont_obser" discrete_action_cont_obser/{src,tests,configs,scripts,Makefile,README.md}`
   must return nothing before deleting.

### A3 — Synthetic package retirement still pending (Phase 5, known/expected)

`src/tier2_benchmark/` remains, and `pyproject.toml` still exposes the `tier2`
console script alongside `real-ecology` / `ecology-benchmark`. This is the
deferred Phase-5 step, not a bug. Keep it until you deliberately retire the
synthetic path; the mixed `tests/` tree (legacy tier2 + real + dummy) currently
passes, so there is no rush. When you do retire it: detach/relocate the legacy
`tests/test_*.py` and `configs/{default,full,tier3_12h}.yaml` in the same commit
that removes `tier2_benchmark`, and drop the `tier2` console script.

### A4 — Minor robustness note (optional)

`realdata.DATA_DIR = PACKAGE_ROOT / "real_ecology_data"` is correct for the
current `src/` layout but is a fixed parent-count. `EnvironmentConfig` already
has a `data_dir` field that most functions honor (`cfg.data_dir or
realdata.DATA_DIR`); preferring an explicit config `data_dir` and treating the
computed path as fallback would survive a future move. Low priority.

---

## Part B — Guide To Remove `claude_build/`

### Why it is safe to remove

- **No external runtime dependency.** All references to `claude_build` live
  *inside* `claude_build` itself (its own scripts referencing `$PWD/claude_build`).
  Verified: `rg -n claude_build discrete_action_cont_obser/{src,tests,configs,scripts}`
  returns nothing. `discrete_action_cont_obser/` does not import or shell into it.
- It is the older **discretized-state** predecessor, superseded by
  `discrete_action_cont_obser/`. Its data is synthetic (simulator-generated), so
  no real ecological data is lost.

### What makes it NOT a trivial delete

- **388 git-tracked files** despite `.gitignore` — including 77 under
  `outputs/`, 6 under `logs/`, 98 under `wandb/`. A filesystem `rm` alone leaves
  git history dirty; you must `git rm`.
- **`claude_build/.wandb_env` is a credentials file** (gitignored, untracked, on
  disk). Do not print or commit it; back it up if the W&B token is still needed.
- The folder is **very large** (per prior handoff: `outputs` ~2.2G, `wandb`
  ~372M, thousands of files). Do not `du`/`find` the whole tree casually — it is
  slow. Use `git ls-files` (indexed, fast) for tracked counts.
- **`baseline_original/`** (repo root) is used **only** by claude_build's
  `HmMDPAdapter`. Removing claude_build orphans it — see Step 7.

### Pre-flight (safety gate — do these first)

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models

# 1. The refactor MUST be committed already (Issue A1). If not, stop and commit it.
git status --short discrete_action_cont_obser | head

# 2. Re-confirm nothing outside claude_build depends on it at runtime (expect no hits).
rg -n "claude_build" discrete_action_cont_obser/src discrete_action_cont_obser/tests \
   discrete_action_cont_obser/configs discrete_action_cont_obser/scripts

# 3. See exactly what is tracked (expect ~388).
git ls-files claude_build | wc -l
```

Do not proceed unless step 1 is clean (refactor committed) and step 2 is empty.

### Step 1 — Preserve `claude_build/docs/`, discard everything else (DECIDED)

User decision (2026-07-09): **keep the docs, remove the code and all artifacts.**

`claude_build/docs/` is 2.2M and **13 git-tracked files**:

```
claude_build/docs/EXPERIMENT_REPORT.md
claude_build/docs/EXPERIMENT_REPORT.docx
claude_build/docs/latex/main.tex
claude_build/docs/latex/main.pdf
claude_build/docs/latex/figures/fig1..fig9_*.png   (9 figures)
```

Relocate them so they survive the folder deletion (they are tracked, so use
`git mv` to preserve history). The repo already has a top-level `docs/`:

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
git mv claude_build/docs docs/claude_build_report
```

(Destination name `docs/claude_build_report/` is a suggestion — rename if you
prefer.) After this, `claude_build/` no longer contains any docs, and everything
left in it is code + generated artifacts to be removed.

Everything else in `claude_build/` — `src/`, `config/`, `scripts/`, `tests/`,
`outputs/`, `wandb/`, `logs/`, checkpoints, `tune116/`, `README.md`,
`.wandb_env` — is discarded.

### Step 2 — Handle the secret

```bash
# If the W&B token is still needed elsewhere, copy it somewhere safe (outside the repo).
# Do NOT cat/echo its contents. It will be destroyed with the folder.
cp claude_build/.wandb_env ~/secure_backup/claude_build.wandb_env   # only if needed
```

### Step 3 — (Optional) cold-storage snapshot of the code

The docs are already preserved in-repo by Step 1, so no archive is required. Only
if you want a recoverable snapshot of the *code* (not the multi-GB artifacts):

```bash
mkdir -p ~/archive
tar czf ~/archive/claude_build_code_$(date +%Y%m%d).tar.gz \
  claude_build/config claude_build/src claude_build/scripts \
  claude_build/tests claude_build/README.md
```

Skip this entirely if you do not want a code snapshot — git history still
contains it up to this commit.

### Step 4 — Remove from git index and disk

Do this **after** Step 1's `git mv` (so the docs are already out). `git rm -r`
only removes *tracked* files, leaving the thousands of untracked `outputs/`/
`wandb/` files and `.wandb_env` behind — so use this two-step to clear both:

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
git rm -r --cached claude_build      # untrack the remaining tracked files (375 after docs moved out)
rm -rf claude_build                  # delete everything on disk (tracked + untracked + .wandb_env)
```

### Step 5 — Remove dead references

- **`.gitignore`**: delete the claude_build-specific lines (currently lines 8–17):
  `claude_build/scripts/slurm_ablation*.sh`, `slurm_online_sweep.sh`, `logs/`,
  `outputs/`, `wandb/`, `.wandb_env`, `.wandb_env.*`.
- **Repo-root `README.md`**: remove any `claude_build` mention.
- Re-scan: `rg -n "claude_build" . -g '!.git'` should return nothing (or only
  historical mentions inside `discrete_action_cont_obser/docs/` handoffs, which
  are fine to leave as history).

### Step 6 — Verify the repo still works

`discrete_action_cont_obser/` never depended on claude_build, so this should all
still pass:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests/real -v
PYTHONPATH=src python -m unittest discover -s tests/dummy -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
```

### Step 7 — Handle orphaned `baseline_original/` (separate decision)

```bash
# After claude_build is gone, check if anything still references baseline_original.
rg -n "baseline_original" . -g '!.git'
```

If the only references were inside claude_build (expected, via `HmMDPAdapter`),
`baseline_original/` is now dead. Archive + remove it as a **separate** commit if
the AAAI21 hmMDP baseline is no longer in scope. Ask the user — do not assume.

### Step 8 — Commit

```bash
git add -A
# review: docs/claude_build_report/ additions (renamed from claude_build/docs),
# claude_build/* deletions, and .gitignore/README edits — nothing else.
git status
git commit -m "Archive claude_build docs to docs/claude_build_report; remove superseded claude_build benchmark"
```

Doing the docs `git mv` and the code removal in the **same commit** keeps the
rename visible to git as a move (history preserved) rather than a delete+add.

### The bigger "one folder" goal (follow-up, not part of this task)

Removing `claude_build` does not by itself leave a single-folder repo. The repo
root still holds `outputs/`, `logs/`, `docs/`, `baseline_original/`,
`slurm-55471800.out`, and a root `README.md`. Consolidating to *only*
`discrete_action_cont_obser/` (or making it the repo root) is a further
restructuring — flag it to the user, keep it out of the claude_build-removal
commit.

### Hard "Do NOT" list

- Do not touch `discrete_action_cont_obser/real_ecology_data/` — authoritative
  real data.
- Do not delete `real_ecology_runs/psafe_overnight_20260705/` (paper provenance
  bundle) as part of this — separate, archive-first decision.
- Do not `rm -rf claude_build` before `git rm --cached` — you would leave 388
  ghost entries in the index.
- Do not print or commit `.wandb_env`.
- Do not start any deletion while the refactor (Part A1) is still uncommitted.
```
