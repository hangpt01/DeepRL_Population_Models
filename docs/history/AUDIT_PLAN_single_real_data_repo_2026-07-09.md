# Audit: PLAN_single_real_data_repo_with_dummy_setpoint_2026-07-09

Auditor pass date: 2026-07-09
Audited document: `discrete_action_cont_obser/docs/PLAN_single_real_data_repo_with_dummy_setpoint_2026-07-09.md`
Scope checked against live code under: `discrete_action_cont_obser/`
Status: **audit only** — no files were moved, edited, or deleted in this pass.

---

## 1. Verdict (TL;DR)

The plan is **directionally correct and its code-fact claims check out**. The real
package already implements set-point `r` + cumulative `K`, exactly as the plan
states, and it is safe to promote as the single canonical package.

But the plan has **one architectural hazard** and **several concrete
correctness fixes** that must be pinned down before implementation, or dummy mode
will silently *not* get set-point semantics (the opposite of the stated goal):

1. **HAZARD — the axis is wrong.** The plan makes `control_mode` the semantic
   axis (rename to `setpoint_cumulative`, alias `real_setpoint`) *and* adds a
   `data_mode`. But ~22 code sites branch on the literal `control_mode == "real_setpoint"`,
   and **most of them encode the set-point/cumulative *semantics* that dummy must
   share**, not the data source. If dummy gets a *different* `control_mode`
   string, all those branches fall through to the Tier-2/3 path and dummy loses
   set-point `r`. The safe design is: **keep one control_mode for "set-point r +
   cumulative K" shared by real and dummy, and make `data_mode` the *only* new
   axis**, touching just the ~6 data-source seams. (Finding F1.)
2. **BUG — `realdata.DATA_DIR` breaks on the package move** (off-by-one in the
   hardcoded parent count). (Finding F2.)
3. **BUG — `config.validate()` hard-rejects dummy** because `real_setpoint`
   requires `num_actions == 11`. (Finding F3.)
4. **GAP — several `real_setpoint` branches read real data** (population caps,
   `real_environment`, `resolve_actions`); dummy must not reach them.
   (Finding F4.)
5. **GAP — dataset cache validation** does not distinguish real vs dummy.
   (Finding F5.)
6. **CLARIFY — existing Tier-2/3 action tables are NOT reusable as dummy
   set-point tables** (their `delta_r` are increments, not set-points, and there
   is no dummy `r_min/r_max` source). (Finding F6.)

Recommendation: proceed, but adopt the corrected architecture in §5 (orthogonal
`data_mode`, no `control_mode` rename) and fix F2–F6 as explicit tasks.

---

## 2. Answers To The Plan's Six Audit Questions

**Q1. Does `real_ecology_benchmark` already encode set-point `r` and cumulative
`K` as the plan says?**
**Yes, confirmed.** Verified in code:
- `config.EnvironmentConfig` defaults: `control_mode="real_setpoint"`,
  `accumulator_decay_r=1.0`, `accumulator_decay_K=0.0`, `r_base_low=r_base_high=0.0`
  ([config.py:46-98](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/config.py#L46-L98)).
- `advance_public_controls`: `next_rho = (1-decay_r)*rho + delta_r` → with
  `decay_r=1`, `rho = delta_r(action)` (set-point); `next_kappa = (1-decay_K)*kappa + delta_K`
  → with `decay_K=0`, `kappa = Σ delta_K` (cumulative)
  ([controls.py:67-69](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/controls.py#L67-L69)).
- `private_r_eff` returns `clip(rho, r_min, r_max)` under `real_setpoint`
  (set-point), else `clip(r_base + rho, ...)` (additive)
  ([controls.py:81-99](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/controls.py#L81-L99)).
- `K_eff = clip(K_base + kappa, K_min, K_max)`
  ([controls.py:48](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/controls.py#L48),
  [envs.py:302-309](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/envs.py#L302-L309)).
The plan's "Current Code Facts To Preserve" section is accurate.

**Q2. Name the shared mode `setpoint_cumulative`, or keep/alias `real_setpoint`?**
**Keep `real_setpoint` as the literal semantic mode; do NOT rename.** See F1.
A rename touches ~22 sites and buys nothing but risk; the correct separation is a
new orthogonal `data_mode` field, not a control_mode alias. If a clearer name is
wanted later, do it as a *pure, atomic* rename behind a single
`is_setpoint_cumulative(cfg)` helper that replaces every literal — but that is
cosmetic and should not be bundled with the dummy-mode work.

**Q3. `dummydata.py` vs dummy CSVs?**
**`dummydata.py` first, confirmed.** In-code profiles are far easier to audit, add
no new CSV loader/validator to trust, and cannot drift from a schema. Structure
the module to mirror `RealPopulation`/`RealEffect`/`real_action_table` so the
dispatch in §5 stays symmetric. CSVs can come later if you want to hand-edit
sweeps.

**Q4. Which files/functions need edits when the package is promoted to top-level
`src/`?**
- `realdata.py` `DATA_DIR` (F2 — **must fix, or every real run breaks**).
- `pyproject.toml`: `packages.find` will now discover both `tier2_benchmark` and
  `real_ecology_benchmark`; console `scripts` and project name need updating.
- `Makefile`, test discovery path, config path.
- `cli.py` default `--config configs/real_default.yaml`
  ([cli.py:144](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/cli.py#L144))
  is CWD-relative; keep runs rooted at the folder or make it package-relative.
- No source imports `tier2_benchmark` from inside the real package (verified), so
  the two packages coexist cleanly during transition.

**Q5. Hidden method assumptions tied to `control_mode == "real_setpoint"`, real
population names, 11 actions, or `realdata.NUM_REAL_ACTIONS`?**
**Yes — and this is the core of the work.** There are ~22 literal
`real_setpoint` branch sites. They split into two classes (full table in §4):
- **Semantics (dummy MUST share):** `controls.py` (37,42,81), `envs.py`
  (82,106,171,321), `reward.py:106`, `gate.py:46`, `pipeline.py:46`,
  `collector.py` (273,400), `backend.py:138`, `beliefs.py` (380,423),
  `methods/plus.py:26`, `methods/moor.py:37`.
- **Real-data-specific (dummy must take a DIFFERENT branch):** `config.py`
  (115-117 num_actions==11, 210 real_environment, 367 load_config expansion,
  434 environment_with_kind_defaults), `actions.py:176` (`resolve_actions` →
  `real_action_table`), `beliefs.py:336-342` (re-reads real `pop.caps`).
Also note the model-side `MechanisticProposal` at
[beliefs.py:336](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/beliefs.py#L336)
does `realdata.pops_for(...)[population]` on a family switch — with a dummy
population this is a **KeyError** unless dummy supplies caps another way.

**Q6. Are the proposed tests sufficient to prove `r` is set each step and `K`
stays cumulative for real and dummy?**
**Nearly — strengthen them.** See §6. In particular add: repeated-growth-action
assertions that `rho` stays equal to the set-point (not summed); `K` clip-at-`K_max`
after enough capacity actions; the `private_r_eff` out-of-caps guard raises for
dummy; dummy runs with `real_ecology_data/` made unreadable (proves no real read);
and cache-validation rejects cross-mode reuse.

---

## 3. Verified Code Facts

| Plan claim | Verified? | Evidence |
|---|---|---|
| `control_mode="real_setpoint"` is the real default | ✅ | config.py:46 |
| `decay_r=1`, `r_base_low=r_base_high=0` → `rho` carries set-point | ✅ | config.py:89-97, controls.py:67,81-92 |
| `decay_K=0` → `kappa` cumulative | ✅ | config.py:98, controls.py:68 |
| Real actions read from `action_effects_long.csv`; `delta_r`=set-point, `delta_K`=increment, `stocking_delta` only a10 | ✅ | actions.py:136-164, realdata.py:195-247 |
| Env advances controls before growth; uses `private_r_eff` and `K_eff=clip(K_base+kappa,…)` | ✅ | envs.py:208-217, 302-309 |
| Real setting = 11 actions, 9 populations | ✅ | realdata.py:36-37 (`NUM_REAL_ACTIONS=11`, `NUM_REAL_POPULATIONS=9`) |
| Tier-2/3 `tier2_one_step` recomputes `r_base+delta_r`, `K_base+delta_K` per step | ✅ | envs.py:219-224 |
| `cumulative_capped` can accumulate both r and K | ⚠️ partial | The table exists (actions.py:63-85) but in practice the real path pins `decay_r=1`; Tier-3 uses `decay_r` per config. Claim is fine as written. |
| Real package has no runtime dependency on `tier2_benchmark` or `claude_build` | ✅ | grep: no such imports in the real `src/` |
| `real_ecology_cont_obser/` has no `pyproject.toml` | ✅ | absent; runs via `PYTHONPATH=src` |
| Real data path resolves to `discrete_action_cont_obser/real_ecology_data` | ✅ *today* | realdata.py:28-29 — **but see F2** |

The plan's factual foundation is sound. The issues below are about the
*implementation*, not the diagnosis.

---

## 4. Findings (ranked)

### F1 — HAZARD: make `data_mode` the only new axis; do not rename `control_mode`

The plan (Configuration Design + Phase 2) treats `control_mode` as the semantic
axis (`setpoint_cumulative`, aliasing `real_setpoint`) *and* introduces
`data_mode`. This double axis is the root risk. The ~22 literal `== "real_setpoint"`
branches partition as:

**Semantics — dummy MUST take the SAME branch as real:**
`control_fields_enabled` (controls.py:37); set-point `rho`/reset (controls.py:42,
envs.py:82,106,171); reward-on-true-next-state (envs.py:321); `private_r_eff`
set-point (controls.py:81); `effective_collapse_penalty` yield→0 (reward.py:106);
output namespacing (pipeline.py:46); gate reset (gate.py:46); collector start
coverage (collector.py:273,400); GPU-exception for mechanistic transition
(backend.py:138); model transition set-point (beliefs.py:380,423); PLUS "r known,
fit K" (plus.py:26); MOOR "r known" (moor.py:37).

**Data source — dummy MUST take a DIFFERENT branch:**
num_actions==11 validation (config.py:115-117); `real_environment` builder
(config.py:210); `load_config` population expansion (config.py:367);
`environment_with_kind_defaults` (config.py:434); `resolve_actions` →
`real_action_table` (actions.py:176); real caps re-read (beliefs.py:336-342).

The plan's own Phase 3 relies on dummy using **exactly** `decay_r=1, r_base=0,
private_r_eff→clip(rho)` — which only fires when `control_mode == "real_setpoint"`
(controls.py:81). So Phase 2 (control_mode is the axis) and Phase 3 (dummy reuses
the real_setpoint config) are only mutually consistent if **dummy keeps
`control_mode == "real_setpoint"`**. Therefore:

**Recommendation:** keep `control_mode = "real_setpoint"` (meaning: set-point r +
cumulative K) shared by both real and dummy. Add `data_mode: real | dummy`. Only
the six data-source sites above branch on `data_mode`. Every semantic branch is
untouched and works for dummy automatically. (Optionally add read-only helpers
`uses_real_data(cfg)`/`uses_dummy_data(cfg)` for the six sites; skip
`is_setpoint_cumulative` unless you do the full atomic rename separately.)

If the literal name `real_setpoint` for a mode also used by dummy feels wrong,
that is a naming nit — solve it with a comment or a later pure rename, not by
forking the semantic axis now.

### F2 — BUG: `realdata.DATA_DIR` off-by-one after the package move

[realdata.py:28-29](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/realdata.py#L28-L29):

```python
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT.parent / "real_ecology_data"
```

The `.parent` count is hardcoded to the current nesting
`real_ecology_cont_obser/src/real_ecology_benchmark/`, where it resolves to
`discrete_action_cont_obser/real_ecology_data` (correct today). After the Phase-1
move to `discrete_action_cont_obser/src/real_ecology_benchmark/`, `PACKAGE_ROOT`
becomes `discrete_action_cont_obser` and `DATA_DIR` becomes
`<repo-root>/real_ecology_data` — one level too high → `FileNotFoundError` on
every real run.

**Fix (post-move):** `DATA_DIR = PACKAGE_ROOT / "real_ecology_data"` (drop the
`.parent`). Add a fail-fast assertion that `DATA_DIR/species.csv` exists so a
future move surfaces immediately. Cover with a test that asserts `DATA_DIR`
resolves to a directory containing `species.csv`.

### F3 — BUG: `config.validate()` rejects dummy set-point outright

[config.py:115-119](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/config.py#L115-L119):

```python
if self.control_mode == "real_setpoint":
    if self.num_actions != realdata.NUM_REAL_ACTIONS:   # 11
        raise ValueError("real_setpoint requires num_actions == 11")
elif self.num_actions not in {5, 10}:
    raise ValueError("num_actions must be 5 or 10")
```

Under the F1 design (dummy shares `control_mode == "real_setpoint"`), a dummy
config with 5 or 10 actions hits the `num_actions == 11` guard and is rejected.
**Fix:** gate the 11-action requirement on `data_mode == "real"`; allow `{5,10}`
(or a chosen dummy count) when `data_mode == "dummy"`.

### F4 — GAP: real-data branches must be guarded by `data_mode`

These sites reach into `real_ecology_data` and will fail or mislead for dummy:
- `resolve_actions` (actions.py:176) → `real_action_table` → `realdata.effects_for`.
  Add a `data_mode == "dummy"` → `dummy_action_table(cfg)` branch.
- `real_environment` (config.py:170-240) and `real_environment_like`
  (config.py:405-423): add `dummy_environment(...)` and route
  `environment_with_kind_defaults`/`load_config` on `data_mode`.
- `load_config` population-expansion (config.py:361-388): the compact
  `population`+`kind` expansion is real-only; dummy configs need a parallel
  `dummy_environment` build path.
- `MechanisticProposal` family switch (beliefs.py:336-342) does
  `realdata.pops_for(...)[population]` → **KeyError** for a dummy population.
  Dummy must provide `r_min/r_max` per family from `dummydata`, or skip the
  re-read when `data_mode == "dummy"`.

### F5 — GAP: dataset cache validation ignores data source

[pipeline.py:54-78](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/pipeline.py#L54-L78)
`_validate_dataset_cell` checks `population`, `num_actions`, `control_mode`, …
but not a `data_mode`. With F1 (shared control_mode), a dummy and a real cell
could share `control_mode` and — if output paths collide — a stale dataset could
be reused across data sources. `population` differs so it is partly protected,
but add `data_mode` (and the effective backend/reward namespacing already at
pipeline.py:46) to the validated keys and to the output-dir namespacing.

### F6 — CLARIFY: existing action tables are not dummy set-point tables

The plan's Phase 3/5 hint at reusing 5/10-action tables. But `TIER2_*`/`TIER3_*`
`delta_r` values are **increments** for `tier2_one_step`/`cumulative_capped`
([actions.py:42-85](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/actions.py#L42-L85)),
not absolute set-points, and there is no dummy `r_min/r_max` source. A dummy
set-point table needs:
- absolute `delta_r` values that are **valid set-points inside `[r_min, r_max]`**
  (else `private_r_eff` raises at
  [controls.py:86-91](discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/controls.py#L86-L91)),
- dummy `r_min/r_max/K_base/K_max/K_min/N0/safety_*` in `dummydata`,
- a0 as the baseline set-point (matching `initial_controls` reading `action[0].delta_r`).

The plan's "Dummy Data Design" section already proposes fresh dummy values, which
is the right call — just make explicit that the old tables are *not* reused for
set-point r.

### F7 — SEQUENCING: don't delete `tier2_benchmark` until top-level tests/configs are detached

Top-level `tests/` (`test_methods.py`, `test_environment.py`,
`test_tier3_controls.py`, …) and `configs/` (`default.yaml`, `full.yaml`,
`tier3_12h.yaml`) target `tier2_benchmark`. Phase 1 adds real tests/configs into
the *same* top-level dirs; `unittest discover -s tests` would then run both
suites. Decide up front: either (a) keep synthetic and real tests in separate
discovery roots, or (b) migrate/retire the synthetic tests in the same PR that
removes `tier2_benchmark` (Phase 5), so discovery is never half-broken.

---

## 5. Corrected Implementation Plan

Adopt the plan's phases with these changes.

**Axis decision (replaces plan §"Configuration Design"):**
- Add one field: `data_mode: str = "real"  # real | dummy` on `EnvironmentConfig`.
- Keep `control_mode = "real_setpoint"` as the shared "set-point r + cumulative K"
  semantics for both data modes. Do **not** add `setpoint_cumulative`.
- Optional helpers (read-only): `uses_real_data(cfg)`, `uses_dummy_data(cfg)`.

**Phase 1 — promote package (add F2 fix):**
- Move `real_ecology_cont_obser/src/real_ecology_benchmark/` → `src/`.
- **Fix `realdata.DATA_DIR`** to `PACKAGE_ROOT / "real_ecology_data"` and add the
  `species.csv` existence assertion + test.
- Update `pyproject.toml` (name, `packages.find` now finds both packages, console
  script), `Makefile`, test/config locations. Keep the two packages coexisting
  until Phase 5.

**Phase 2 — data_mode plumbing (replaces alias work):**
- Add `data_mode` to `EnvironmentConfig`; validate `∈ {real, dummy}`.
- Fix F3: gate `num_actions == 11` on `data_mode == "real"`.
- Fix F5: add `data_mode` to `_validate_dataset_cell` keys and output namespacing.
- Route the six data-source sites (F4) on `data_mode`.
- Acceptance: **all existing real configs pass unchanged** (they omit
  `data_mode`, defaulting to `real`).

**Phase 3 — dummy set-point data (F6):**
- Add `dummydata.py` mirroring `RealPopulation`/`RealEffect`, with a dummy
  profile and a `dummy_action_table(cfg)` of valid in-caps set-points.
- Add `dummy_environment(...)` (parallels `real_environment`), pinning
  `control_mode="real_setpoint"`, `decay_r=1`, `r_base=0`, `decay_K=0`.
- Branch `resolve_actions`, `beliefs.MechanisticProposal` caps, `load_config`,
  and `environment_with_kind_defaults` on `data_mode` (F4).

**Phase 4 — tests:** see §6.

**Phase 5 — retire `tier2_benchmark` (with F7 sequencing).**

**Phase 6 — artifact cleanup:** unchanged from the plan; keep its git-tracked
caution (see §7).

---

## 6. Test Sufficiency (strengthen Phase 4)

Keep the plan's list and add:
- **Set-point non-accumulation (real & dummy):** apply the same growth action
  twice; assert `rho_t2 == delta_r` (not `2·delta_r`) and `r_eff` unchanged.
- **K cumulative + clip:** apply capacity actions repeatedly; assert `kappa`
  increases and `K_eff` saturates exactly at `K_max`.
- **Caps guard:** a dummy set-point outside `[r_min, r_max]` makes `private_r_eff`
  raise (controls.py:86-91) — assert the guard fires.
- **No real read in dummy:** run a dummy smoke with `realdata.DATA_DIR` pointed at
  a nonexistent path (monkeypatch); assert dummy still runs → proves dummy never
  touches `real_ecology_data/`.
- **DATA_DIR resolution (F2):** assert `realdata.DATA_DIR/species.csv` exists.
- **Cache cross-mode rejection (F5):** a dummy dataset must be rejected for a real
  cell and vice-versa.
- **Leakage boundary unchanged:** public info exposes `rho/kappa/K_eff` only;
  `r_eff_true` and `state` stay evaluator-only for both modes
  (envs.py:124-152).

---

## 7. Cleanup Cautions (Phases 5–6)

- The plan's git-tracking caution is correct and important. Prior handoffs and the
  P_safe run note that some `outputs/`, `logs/`, and `wandb/` files are **tracked
  in git despite `.gitignore`**. Run `git ls-files` on every target before any
  `rm`/`git rm`, exactly as the plan says.
- `real_ecology_runs/psafe_overnight_20260705/` is the **paper-facing
  reproducibility bundle** (~11G). Archive, never bulk-delete. Preserve at least
  `analysis/PAPER_RESULT_PACKAGE.md`, the results `.tex/.pdf`, `report_figures/`,
  `all_metrics.csv`, `rollup.csv`, decision notes, and `manifests/`.
- Keep `real_ecology_data/` **tracked** — it is the authoritative real data and
  the whole point of the single-folder repo. Confirm `.gitignore` does not sweep
  it (current ignores are generic `outputs/`, `data/`, `*.npz`).
- Known pre-existing issue to be aware of, unrelated to this refactor: a CLI
  aggregate bug around `manifest.py:201` was noted in the P_safe run; verify it
  is not on the smoke/aggregate path you use for acceptance.

---

## 8. Bottom Line For The User

- The plan correctly diagnoses the code and is safe to build on.
- **Change the one design decision:** dummy and real should share
  `control_mode="real_setpoint"` (set-point r + cumulative K); add an orthogonal
  `data_mode: real|dummy` and branch only the six data-source seams. Do not rename
  `control_mode`.
- **Fix before shipping:** `DATA_DIR` off-by-one on the move (F2), the
  `num_actions==11` guard (F3), the real-data branches dummy must avoid (F4,
  incl. the `beliefs.py` KeyError), and cache validation (F5). Build dummy
  set-points fresh (F6); sequence the `tier2_benchmark` removal after tests
  detach (F7).
- With those changes the acceptance commands in the plan are the right gate; add
  the strengthened tests in §6.
