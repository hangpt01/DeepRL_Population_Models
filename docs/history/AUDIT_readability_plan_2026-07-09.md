# Audit: PLAN_readability_fixes_axes_and_setpoint_rename_2026-07-09

Auditor: Claude
Date: 2026-07-09
Audited: `docs/PLAN_readability_fixes_axes_and_setpoint_rename_2026-07-09.md`
Checked against live code under `src/real_ecology_benchmark/`.

## Verdict

The plan is sound and well-sequenced. The commit split (docs → rename →
tier-retire) is the right order and keeps review tractable. Fix 1 (glossary) and
Fix 2 (rename) are clearly worth doing; Fix 3 (retire "Tier") is worth doing but
carries reference-integrity risk the plan under-weights.

One correction matters most: **not every `control_mode == "real_setpoint"` branch
is pure semantics — three of them use `real_setpoint` as a proxy for the data
source.** Blindly replacing all literals with `is_setpoint_cumulative()` would
break two of those. Details in Q3.

---

## Answers to the plan's 7 audit questions

### Q1 — Is `setpoint_cumulative` the best name, or `setpoint_r_cumulative_K`?

**Use `setpoint_cumulative`.** It names both axes of the semantics (set-point
control, cumulative capacity) without baking variable letters into an identifier.
`setpoint_r_cumulative_K` is more precise but verbose and reads like a formula;
the precision belongs in the glossary line ("set-point `r`, cumulative `K`"), not
in the enum value. Keep the value short; document the meaning once.

### Q2 — Normalize aliases immediately, or preserve the raw string + rely on the helper?

**Normalize immediately to the canonical value at every input boundary; keep the
canonical value in memory.** Rationale:
- If the in-memory `EnvironmentConfig.control_mode` is *always* canonical, then
  everything downstream (metadata emission, output-dir namespacing, cache keys,
  logs, `to_dict`) is automatically correct and consistent — you don't have to
  find and route every consumer through `is_setpoint_cumulative()`, and a missed
  literal can't leak the old label into a generated artifact.
- The only code that should ever see the legacy string is the input edges: YAML
  load (`load_config`), direct `EnvironmentConfig(...)` construction, and reading
  **recorded cache metadata**. Normalize on the way in at all three.

**Critical caveat that the plan must state:** cache validation compares *recorded*
metadata against the *expected* config. If you normalize the expected side but not
the recorded side, every old cache mismatches. So normalize **both sides** before
comparison (see Q3 site 2 and Q4). Net: normalize eagerly, and treat
`normalize_control_mode()` as the single choke point applied at all boundaries —
including the recorded-metadata read.

Keep `is_setpoint_cumulative()` as a convenience for branch sites, but the
invariant should be "in-memory config is canonical," not "raw string preserved."

### Q3 — Any branch where `"real_setpoint"` still means "real data," not set-point semantics?

**Yes — three sites. The rest (the large majority) are pure semantics and are safe
to convert to the helper.**

Pure-semantics sites (safe → `is_setpoint_cumulative(cfg)`): `controls.py:37,42,81`;
`envs.py:82,106,171,321`; `reward.py:106`; `beliefs.py:336,383,426`; `gate.py:46`;
`backend.py:138`; `actions.py:210,212`; `methods/moor.py:37`; `methods/plus.py:26`;
`collector.py:400`.

Proxy-for-data-source sites — **do NOT blind-replace**:

1. **`config.py:531-534` (`load_config` default).**
   ```python
   data_mode = env_raw.get("data_mode",
                           "real" if control_mode == "real_setpoint" else "synthetic")
   ```
   This is *default data-source inference*: a set-point config that omits
   `data_mode` defaults to `real`. That default is defensible (paper-facing), but
   it is data-source logic, not semantics. Keep the behavior; just make it robust
   to the alias: compare on `normalize_control_mode(control_mode) ==
   SETPOINT_CUMULATIVE`, still defaulting to `"real"`.

2. **`pipeline.py:76-79` (`_validate_dataset_cell` back-compat).**
   ```python
   return "real" if recorded_value("control_mode") == "real_setpoint" else "synthetic"
   ```
   Infers a missing `data_mode` for **legacy caches** from the recorded control
   mode. It already conflates dummy (an old set-point cache → "real"), but that is
   harmless because dummy postdates `data_mode`, so real dummy caches always
   record it. After the rename this line must accept **both** `real_setpoint` and
   `setpoint_cumulative` as recorded values → normalize the recorded string first.

3. **`collector.py:273` (`is_real`).**
   ```python
   is_real = env.cfg.control_mode == "real_setpoint"
   ```
   The **behavior is correct** (dummy takes the same set-point start path, which is
   intended), but the **variable name lies** — it's true for dummy too. This is
   precisely the misnaming the rename should fix: rename to `is_setpoint =
   is_setpoint_cumulative(env.cfg)`. Pure cleanup, no behavior change.

So the plan's phrase "replace literal semantic branches with the helper" is right
for most sites, but sites 1 and 2 are *data-source defaults* that must retain
their `→ "real"` / `→ "synthetic"` mapping, and site 3 is a rename-the-variable
fix. Call these three out explicitly in the implementation.

### Q4 — Accept old cache metadata as equivalent, or invalidate?

**Accept as equivalent — do not invalidate.** The rename changes only the *string
label*; the control mode's meaning, the dynamics, the reward, and the logged data
are all identical. Invalidating would force regeneration of expensive real
datasets and diverge from the P-safe provenance for a cosmetic change. Treat
`real_setpoint ≡ setpoint_cumulative` in `_validate_dataset_cell` by normalizing
both recorded and expected `control_mode` before comparison. Add a test: an old
cache with `control_mode="real_setpoint"` validates against a new config carrying
`setpoint_cumulative`.

### Q5 — Live docs beyond `README.md` and `docs/architecture.md`?

The live (non-historical) reference docs are:
`docs/architecture.md`, `docs/methods.md`, `docs/experiment_protocol.md`,
`docs/reproducibility.md`, plus the two supervisor-facing results docs
`docs/22_6_Continuous_Observation_Experiment_Results.tex` and
`docs/22_6_Continuous_Observation_New_Baselines.tex`.

Only `docs/architecture.md` currently contains `real_setpoint` / "Tier" wording,
so it is the primary target. But **sweep all six** for the old names during Fix 2
and Fix 3 (a supervisor will read `methods.md`/`experiment_protocol.md` as current
truth). Everything under `docs/real_ecology_history/`, `docs/claude_build_report/`,
the `29_6_*`, `HANDOFF_*`, `AUDIT_*`, `CODEX_*`, and `PLAN_*` files is historical —
leave as-is.

### Q6 — Rename the code value `tier2_one_step` now?

**No — keep it as a legacy internal identifier; fix only the reader-facing name in
docs.** Reasons:
- `real_setpoint` is renamed because it is *actively misleading* (it fires on
  dummy, implying a data source it doesn't control). `tier2_one_step` /
  `cumulative_capped` are merely *jargon* — not wrong, just opaque — and a glossary
  entry fully resolves that.
- These modes serve only the retained synthetic regression/reference path, the
  least paper-facing surface; a code-value rename means another alias layer,
  another cache-compat mapping, and more churn for the lowest ROI.
- Keeping Fix 2 focused on the one genuinely-wrong name makes the diff far easier
  to review — which is the whole point.

Document `tier2_one_step` → "one-step synthetic control" and `cumulative_capped`
→ "cumulative-control synthetic" in the glossary and stop there.

### Q7 — Are the proposed file/script names too long?

Split the decision by visibility:
- **Worth it (rename):** `configs/tier3_12h.yaml` → `configs/cumulative_controls_12h.yaml`
  and `tests/test_tier3_controls.py` → `tests/test_cumulative_controls.py`. These
  are read and cited often; the clarity gain justifies the length.
- **Lighter touch (Slurm wrappers):** `make_cumulative_controls_12h_manifests.py`,
  `run_cumulative_controls_12h_aggregate.sh`,
  `launch_cumulative_controls_12h_after_probe.sh` are a mouthful and are
  operational scripts a supervisor rarely opens. Either accept the length (one-time
  rename, shell-completed) or use a slightly shorter stem
  (e.g. `cumulative_controls_12h` is fine; avoid re-introducing jargon like
  `cumctl`). Do not over-optimize here.
- **Output/job-name roots** (`outputs/tier3_c0_g1g2_12h`, `tier3_%A_%a.out`): renaming
  these is **forward-only** — old artifacts already live under the old paths and
  should not be migrated. New runs will land under the new root; note this
  explicitly so nobody expects historical outputs to move.

---

## Additional findings (not in the plan's questions)

**A. Fix 3 reference-integrity risk (strengthen the plan here).** Renaming
`configs/tier3_12h.yaml` breaks the scripts that name it. Confirmed live
references:
```
scripts/make_tier3_12h_manifests.py
scripts/slurm/run_tier3_12h_gates.sh
scripts/slurm/run_tier3_12h_row.sh
scripts/slurm/run_tier3_12h_aggregate.sh
scripts/slurm/run_tier3_12h_gate_row.sh
scripts/slurm/launch_tier3_12h_after_probe.sh
```
Before renaming any tier-named file, `rg` the OLD name as a string across
`scripts/`, `Makefile`, `configs/`, and `docs/` and update every reference in the
same commit. The plan's acceptance checks (`rg` for tier, `bash -n`, unit suite)
catch some of this but won't catch a Slurm script silently pointing at a renamed
config until it's submitted — so do the string sweep deliberately, not just the
post-hoc `rg`.

**B. Enumerate the validation-message and constant sites in Fix 2.** Beyond branch
logic, the literal appears in user-facing strings and the mode set that also need
the canonical name (or an alias-aware message):
`config.py:26` (`CONTROL_MODES`), `config.py:122,128,136,138` (ValueError
messages), `config.py:294-295` (`dummy_environment` guard), `envs.py:31`
(docstring). Add `setpoint_cumulative` to `CONTROL_MODES` and keep `real_setpoint`
accepted; update the messages to say the canonical name.

**C. Add a metadata round-trip acceptance test.** Assert that a config built from
legacy YAML (`real_setpoint`) emits `setpoint_cumulative` in generated dataset
metadata, and that re-loading that metadata validates. This proves the "canonical
in memory, alias at the edges" invariant (Q2) end-to-end.

**D. `pyproject.toml` author string** ("Tier-2 benchmark contributors") — trivial;
fold into the Fix 3 doc/prose pass.

---

## Bottom line

Approve the plan with these adjustments:
1. **Q2:** normalize eagerly to canonical in memory; apply `normalize_control_mode`
   at *all* input edges including recorded cache metadata (both sides of the
   comparison).
2. **Q3:** treat `config.py:531-534`, `pipeline.py:76-79`, and `collector.py:273`
   as special — the first two keep data-source-default behavior (normalize, don't
   drop the `→ real` map); the third is a variable rename. All other sites convert
   to the helper.
3. **Q4:** accept old caches as equivalent (normalize both sides); add the test.
4. **Q6:** do NOT rename `tier2_one_step`/`cumulative_capped` at the code level.
5. **Q7 / Finding A:** rename the high-visibility config + test; go light on Slurm
   wrappers; treat output-root renames as forward-only; do a pre-rename string
   sweep of the old names across scripts/Makefile/configs/docs.
6. **Findings B–C:** include the constant/message sites and a metadata round-trip
   test in Fix 2's scope.

No behavior changes anywhere — this stays a pure readability/labeling refactor,
which is exactly right for supervisor-facing polish.
