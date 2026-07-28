# Plan: Readability Fixes for Supervisor-Facing Benchmark Code

Date: 2026-07-09

Status: draft for implementation after Claude audit. Incorporates
`AUDIT_readability_plan_2026-07-09.md`.

## Goal

Make the unified `real_ecology_benchmark` package easier to run, scan, and
explain now that it supports real ecology data, the set-point dummy profile, and
legacy continuous-state synthetic experiments in one code path.

The code is already functionally unified. These fixes are readability and
reviewability work, not a change to the scientific setting.

## Current Diagnosis

Running real vs dummy is already easy:

- real: `configs/real_smoke.yaml`, `configs/real_default.yaml`, or
  `--data-mode real`;
- dummy: `configs/dummy_setpoint_smoke.yaml`,
  `configs/dummy_setpoint_default.yaml`, or `--data-mode dummy`;
- both use the same `generate -> run -> gate -> aggregate` pipeline;
- both share the intended action semantics: actions set `r` directly and update
  `K` cumulatively.

The readability debt is that this shared action semantic is currently named
`control_mode: real_setpoint`. That name made sense when only the real ecology
fork used it, but it is now misleading because dummy data also intentionally
uses the same mode. The actual data-source axis is `data_mode`.

Claude audit correction: most `real_setpoint` branches are semantic and should
be renamed, but three sites use the old value as a data-source/defaulting proxy.
Those sites need explicit handling rather than a blanket helper replacement.

## Fix 1: Add an Axes, Naming, and Glossary Section

### Target Files

- Update existing `docs/architecture.md`.
- Add a short pointer from `README.md`.
- Do not create a separate uppercase `docs/ARCHITECTURE.md`; the repo already
  has `docs/architecture.md`, and duplicate case-only names are noisy.

### Content to Add

Add a supervisor-facing section near the top of `docs/architecture.md`:

```text
## Benchmark Axes

data_mode says where the ecological cell comes from.
control_mode says how an action changes the population dynamics.
```

Use a valid-combinations table:

| `data_mode` | Data source | Valid `control_mode` | Meaning |
| --- | --- | --- | --- |
| `real` | CSV ecology tables in `real_ecology_data/` | `real_setpoint` now, canonical `setpoint_cumulative` after Fix 2 | action-specific set-point `r`; cumulative `K` |
| `dummy` | in-code dummy ecology profile | `real_setpoint` now, canonical `setpoint_cumulative` after Fix 2 | same semantics as real, but small and synthetic |
| `synthetic` | in-code continuous-state simulator | `tier2_one_step`, `cumulative_capped` | synthetic controls retained for regression/reference experiments |

Add a reader-facing naming table that avoids the project-internal "Tier" labels:

| Reader-facing name | Current code/config identifier | What distinguishes it |
| --- | --- | --- |
| discretized-state synthetic benchmark | archived `claude_build/` lineage | synthetic data with discretized states |
| continuous-state one-step synthetic benchmark | `data_mode: synthetic`, `control_mode: tier2_one_step` | continuous latent abundance; one-step action effects |
| continuous-state cumulative-control synthetic benchmark | `data_mode: synthetic`, `control_mode: cumulative_capped` | continuous latent abundance; public cumulative control state |
| real-ecology set-point benchmark | `data_mode: real`, `control_mode: real_setpoint` now, canonical `setpoint_cumulative` after Fix 2 | real ecology tables; action sets `r`, accumulates `K` |
| dummy-ecology set-point benchmark | `data_mode: dummy`, same control mode as real | small in-code profile for quick experiments with real-like action semantics |

Add a vocabulary block:

- `real`: paper-facing real ecology table setting.
- `dummy`: quick in-code ecology-like profile with the same set-point `r` and
  cumulative `K` semantics as real.
- `synthetic`: in-code continuous-state synthetic controls preserved for
  regression tests and reference experiments.
- `tier2_one_step`: legacy code identifier for the continuous-state one-step
  synthetic setting. Reader-facing docs should call this "one-step synthetic",
  not "Tier-2".
- `cumulative_capped`: continuous-state cumulative-control synthetic setting.
  Reader-facing docs should call this "cumulative-control synthetic", not
  "Tier-3".
- `setpoint_cumulative`: proposed replacement name for `real_setpoint`.

Add a short note that `data_mode` and `control_mode` are intentionally separate:

- changing `data_mode` swaps the ecological source;
- changing `control_mode` changes the intervention semantics;
- real and dummy are comparable because they share one intervention semantics
  and one evaluator/method stack.

### Acceptance Checks

- A reader can answer "real vs dummy vs synthetic" from one table.
- A reader does not need to understand "Tier-2/Tier-3" vocabulary to understand
  the repo.
- A reader can answer "what does an action do to `r` and `K`?" without reading
  `envs.py`.
- The README quick-start still points to real and dummy smoke commands.
- No behavior changes.

### Naming Rule

Use characteristic names in live docs and presentation-facing text:

- discretized-state synthetic;
- continuous-state one-step synthetic;
- continuous-state cumulative-control synthetic;
- real-ecology set-point;
- dummy-ecology set-point.

Avoid "Tier-2" and "Tier-3" except in historical handoffs, old provenance
bundles, or compatibility notes that explicitly say they are old labels.

## Fix 2: Rename `real_setpoint` to `setpoint_cumulative`

### Design Decision

Use `setpoint_cumulative` as the canonical control-mode name because it
describes the semantics without implying a data source:

- action sets growth parameter `r` to an action-specific set point;
- action updates carrying capacity `K` cumulatively through the public
  accumulator.

Keep `real_setpoint` as a legacy alias for configs, old caches, and old
experiment metadata. This avoids breaking old runs while making new code and
docs read correctly.

Normalize immediately at input edges so in-memory configs and newly generated
metadata use the canonical name. Old strings should be accepted but not emitted.

### Implementation Shape

Add constants and helpers in `src/real_ecology_benchmark/config.py`:

```python
SETPOINT_CUMULATIVE = "setpoint_cumulative"
LEGACY_REAL_SETPOINT = "real_setpoint"
CONTROL_MODE_ALIASES = {LEGACY_REAL_SETPOINT: SETPOINT_CUMULATIVE}

def normalize_control_mode(mode: str) -> str:
    return CONTROL_MODE_ALIASES.get(mode, mode)

def is_setpoint_cumulative(cfg_or_mode) -> bool:
    mode = getattr(cfg_or_mode, "control_mode", cfg_or_mode)
    return normalize_control_mode(mode) == SETPOINT_CUMULATIVE
```

Then use `is_setpoint_cumulative(...)` at semantic branch sites instead of
literal comparisons to `"real_setpoint"`.

Important: do not blindly replace every literal comparison. Use the helper for
semantic control-mode logic. Treat the data-source/defaulting proxy sites below
as special cases.

### Files Likely Touched

Semantic branch sites:

- `config.py`
- `controls.py`
- `actions.py`
- `envs.py`
- `reward.py`
- `beliefs.py`
- `collector.py`
- `pipeline.py`
- `backend.py`
- `gate.py`
- `methods/moor.py`
- `methods/plus.py`

Input/enumeration sites:

- `CONTROL_MODES` in `config.py` should contain the canonical
  `setpoint_cumulative` value.
- Legacy `real_setpoint` should be accepted through normalization at input
  edges, not treated as the preferred enum.
- `ValueError` messages should mention `setpoint_cumulative` and, if helpful,
  say `real_setpoint` is accepted only as a legacy alias.

Proxy sites that need special handling:

- `config.py` `load_config` data-mode defaulting currently infers
  `data_mode="real"` when `control_mode` is set-point. Keep that behavior, but
  make it alias-robust for both `real_setpoint` and `setpoint_cumulative`.
- `pipeline.py` cache validation currently infers missing legacy
  `data_mode="real"` from recorded `control_mode="real_setpoint"`. Keep this
  compatibility path and accept both names.
- `collector.py` has a variable named `is_real` that is true for dummy too
  because it is really checking set-point semantics. Rename it to `is_setpoint`
  or similar.

Surface/config/test/docs sites:

- `configs/real_*.yaml`
- `configs/dummy_setpoint_*.yaml`
- `tests/real/`
- `tests/dummy/`
- shared tests that assert metadata/control-mode behavior
- `README.md`
- `docs/architecture.md`
- `docs/methods.md`
- `docs/experiment_protocol.md`
- `docs/reproducibility.md`
- `docs/22_6_Continuous_Observation_Experiment_Results.tex`
- `docs/22_6_Continuous_Observation_New_Baselines.tex`
- any current docs with runnable commands or current package semantics

Historical handoffs and archived provenance docs can keep the old name if they
are clearly historical.

### Backward Compatibility Requirements

- `EnvironmentConfig(control_mode="real_setpoint", data_mode="real")` still
  validates, but the resulting config should normalize to
  `control_mode="setpoint_cumulative"` if feasible.
- `load_config()` should accept old YAML with `real_setpoint`.
- Dataset/cache validation should treat old metadata
  `control_mode="real_setpoint"` as equivalent to new
  `control_mode="setpoint_cumulative"` by normalizing both the recorded and
  expected sides before comparison.
- CLI overrides and generated metadata should emit the new canonical name.
- New real and dummy configs should use `control_mode: setpoint_cumulative`.

### Tests to Add or Update

Add focused tests for the rename:

- legacy alias loads and normalizes;
- real config uses canonical `setpoint_cumulative`;
- dummy config uses canonical `setpoint_cumulative`;
- old cached metadata with `real_setpoint` still validates against a new config;
- cache validation normalizes both recorded and expected control-mode values;
- metadata round trip: legacy YAML loads, emits canonical metadata, and
  re-validates under the new canonical config;
- dummy still never reads real data loaders;
- real and dummy smoke tests still run through the same method/evaluator path.

Keep the existing full suite:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m compileall -q src/real_ecology_benchmark tests scripts
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/default.yaml
```

### Acceptance Checks

- Active source contains no semantic branch of the form
  `control_mode == "real_setpoint"` outside alias/compatibility helpers.
- The three proxy sites above are explicitly handled and commented or renamed.
- New configs and newly generated metadata use `setpoint_cumulative`.
- Old configs and old cache metadata using `real_setpoint` remain accepted.
- The real and dummy smoke commands still pass.
- The full test suite still passes.

## Fix 3: Retire "Tier" Names From Live Surfaces

This is separate from Fix 2 because it may include file names and Slurm job
names, not only a control-mode value.

### Current Live Occurrences to Rename

Live-facing occurrences found in the current tree:

- `README.md`: "Legacy Tier-2/Tier-3" wording.
- `docs/architecture.md`: "Tier-3" wording and `tier2_one_step` explanation.
- `docs/methods.md`, `docs/experiment_protocol.md`, and
  `docs/reproducibility.md`: treat as live supervisor-facing docs.
- `docs/22_6_Continuous_Observation_Experiment_Results.tex` and
  `docs/22_6_Continuous_Observation_New_Baselines.tex`: treat as live
  result/spec documents unless the user marks them historical.
- `pyproject.toml`: author label says "Tier-2 benchmark contributors".
- `tests/test_tier3_controls.py`: test file/class/local names.
- `configs/tier3_12h.yaml`: cumulative-control diagnostic config name.
- `scripts/make_tier3_12h_manifests.py`: cumulative-control manifest helper.
- `scripts/slurm/run_tier3_12h_*.sh` and
  `scripts/slurm/launch_tier3_12h_after_probe.sh`: cumulative-control Slurm
  wrappers.
- Slurm job/log/output roots such as `tier3_c0_g1g2_12h`,
  `tier3_%A_%a.out`, and `tier2-cell`.

### Proposed Replacements

| Old live name | Proposed reader-facing replacement |
| --- | --- |
| `Tier-2` | continuous-state one-step synthetic |
| `Tier-3` | continuous-state cumulative-control synthetic |
| `tier2_one_step` in prose | one-step synthetic control, keeping the code enum unchanged |
| `cumulative_capped` in prose | cumulative-control synthetic, keeping the code enum unchanged |
| `configs/tier3_12h.yaml` | `configs/cumulative_controls_12h.yaml` |
| `test_tier3_controls.py` | `test_cumulative_controls.py` |
| `make_tier3_12h_manifests.py` | preferably `make_cumulative_controls_12h_manifests.py`; if too long, keep file name and fix prose/docstrings |
| `run_tier3_12h_row.sh` and related Slurm wrappers | optional/light rename; prioritize updating hardcoded config paths and job descriptions |
| `outputs/tier3_c0_g1g2_12h` | forward-only rename to `outputs/cumulative_controls_12h`; do not migrate old artifacts |

### Compatibility Choice

Prefer to rename high-visibility files and update references in the same commit.
Do a pre-rename string sweep because several scripts hardcode `tier3_12h`; do
not rely only on the post-hoc `rg` check. If old Slurm scripts or paths are
needed for reproducibility, keep only a short compatibility note in docs rather
than duplicate wrappers.

Claude audit guidance: do not rename the code-level `tier2_one_step` or
`cumulative_capped` enum values now. They are jargon, but not actively
misleading in the same way as `real_setpoint`; renaming them would create
another alias/cache-compatibility layer for a less paper-facing surface.

Do not rename historical documents under `docs/real_ecology_history/`,
provenance snapshots under `real_ecology_runs/`, or old handoff/audit files
unless the user explicitly wants a full historical-doc rewrite.

### Acceptance Checks

- `rg -n "Tier-?2|Tier-?3|tier2|tier3"` over live docs/configs/tests/scripts
  returns only compatibility notes or historical files.
- A pre-rename sweep identifies and updates hardcoded references to any renamed
  config/script path in the same commit.
- Real, dummy, and synthetic smoke commands still pass.
- Slurm scripts pass `bash -n`.
- The full unit suite still passes.

## Suggested Commit Split

1. `Document benchmark axes and characteristic names`
   - docs-only;
   - no behavior changes.

2. `Rename setpoint cumulative control mode`
   - code/config/test rename with compatibility alias;
   - full verification required.

3. `Retire Tier names from live surfaces`
   - rename high-visibility config/test names and update references;
   - keep code-level `tier2_one_step` and `cumulative_capped` as legacy enums;
   - keep Slurm wrapper renames light unless the shorter alternatives are clear;
   - keep historical handoff/provenance files untouched.

This split makes Claude/supervisor review easier: first inspect the conceptual
framing, then inspect the semantic control-mode rename, then inspect the file
and script naming cleanup.

## Audit Questions for Claude

1. Is `setpoint_cumulative` the best replacement name, or would
   `setpoint_r_cumulative_K` be clearer despite being longer?
2. Should `EnvironmentConfig` normalize legacy aliases immediately, or should it
   preserve the raw legacy string and rely only on `is_setpoint_cumulative()`?
3. Are there any branch sites where `"real_setpoint"` still means "real data"
   rather than "set-point `r` plus cumulative `K`"?
4. Is accepting old cache metadata as equivalent safe, or should old caches be
   invalidated deliberately after the rename?
5. Confirm that `methods.md`, `experiment_protocol.md`, `reproducibility.md`,
   and the two `22_6_...tex` files are live docs that should be swept.
6. Confirm that code-level `tier2_one_step` and `cumulative_capped` should stay
   as legacy internal identifiers while reader-facing docs use characteristic
   names.
7. Which Slurm/script filenames, if any, are worth renaming versus only updating
   docstrings, job names, and default paths?
