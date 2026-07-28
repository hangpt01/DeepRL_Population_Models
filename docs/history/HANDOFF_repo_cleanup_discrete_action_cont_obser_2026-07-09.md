# Handoff: discrete_action_cont_obser Cleanup / Keep-Remove Review

Date: 2026-07-09  
Scope checked: `discrete_action_cont_obser/` in `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`

## Short Answer

`discrete_action_cont_obser/` can run independently of `claude_build/`.

The folder is already structured as a standalone Python benchmark package for the
continuous-state / continuous-observation ecology setting.  Its top-level
`README.md` explicitly says it has no runtime dependency on the parent repository
or `claude_build`, and the code paths I checked do not import or shell into
`claude_build`.

Important nuance: the real-ecology subpackage is standalone relative to
`claude_build`, but it expects the authoritative real data table at the sibling
path:

```text
discrete_action_cont_obser/real_ecology_data/
```

So if this folder is moved into a separate repo, keep `real_ecology_data/` with it
or update `real_ecology_cont_obser/src/real_ecology_benchmark/realdata.py`.

## Evidence For Independence From claude_build

Static search inside `discrete_action_cont_obser/` found no runtime dependency on
`claude_build`.  The hits involving `claude_build` are documentation notes saying
not to edit it, or the standalone README stating there is no dependency.

Runtime code uses local package paths:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m tier2_benchmark.cli ...
```

and for the real setting:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m real_ecology_benchmark.cli ...
```

The `sys.path.insert(...)` uses in scripts point to the local `src/` directory,
not to `claude_build`.

## Current Folder Roles

### 1. Top-Level Synthetic / Tier-2-Tier-3 Benchmark

Main source:

```text
discrete_action_cont_obser/src/tier2_benchmark/
```

This is the standalone synthetic continuous-observation ecology benchmark:

- continuous latent abundance;
- noisy log-normal observations;
- 5/10 action synthetic management tables;
- Tier-2 / Tier-3 cumulative-control experiments;
- methods: MOPO, RefPlan, BA-MCTS, PLUS, MOOR, Delphic-CQL, OGSRL;
- public/private dataset schema;
- CLI, manifest, gate, evaluator, calibration scripts.

Top-level packaging exists:

```text
discrete_action_cont_obser/pyproject.toml
discrete_action_cont_obser/requirements.txt
discrete_action_cont_obser/Makefile
```

The installed console script is:

```bash
tier2
```

Without install, use:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m tier2_benchmark.cli smoke --config configs/default.yaml
PYTHONPATH=src python -m unittest discover -s tests -v
```

### 2. Real-Ecology Subpackage

Main source:

```text
discrete_action_cont_obser/real_ecology_cont_obser/src/real_ecology_benchmark/
```

This is the current real-data setting.  It is a vendored/adapted copy of the
Tier-2/3 benchmark with a different package name and real-ecology data plumbing.
It has its own:

```text
real_ecology_cont_obser/configs/
real_ecology_cont_obser/scripts/
real_ecology_cont_obser/tests/
real_ecology_cont_obser/docs/
```

It does not have its own `pyproject.toml`; run it by setting `PYTHONPATH=src` from
inside `real_ecology_cont_obser/`:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

The real package reads:

```text
discrete_action_cont_obser/real_ecology_data/
```

Current expected files there:

```text
actions.csv
species.csv
species_lambda.csv
action_effects_long.csv
cost_sources.csv
cost_anchors_portal.csv
README.md
```

### 3. Real-Ecology Run Artifacts

Large run package:

```text
discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705/
```

This contains frozen code/data snapshots, manifests, logs, outputs, analysis,
report figures, and final paper-facing artifacts from the P-safe run.  It is a
reproducibility bundle, not source needed by the live package.

Do not delete this casually if the paper/result provenance still matters.  If
space cleanup is urgent, archive it first.

## Size / Cleanup Hotspots

Approximate sizes from this checkout:

```text
23G   discrete_action_cont_obser/
12G   discrete_action_cont_obser/outputs/
11G   discrete_action_cont_obser/real_ecology_runs/
994M  discrete_action_cont_obser/real_ecology_cont_obser/
974M  discrete_action_cont_obser/real_ecology_cont_obser/outputs/
18M   discrete_action_cont_obser/real_ecology_cont_obser/log/
92K   discrete_action_cont_obser/real_ecology_data/
1.3M  discrete_action_cont_obser/real_ecology_cont_obser/src/
944K  discrete_action_cont_obser/src/
```

The source/data footprint is small.  The bulk is generated outputs, frozen run
artifacts, and caches.

## Keep / Remove Guidance

### Keep If You Want The Synthetic Benchmark

```text
pyproject.toml
requirements.txt
Makefile
README.md
configs/
src/tier2_benchmark/
tests/
scripts/
docs/architecture.md
docs/experiment_protocol.md
docs/methods.md
docs/reproducibility.md
```

Keep Tier-2/Tier-3 plan docs only if you still need historical design context.
Otherwise they can be archived to a documentation bundle.

### Keep If You Want The Real-Ecology Benchmark

```text
real_ecology_cont_obser/src/real_ecology_benchmark/
real_ecology_cont_obser/configs/
real_ecology_cont_obser/scripts/
real_ecology_cont_obser/tests/
real_ecology_cont_obser/README.md
real_ecology_data/
```

Also keep the current specs and method notes:

```text
docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex
docs/29_6_Real_Ecological_Data_Actions_and_Costs.tex
docs/29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md
real_ecology_cont_obser/docs/29_6_algorithm_method_notes.tex
```

### Archive Or Remove After Confirmation

Likely generated / removable after provenance is secured:

```text
outputs/
logs/
real_ecology_cont_obser/outputs/
real_ecology_cont_obser/log/
scripts/__pycache__/
real_ecology_cont_obser/scripts/__pycache__/
real_ecology_cont_obser/tests/__pycache__/
```

Large provenance bundle, archive before deleting:

```text
real_ecology_runs/psafe_overnight_20260705/
```

Suggested minimum if archiving that run:

```text
analysis/PAPER_RESULT_PACKAGE.md
analysis/REAL_ECOLOGY_EXPERIMENT_RESULTS.tex
analysis/REAL_ECOLOGY_EXPERIMENT_RESULTS.pdf
analysis/report_figures/
analysis/all_metrics.csv
analysis/rollup.csv
analysis/DECISION_psafe.md
analysis/P5_control_review.md
analysis/learned_vs_raw_p5.md
code/real_ecology_cont_obser/
code/real_ecology_data/
configs/real_experiment_p*.yaml
manifests/
```

## Suggested Cleanup Decision Tree

1. If the goal is to remove `claude_build/` because it is dummy-data-only, this
   should not break `discrete_action_cont_obser/`.

2. If the goal is to keep only the real-data work, keep:

   ```text
   real_ecology_cont_obser/
   real_ecology_data/
   docs/29_6_Real_*.tex
   docs/29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md
   ```

   Then decide whether to keep `src/tier2_benchmark/` as reference code.  The
   real package is currently vendored, so it does not import `src/tier2_benchmark`
   at runtime.

3. If the goal is to keep a clean standalone benchmark repo, keep both:

   ```text
   src/tier2_benchmark/
   real_ecology_cont_obser/src/real_ecology_benchmark/
   real_ecology_data/
   ```

   and remove or archive generated outputs.

4. If the goal is to split real ecology into its own repository, add packaging for
   `real_ecology_cont_obser` or move it to the repo root, and make the data path
   explicit.  The main code change would be around `realdata.DATA_DIR`.

## Commands To Re-Check Before Deleting Anything

Synthetic package:

```bash
cd discrete_action_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m tier2_benchmark.cli smoke --config configs/default.yaml
```

Real package:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

Static independence check:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models
rg -n "claude_build|from claude|import claude" discrete_action_cont_obser
```

Expected result: documentation references may appear; runtime source should not
depend on `claude_build`.

## Notes / Caveats

- I did not delete or move anything for this handoff.
- I did not run the tests or smokes in this pass; the commands above are the
  recommended confirmation step before cleanup.
- This handoff is based on the 2026-07-09 state of the
  `/home/hphung/ce25_scratch2/...` checkout.  If the `/fs04/...` checkout has
  diverged, repeat the static search and size check there before deleting large
  folders.
- The current `.gitignore` ignores generic `outputs/`, `data/`, and `*.npz`.
  That is fine for generated data, but make sure `real_ecology_data/` remains
  tracked if this becomes the authoritative real-data benchmark.
