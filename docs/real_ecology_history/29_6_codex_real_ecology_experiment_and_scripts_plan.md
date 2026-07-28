# 29_6 Codex Plan: Real-Ecology Experiment + Runner Scripts

Date: 2026-07-03

Status: runner scripts implemented after Claude audit. No experiment jobs have
been submitted from this plan. Update after Claude runner verification: probe,
gate, and pilot are launch-ready; full-fast manifests must be sharded because the
cluster reports `MaxArraySize=1001`.

Scope for future implementation: only
`discrete_action_cont_obser/real_ecology_cont_obser/`.

Update after Claude audit: accepted the blocking PLUS timing critique. The probe
now includes `plus, learned`; pilot/full method arrays are split into PLUS and
non-PLUS manifests so walltime/concurrency can be sized from the true long pole.

2026-07-03 resource update: live Slurm checks show `comp` has `MaxTime=7-00:00:00`,
the user association has `MaxJobs=500` and `MaxSubmitJobs=1000`, and the `normal`
QOS caps the user at `cpu=250,gres/gpu=4`. The `rtq` QOS can expose
`gres/gpu=6`, but only for `2:00:00`, which is not useful for an 18h run. The
real-ecology package has no `torch`/`jax`/`cupy`/`cuda` imports in
`src/`, `scripts/`, or `configs/`, so GPUs are not requested; the 18h plan uses
up to 240 one-core CPU tasks concurrently and leaves a 10-CPU buffer under the
QOS cap.

## 1. Goal

Run the real-ecology benchmark after the state-dependent reward update:

- two separately trained reward settings: `reward_mode=safe` and
  `reward_mode=yield`;
- common reward-agnostic evaluation battery for both modes;
- cells crossed by population, dynamics family, observation noise, method, and
  filter;
- calibration/gate artifacts saved before interpreting rankings;
- CPU Slurm arrays, because this package is NumPy-only. GPUs would reserve scarce
  hardware but not accelerate the current implementation.

The first launch should be a timing/probe and diagnostic run, not the full
4608-row headline run.

## 2. Current Code Facts The Plan Relies On

- CLI entrypoint exists:
  `PYTHONPATH=src python -m real_ecology_benchmark.cli ...`.
- Existing CLI supports `smoke`, `calibrate`, `gate`, `run`, `manifest`, and
  `aggregate`.
- Default full manifest from `manifest.make_manifest` is:
  `2 reward modes x 9 populations x 4 families x 4 sigmas x
  (7 methods x 2 filters + 2 ricker ablations) = 4608 rows`.
- `run_method` writes evaluation under
  `evaluation.output_dir/reward_<mode>/<method>/<filter>`.
- Dataset rewards depend on `reward_mode`; therefore dataset cache paths must
  include `reward_mode`, or the validator must reject reuse.
- `collect_dataset` defaults to a real-ecology start-coverage profile for offline
  training, while evaluation/gate resets start deterministically at `N0`.
- `aggregate_summaries` already keys paired comparisons by
  `(reward_mode, population, family, sigma, filter, seed, method)` and reports
  sinks separately from recoverable populations.

## 3. Scripts To Add Before Running

Do not edit the existing Tier-3 scripts. Add self-contained real-ecology scripts:

1. `scripts/make_real_experiment_manifests.py`
   - Creates contiguous-index CSV manifests.
   - Emits:
     - `manifest_probe.csv`
     - `manifest_gates.csv`
     - `manifest_pilot.csv`
     - `manifest_pilot_fast.csv`
     - `manifest_pilot_plus.csv`
     - `manifest_full_learned.csv`
     - `manifest_full_learned_fast.csv`
     - `manifest_full_learned_plus.csv`
     - optional `manifest_full_all_filters.csv`
     - optional `manifest_full_all_filters_fast.csv`
     - optional `manifest_full_all_filters_plus.csv`
   - Columns:
     `index,reward_mode,population,recoverable,environment,sigma_obs,method,filter,job_kind`.

2. `scripts/shard_manifest.py`
   - Splits any oversized manifest into `<stem>_partNNN.csv`.
   - Re-indexes each shard contiguously from `0`.
   - Default shard size is `1000`, which is safe for this cluster's
     `MaxArraySize=1001` (maximum usable array index is `1000`).
   - Needed before full-fast launches:
     - `manifest_full_learned_fast.csv` (`2016` rows -> `3` shards)
     - `manifest_full_all_filters_fast.csv` (`3744` rows -> `4` shards)

3. `scripts/run_real_manifest_row.py`
   - Reads exactly one manifest row by contiguous `index`.
   - Loads a base config.
   - Rebuilds the real cell with `real_environment_like` / `real_environment`.
   - Applies row fields: population, family, `sigma_obs`, `reward_mode`.
   - Sets cell-specific paths:
     - dataset:
       `OUTPUT_ROOT/datasets/reward_<mode>/<pop_slug>/<family>/sigma_<sigma>/public.npz`
     - private:
       `OUTPUT_ROOT/private/reward_<mode>/<pop_slug>/<family>/sigma_<sigma>/truth.npz`
     - evaluation root:
       `OUTPUT_ROOT/evaluation/<pop_slug>/<family>/sigma_<sigma>`
       so `run_method` appends `reward_<mode>/<method>/<filter>`.
     - calibration:
       `OUTPUT_ROOT/calibration/reward_<mode>/<pop_slug>/<family>/sigma_<sigma>.json`
   - Runs `ensure_dataset`, writes calibration summary, then runs the requested
     method/filter.
   - Does not abort merely because collapse-band calibration fails. Real robust
     populations and sinks must be measured and reported, not forced into band.
   - Keeps reward-mode-specific datasets for the first implementation. The
     current `TrajectoryDataset` stores rewards, and at least Delphic directly
     trains on `dataset.rewards`; sharing transitions and recomputing rewards per
     mode would be a separate data-layer optimization, not part of this first
     runner.
   - Optional flags:
     `--regenerate`, `--require-gate`, `--allow-uncalibrated` defaulting to the
     diagnostic-safe behavior.

4. `scripts/run_real_gate_row.py`
   - Reads one gate row: reward mode, population, family, sigma.
   - Saves gate JSON to:
     `OUTPUT_ROOT/gates/reward_<mode>/<pop_slug>/<family>/sigma_<sigma>.json`.
   - Uses the same config reconstruction and per-cell path convention.

5. `scripts/slurm/run_real_row.sh`
   - CPU-only wrapper:
     `#SBATCH --partition=comp`, `--time=18:00:00`,
     `--cpus-per-task=1`, `--mem=8G`.
   - Exports:
     `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
     `PYTHONPATH=$ROOT/src`.
   - Accepts `CONFIG`, `OUTPUT_ROOT`, and manifest path.
   - Calls `scripts/run_real_manifest_row.py`.

6. `scripts/slurm/run_real_gate_row.sh`
   - Same CPU wrapper but calls `scripts/run_real_gate_row.py`.

7. `scripts/slurm/run_real_aggregate.sh`
   - Runs:
     `PYTHONPATH=src python -m real_ecology_benchmark.cli aggregate --root ...`.
   - Writes `aggregate.json` plus a compact text summary of:
     - completed row count;
     - missing rows by manifest;
     - beats-both rates split by `reward_mode`;
     - safe-vs-yield differences on reward-agnostic metrics;
     - calibration and gate pass/fail counts.

## 4. Configs To Add

Add two YAMLs so the runs are reproducible and not dependent on CLI overrides.

### `configs/real_probe.yaml`

Purpose: timing probe and smoke-scale diagnostic.

Recommended values:

- `dataset.transitions: 2000`
- `dataset.episode_length: 25`
- `filter.particles: 128`
- `model.ensemble_size: 3`
- `planner.horizon: 4`
- `planner.sequences: 48`
- `planner.particles: 16`
- `evaluation.seeds: [7001, 7051]`
- `evaluation.episodes_per_seed: 2`
- `evaluation.horizon: 50`

### `configs/real_experiment.yaml`

Purpose: first interpretable diagnostic/pilot.

Recommended values:

- `dataset.transitions: 4000`
- `dataset.episode_length: 25`
- `filter.particles: 256`
- `model.ensemble_size: 5`
- `planner.horizon: 5`
- `planner.sequences: 96`
- `planner.particles: 32`
- `evaluation.seeds: [7001, 7051, 7101, 7151, 7201]`
- `evaluation.episodes_per_seed: 4`
- `evaluation.horizon: 50`
- `environment.reward_mode: safe` as the base default; the row runner overrides it
  from the manifest.

This intentionally stays lighter than the spec's aspirational
`1024`-particle / `15`-ensemble setting. A later confirmation run can increase to
`filter.particles=512-1024`, `model.ensemble_size=10-15`, and
`planner.sequences=128+` after timing is known.

## 5. Manifests

### Probe Manifest

Small timing coverage, 5 rows:

1. `safe`, `Amur tiger`, `allee`, `sigma=0.4`, `bamcts`, `learned`
2. `safe`, `Amur tiger`, `regime`, `sigma=0.4`, `ogsrl`, `learned`
3. `yield`, `Egyptian vulture`, `theta`, `sigma=0.4`, `ogsrl`, `learned`
4. `safe`, `Jaguar`, `regime`, `sigma=0.4`, `mopo`, `learned`
5. `safe`, `Amur tiger`, `allee`, `sigma=0.4`, `plus`, `learned`

Rationale: includes an exploitable recoverable population, a sink, a robust
recoverable population, the high-noise setting, and methods expected to be among
the slower rows. The PLUS row is mandatory: PLUS uses 21 candidate Ricker/K
models and candidate-wise MPC, so it is the timing long pole.

### Gate Manifest

All decision-gate cells:

`2 reward modes x 9 populations x 4 families x 4 sigmas = 288 rows`.

Gate results are diagnostic, not a hard launch blocker for the first benchmark
run. They determine which cells are scientifically decision-relevant.

### Pilot Manifest

First interpretable method run:

- reward modes: `safe`, `yield`
- populations:
  `Amur tiger`, `Spotted turtle`, `Egyptian vulture`, `Bottlenose dolphin`,
  `Jaguar`, `Crab-eating fox`
- families: `ricker`, `allee`, `theta`, `regime`
- sigmas: `0.0`, `0.4`
- rows per cell:
  - all 7 methods with `learned` filter;
  - `plus` and `moor` with `ricker` filter.

Total pilot rows:

`2 x 6 x 4 x 2 x 9 = 864`.

Split for launch:

- `manifest_pilot_plus.csv`: `96 cells x 2 PLUS rows = 192 rows`
  (`plus, learned` and `plus, ricker`).
- `manifest_pilot_fast.csv`: `864 - 192 = 672 rows`.

This is big enough to test reward-mode separation, sinks vs recoverable split,
and high-noise behavior without committing to the full all-filter run.

### Full Learned Manifest

All populations/families/sigmas, still no raw filter:

`2 x 9 x 4 x 4 x 9 = 2592 rows`.

Split for launch:

- PLUS rows: `2 x 9 x 4 x 4 x 2 = 576`.
- fast rows: `2592 - 576 = 2016`.

Use this as the first serious result if the pilot is healthy.

### Full All-Filters Manifest

Matches the built-in default manifest:

`2 x 9 x 4 x 4 x 16 = 4608 rows`.

Split for launch:

- PLUS rows: `2 x 9 x 4 x 4 x 3 = 864`
  (`plus, learned`, `plus, raw`, `plus, ricker`).
- fast rows: `4608 - 864 = 3744`.

This should run only after timing, gate artifacts, and pilot aggregation look
clean. It adds the `raw` filter for every method.

## 6. Launch Order After Approval

Do not skip the probe.

1. Preflight:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

2. Create manifests:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python scripts/make_real_experiment_manifests.py \
  --output-root outputs/real_reward_modes_20260703
```

3. Submit timing probe:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
sbatch --parsable --array=0-4%5 --time=02:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_probe.csv
```

Probe decision:

- the PLUS row is the sizing row;
- for this 18h CPU-saturation plan, use up to 240 concurrent one-core tasks
  across the arrays in a phase, because the live `normal` QOS cap is `cpu=250`;
- if the PLUS probe row exceeds `12h`, do not launch full PLUS grids under the
  current hyperparameters, because two waves would risk missing the 18h target;
- if PLUS exceeds `90 min` on the light probe or any probe row fails, launch only
  the pilot and inspect the new telemetry before committing to full grids.

4. Submit gate sweep before the pilot:

```bash
sbatch --parsable --array=0-287%240 --time=04:00:00 \
  scripts/slurm/run_real_gate_row.sh \
  outputs/real_reward_modes_20260703/manifest_gates.csv
```

5. Submit pilot method arrays:

```bash
sbatch --parsable --array=0-671%180 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_pilot_fast.csv

sbatch --parsable --array=0-191%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_pilot_plus.csv
```

Together, the pilot method arrays request at most `180 + 60 = 240` concurrent
one-core tasks, below the live `cpu=250` QOS cap.

6. Aggregate:

```bash
sbatch --parsable --time=01:00:00 \
  scripts/slurm/run_real_aggregate.sh outputs/real_reward_modes_20260703
```

7. Only after pilot audit, launch either. First shard the full-fast manifest:

```bash
python scripts/shard_manifest.py \
  outputs/real_reward_modes_20260703/manifest_full_learned_fast.csv \
  --time 18:00:00 --concurrency 60
```

Then submit the printed `manifest_full_learned_fast_partNNN.csv` commands, each
with an array range no larger than `0-999`. The expected shards are
`1000/1000/16` rows:

```bash
sbatch --parsable --array=0-999%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_learned_fast_part000.csv
sbatch --parsable --array=0-999%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_learned_fast_part001.csv
sbatch --parsable --array=0-15%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_learned_fast_part002.csv

sbatch --parsable --array=0-575%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_learned_plus.csv
```

If all four learned/full arrays above are submitted together, the concurrency cap
is `60 + 60 + 60 + 60 = 240` one-core tasks.

or, if Claude agrees the learned-only run is insufficient, shard the larger
all-filter fast manifest first:

```bash
python scripts/shard_manifest.py \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_fast.csv \
  --time 18:00:00 --concurrency 45
```

Then submit the printed `manifest_full_all_filters_fast_partNNN.csv` commands.
The expected shards are `1000/1000/1000/744` rows:

```bash
sbatch --parsable --array=0-999%45 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_fast_part000.csv
sbatch --parsable --array=0-999%45 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_fast_part001.csv
sbatch --parsable --array=0-999%45 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_fast_part002.csv
sbatch --parsable --array=0-743%45 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_fast_part003.csv

sbatch --parsable --array=0-863%60 --time=18:00:00 \
  scripts/slurm/run_real_row.sh \
  outputs/real_reward_modes_20260703/manifest_full_all_filters_plus.csv
```

If all five all-filter/full arrays above are submitted together, the concurrency
cap is `45 + 45 + 45 + 45 + 60 = 240` one-core tasks.

If a later high-fidelity confirmation config raises to
`filter.particles=512-1024`, `model.ensemble_size=10-15`, and
`planner.sequences=128+`, do not assume it will fit the 18h turnaround target.
PLUS should remain in its own array, and a confirmation run may need fewer cells,
fewer PLUS filters, or a separately audited `PLUSPolicy(candidate_count=...)`
config knob.

## 7. What Counts As Success

For the pilot:

- 100% of probe rows finish and produce `summary.json`.
- At least 95% of pilot rows finish without fallback explosions or non-finite
  observations.
- Calibration summaries exist for every populated cell.
- Gate JSONs exist for every gate row.
- Aggregation reports:
  - safe and yield separated, never pooled by raw return;
  - reward-agnostic battery split by reward mode;
  - recoverable and sink populations separated;
  - beats-both statistics only on recoverable populations unless explicitly
    labelled otherwise.
- For fixed method/filter/cell, safe-vs-yield differences are interpreted via
  collapse/persistence/economic-cost metrics, not raw return.

Scientific validation target:

- In recoverable robust populations, `safe` and `yield` should be close on
  persistence/collapse.
- In sinks or Allee-stressed cells, `safe` should reduce collapse/unsafe
  occupancy or improve persistence relative to `yield`, at a measurable economic
  cost.
- Under higher `sigma_obs`, learned/structured filters should retain better
  reward-agnostic metrics than raw observation filtering in decision-relevant
  cells.

## 8. Main Risks For Claude To Audit

1. Dataset reuse across reward modes.
   - The row runner must put `reward_mode` in dataset paths.
   - The validator already rejects mismatched cached rewards, but path isolation
     avoids wasted failed jobs.

2. Output path uniqueness.
   - `run_method` appends `reward_<mode>`, but the row runner still must isolate
     population/family/sigma by `evaluation.output_dir`.

3. Calibration interpretation.
   - Do not require every recoverable population to hit `[0.15,0.24]`; several
     are structurally robust under the real action table.
   - Report calibration gaps rather than forcing biology.

4. `P_safe` interpretation.
   - `collapse_penalty=10` is a default smoke threshold, not full E9 per-cell
     calibration.
   - If the pilot shows weak safe/yield separation in sink/Allee cells, tune
     `P_safe` in a documented follow-up sweep.

5. Runtime uncertainty.
   - No large run should launch until the 5-row probe has real wall times.
   - Size walltime and concurrency from the PLUS row, not from the fast methods.
   - CPU arrays should set BLAS thread counts to 1 to avoid oversubscribing nodes.

6. Raw filter cost.
   - The full 4608-row run doubles many cells with `raw`; keep it out of the
     first pilot unless Claude specifically wants raw-filter evidence immediately.

7. PLUS walltime.
   - PLUS is the expected long pole because it evaluates 21 candidate
     Ricker/K models with MPC. Split it from fast methods in pilot/full arrays.
   - Reducing `candidate_count` is not currently a config/CLI option, so any
     candidate-count cut would require a separate code change and audit.

8. Slurm array size.
   - This cluster reports `MaxArraySize=1001`, so `--array` ranges must not
     exceed `0-1000`.
   - Probe, gates, pilot, and full PLUS arrays already fit.
   - Shard the two full-fast manifests before submitting full learned/all-filter
     runs.

## 9. Proposed Ask For Claude

Please audit this plan before Codex implements the runner scripts. In particular,
check:

- whether the row path scheme prevents reward-mode and cell cache collisions;
- whether the pilot manifest covers enough recoverable/sink and low/high-noise
  cases;
- whether the gate sweep should be run before, alongside, or after the pilot;
- whether the proposed pilot hyperparameters are acceptable as a first diagnostic
  despite being lighter than the spec's 1024-particle / 15-ensemble target;
- whether `P_safe=10` should be held fixed for the pilot or swept before method
  rankings.
