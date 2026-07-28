# Tier-3 12-hour diagnostic experiment plan

Date: 2026-06-27

Status: **revised after Claude audit; plan only; do not run yet**.

This document supersedes the first 12h plan. Claude's audit found that the first plan would likely
produce zero method rows because it required in-band calibration and passing hard gates before the
Tier-3 collector has been recalibrated. This version changes the run into an explicit diagnostic:
measure the C8 calibration/gate gap first, then read method rows only where that is scientifically
allowed.

## Core decision

Use Claude's recommended path A:

- run with `--allow-uncalibrated`;
- set `REQUIRE_GATE=false` for row jobs;
- do not stop the matrix when C8 calibration or gates fail;
- treat calibration/gate failures as the **primary diagnostic output**, not as a surprise;
- do not claim final rankings from cells that are out of band or fail hard gates.

Path B, recalibrating collector profiles first, is real Phase-4/C8 work and should be a separate
implementation/calibration pass rather than hidden inside a 12h run.

## Resource snapshot

Checked before the original plan:

- user queue: no active Slurm jobs for `hphung`;
- current host: 128 logical CPUs;
- Slurm `gpu` partition: 40 nodes, 132 GPUs across A40/L40S/A100/T4;
- this repo is NumPy-only (`numpy`, `PyYAML`), with no Torch/CUDA path.

Conclusion: use CPU array parallelism on `comp`; do not reserve GPUs.

## Output root

Use a fresh root:

```text
discrete_action_cont_obser/outputs/tier3_c0_g1g2_12h/
```

Do not reuse `discrete_action_cont_obser/outputs/` directly.

## Required support files after audit approval

Do not launch until these are created or equivalent one-off commands are used:

1. `configs/tier3_12h.yaml`
   - copy `configs/full.yaml`;
   - set `environment.control_mode: cumulative_capped`;
   - set `evaluation.episodes_per_seed: 20` for this diagnostic pass.
   - keep `control_mode` in the YAML. Do not replace this with the CLI `--control-mode` flag,
     because `cli._config` applies kind defaults before command-line control-mode overrides.
2. A custom manifest generator.
   - produce no-raw manifests;
   - split prewarm/main;
   - re-index each CSV from `0..N-1`.
3. A Tier-3 row wrapper or direct `sbatch --wrap`.
   - pass `--config "$ROOT/configs/tier3_12h.yaml"`;
   - pass `--output-root "$ROOT/outputs/tier3_c0_g1g2_12h"`;
   - pass `--allow-uncalibrated`;
   - set `REQUIRE_GATE=false`;
   - use `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

The existing `scripts/slurm/run_manifest_row.sh` is not sufficient because it hardcodes
`configs/full.yaml`, does not forward `--output-root`, and defaults to `REQUIRE_GATE=true`.

## Hyperparameters

Use full-quality training/model/filter settings, with evaluation shortened for the diagnostic:

```yaml
environment:
  control_mode: cumulative_capped

dataset:
  transitions: 75000
  episode_length: 25

filter:
  particles: 1024
  proposal: learned
  proposal_sigma: 0.15
  context_rejuvenation: 0.025

model:
  ensemble_size: 15
  ridge: 0.001

planner:
  horizon: 5
  sequences: 64
  particles: 16
  discount: 0.95
  pessimism: 0.5

evaluation:
  seeds: [7001, 7051, 7101, 7151, 7201]
  episodes_per_seed: 20
  horizon: 50
  discount: 0.95
```

Rationale: keep the learned-filter and planner budgets intact because this run validates G1/G2 in
the real path. Use 20 episodes per seed because C8 is not calibrated yet and the first useful
answer is diagnostic, not paired-CI-quality ranking.

If the timing probe is comfortably fast and calibration is unexpectedly in band, a second run can
raise `episodes_per_seed` back to 50.

## Matrix

Cells:

- environments: `allee`, `theta`, `regime`;
- action counts: `5`, `10`;
- observation noise: `0.0`, `0.1`, `0.2`, `0.4`.

Primary rows:

- filter: `learned`;
- methods: `mopo`, `refplan`, `bamcts`, `plus`, `moor`, `delphic`, `ogsrl`.

Baseline-fidelity rows:

- `plus` with `ricker` filter;
- `moor` with `ricker` filter.

Skip for this diagnostic:

- `raw` filter ablation;
- `oracle` filters.

Row count:

```text
24 cells * 7 learned rows = 168
24 cells * 2 ricker-fidelity rows = 48
total = 216 rows
```

## Custom manifest generation

`real_ecology_benchmark.manifest.make_manifest` writes the standard synthetic
and real grids. This diagnostic uses custom CSVs with contiguous indices per
file so the prewarm and main arrays can be submitted separately.

Proposed generator logic:

```python
import csv
from pathlib import Path

root = Path("discrete_action_cont_obser/outputs/tier3_c0_g1g2_12h")
root.mkdir(parents=True, exist_ok=True)
fields = ["index", "environment", "num_actions", "sigma_obs", "method", "filter"]

envs = ("allee", "theta", "regime")
actions = (5, 10)
sigmas = (0.0, 0.1, 0.2, 0.4)
learned_methods = ("mopo", "refplan", "bamcts", "plus", "moor", "delphic", "ogsrl")

all_rows = []
for env in envs:
    for num_actions in actions:
        for sigma in sigmas:
            for method in learned_methods:
                all_rows.append({
                    "environment": env,
                    "num_actions": num_actions,
                    "sigma_obs": sigma,
                    "method": method,
                    "filter": "learned",
                })
            for method in ("plus", "moor"):
                all_rows.append({
                    "environment": env,
                    "num_actions": num_actions,
                    "sigma_obs": sigma,
                    "method": method,
                    "filter": "ricker",
                })

prewarm = [
    row for row in all_rows
    if row["method"] == "moor" and row["filter"] in {"learned", "ricker"}
]
main = [row for row in all_rows if row not in prewarm]

def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, **row})

write(root / "manifest_prewarm.csv", prewarm)
write(root / "manifest_main.csv", main)
print(len(prewarm), len(main))
```

Expected counts:

```text
manifest_prewarm.csv: 48 rows
manifest_main.csv: 168 rows
```

## Diagnostic gates

Do **not** use `scripts/run_gate_matrix.py` for this diagnostic, because it exits after hard gate
failures. Instead, run `real_ecology_benchmark.cli gate` per cell so every gate
JSON is saved even when the gate fails.

Proposed loop:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/discrete_action_cont_obser
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/outputs/tier3_c0_g1g2_12h"
mkdir -p "$ROOT/gates"

for ENV in allee theta regime; do
  for NA in 5 10; do
    for SIGMA in 0.0 0.1 0.2 0.4; do
      python -m real_ecology_benchmark.cli gate \
        --config configs/tier3_12h.yaml \
        --environment "$ENV" \
        --actions "$NA" \
        --sigma "$SIGMA" \
        --episodes 20 \
        --output "$ROOT/gates/${ENV}_${NA}a_sigma${SIGMA}.json"
    done
  done
done
```

Gate outputs are diagnostic. Failed gates should be recorded and summarized, not used to kill the
run.

## Timing probe

Before the full arrays, run one slow representative row:

```text
environment=allee, num_actions=10, sigma_obs=0.4, method=ogsrl, filter=learned
```

Use the same config, output root, `--allow-uncalibrated`, and `REQUIRE_GATE=false`.
The high-noise `sigma=0.4` cell is the conservative probe because the exact-observation shortcut
does not apply and filter cost rises with noise.

Decision rule:

- if wall time is <= 3h, proceed with `%48`;
- if wall time is 3-4h, proceed with `%32` or reduce the matrix to learned-only selected methods;
- if wall time is > 4h, do not launch the full 216 rows inside a 12h target.

## Array launch plan

Use per-task time sized to a row, not the whole experiment. Start with 4h per task.

Prewarm:

```bash
sbatch \
  --partition=comp \
  --time=04:00:00 \
  --cpus-per-task=1 \
  --mem=8G \
  --array=0-47%24 \
  <tier3-row-wrapper> \
  "$ROOT/manifest_prewarm.csv"
```

Main:

```bash
sbatch \
  --partition=comp \
  --time=04:00:00 \
  --cpus-per-task=1 \
  --mem=8G \
  --array=0-167%48 \
  <tier3-row-wrapper> \
  "$ROOT/manifest_main.csv"
```

Use `%48` by default. Drop to `%32` if the timing probe or shared filesystem behavior looks poor.
Avoid `%72` for this first Tier-3 diagnostic.

## Row wrapper requirements

The wrapper must effectively run:

```bash
python "$ROOT_REPO/scripts/run_manifest_row.py" \
  "$MANIFEST" \
  "$SLURM_ARRAY_TASK_ID" \
  --config "$ROOT_REPO/configs/tier3_12h.yaml" \
  --output-root "$ROOT_REPO/outputs/tier3_c0_g1g2_12h" \
  --allow-uncalibrated
```

and must have:

```bash
export REQUIRE_GATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
```

Do not "simplify" this by passing `--control-mode cumulative_capped` at launch while leaving the
YAML in Tier-2 mode. The scripts call `environment_with_kind_defaults(...)` before applying that
CLI override, so the YAML itself must already say `control_mode: cumulative_capped`.

## Aggregation

After all rows finish:

```bash
cd /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/discrete_action_cont_obser
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m real_ecology_benchmark.cli aggregate \
  --root outputs/tier3_c0_g1g2_12h/evaluation \
  --output outputs/tier3_c0_g1g2_12h/aggregate.json
```

Do not rank on `model_return_mean`; it is filter-mixed. Use per-filter paired rows such as
`beats_both_cells` / `beats_both_rate`, and only for scientifically admissible cells.

## Primary outputs

This diagnostic run should report:

1. Per-cell healthy-start incident collapse rate and distance from `[0.15, 0.24]`.
2. Harvest-driven delayed-collapse coverage.
3. Per-cell gate pass/fail and reward/collapse gaps.
4. Which learned-filter method rows completed without fallback spikes.
5. Return/collapse metrics only as provisional context, not final rankings.

Also record timing details from the probe and first array wave, especially whether belief-cache
reuse is working as expected. Prewarm de-risks dataset/proposal creation, but it may not remove all
offline belief-cache or dynamics-ensemble costs.

## Acceptance criteria for this diagnostic

The run is successful if:

1. unit tests pass before launch;
2. all 24 gate JSONs are produced;
3. all 24 calibration JSONs are produced;
4. all 216 planned rows either complete or have explicit logged failures;
5. learned-filter rows complete for all 24 cells and all 7 methods, unless the timing probe blocks
   the full matrix;
6. ricker-filter fidelity rows complete for `plus` and `moor`;
7. aggregate output is produced;
8. the final report clearly separates:
   - C8 calibration/gate gap;
   - provisional method behavior;
   - cells eligible for ranking, if any.

The run is **not** considered a ranking validation unless the relevant cells are in band and pass
the hard gate.

## What not to claim

- Do not claim final Tier-3 ranking success.
- Do not compare Tier-3 directly to Tier-2 as if only an implementation bug changed.
- Do not claim Phase-4 economics is accepted; G3 remains open.
- Do not rank across filter labels.
- Do not rank on `model_return_mean`.
- Do not treat `sigma=0.4` as a hard decision gate unless the gate output supports it.

## Claude audit questions for this revision

1. Is the diagnostic framing now consistent with the uncalibrated collector?
2. Is `--allow-uncalibrated` plus `REQUIRE_GATE=false` the right choice for a C8-gap measurement?
3. Is `allee_10a_sigma0.4/ogsrl/learned` the right conservative timing probe?
4. Is `%48` still acceptable after prewarm, or should the first launch use `%32`?
5. Should the first diagnostic keep all 7 learned methods, or use a selected subset before the
   full 216-row matrix?
