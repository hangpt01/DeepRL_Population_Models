# Overnight Plan — General-Method Search-Budget Audit (REV 6 — RELAUNCHED, GRID CUT)

## **[R6]** The window shrank from 6–8 h to 3–4 h. The CELL grid was cut; the CONFIG grid was NOT.

Mid-run, the available window dropped to 3–4 h. The sweep was projecting ~6.5 h, so it could not
finish. There is no more compute to buy: this account's QOS options are `normal` (250 CPU, in use),
`shortq` (280 CPU but a **30-minute** walltime — useless for 1–3 h BA-MCTS rows) and `rtq` (72 CPU).
Something had to be cut.

**What was cut: reward mode `yield` → 144 cells become 72** (safe mode only; all 9 populations,
all 4 families, both σ endpoints retained).

**What was NOT cut: any config.** All 24 remain, including BA-MCTS at `depth 20 × simulations 256`.

This follows the §9 rule rather than breaking it. The config axis *is* the experiment — the
max-budget corner is the single configuration most likely to close the gap and **falsify the
negative headline**, so cutting it under time pressure would be suppressing the evidence against my
own conclusion. The cell axis is replication: cutting it costs statistical power, not the ability to
answer the question.

**Why `yield` is the cheapest axis to lose:** in the completed motivation run, `refplan` and `bamcts`
had a beats-both rate of exactly **0.000 in *both* reward modes at every σ** — the modes agree. By
contrast σ is kept at both endpoints, because "does more search compensate for observation noise?"
is a live mechanism, whereas "does the collapse penalty change whether search helps?" is not.

**Cost after the cut:** 341 already-completed safe-mode rows are **skipped, not re-run** (81 CPU-h
banked). Remaining **548 CPU-h → ~2.3 h** at 240 cores. 227 yield rows (64 CPU-h) are sunk.

| array | rows | BS | tasks | CPU-h | cap | wall | job |
|---|---|---|---|---|---|---|---|
| `refplan` | 626 | 11 | 57 | 107 | %47 | 2.28 h | 58265235 |
| `bamcts` | 667 | 2 | 334 | 428 | %187 | 2.29 h | 58265236 |
| `ogsrl` | 94 | 8 | 12 | 13 | %6 | 2.10 h | 58265237 |
| analysis | — | — | 1 | — | — | — | 58265481 |
| **total** | **1,387** | | **404** | **548** | **240** | **≈2.3 h** | |

The pre-registered reading in §7 is unchanged. The result will be reported on 72 safe-mode cells,
and that reduced power will be stated plainly rather than hidden.

---

# (REV 5 — superseded)

**Date:** 2026-07-12
**Status:** **LAUNCHED.** Jobs `58264245` refplan / `58264246` bamcts / `58264247` ogsrl / `58264248` analysis.
Run dir `real_ecology_runs/general_audit_20260712/`. All hard gates passed (G1, G2, canary) — artifacts in `gates/`.

## **[R5]** The cost model was wrong — replaced with canary measurements

The canary ran all 24 configs on one cell before release. It falsified the rev-4 cost model
(`refplan ∝ sequences × horizon`, `bamcts ∝ simulations × depth`, `ogsrl ∝ rollout horizon`)
**in both directions**:

| method | model error | why |
|---|---|---|
| `refplan` | **1.64× under** (worst 1.79×) | per-row overhead the model ignored |
| `bamcts` | **1.60× under** (worst 1.86×) | ditto |
| `ogsrl` | **0.26× — 4× OVER** | **the scaling law itself was wrong.** OGSRL's runtime is dominated by *training*, not rollout depth: `roll24` was predicted at 3,271 s and took **636 s**. |

**Measured total: 1,478 CPU-h, not the modelled 1,069 (1.38×).** Caps, block sizes and walltimes
were recomputed from measurements, not the model:

| array | rows | BS | tasks | CPU-h | cap | wall | worst task | `--time` |
|---|---|---|---|---|---|---|---|---|
| `refplan` | 1584 | 11 | 144 | 313 | **%51** | 6.13 h | 7.78 h | 10:00:00 |
| `bamcts` | 1584 | **2** | 792 | 1,125 | **%182** | 6.18 h | 6.02 h | 08:00:00 |
| `ogsrl` | 288 | 8 | 36 | 40 | **%7** | 5.77 h | 1.90 h | 04:00:00 |
| analysis | — | — | 1 | — | — | — | — | 01:00:00 |
| **total** | **3,456** | | **973** | **1,478** | **240** | **≈6.2 h** | | under `MaxSubmit=1000` |

BA-MCTS moved `BS=3 → BS=2`: at the *measured* cost, a `BS=3` task would have been **8.6 h**, not
the modelled 5.4 h.

> **Lesson: do not size a sweep from a cost model when a canary can measure it.** The canary cost
> ~40 min and caught a 1.38× error plus a qualitatively wrong scaling law. Both would otherwise
> have surfaced as a blown window or killed tasks mid-sweep.

**Also found at submit time:** Slurm here has **`MaxArraySize = 1001`** — array *task indices* must
be ≤ 1000, a tighter constraint than `MaxSubmit` when packing finely. The main arrays (max index
791) fit; the canary initially did not, because it used raw manifest row indices (up to 1456) as
array IDs.

---

**Revision 2:** patched all five fix-before-launch points from codex's first review (**[R2]**).
**Revision 3:** fixed the Slurm concurrency blocker codex found in rev-2 — equal `%80` caps do not pool, so BA-MCTS alone implied a **10.4 h** wall, not 6 h. Caps are now cost-proportional (§5); the "drop the BA-MCTS corner if it overruns" contingency is **withdrawn** (§9). **[R3]**
**Revision 4:** fixes three blockers codex found in rev-3. (a) **Timings were stale** — rev-3 used a single contended login-node measurement, under by up to 1.8×; recomputed from 288 live rows/method, the rev-3 grid actually costs **2,139 CPU-h ≈ 8.9 h** and does **not** fit the window. (b) **BA-MCTS `--time=08:00:00` was unsafe** — a whole-cell task on the slowest cell is ~9.3 h and would have been killed. (c) **The runner had no config dimension in its output path**, so 11 configs per cell would have silently overwritten each other. Changes marked **[R4]**.
**Budget:** 6–8 h available. Honest estimate: **~5.5 h wall** — cost-proportional caps (§5), live timings (§4), and the cell grid trimmed to 144 up front (§4). The full config grid, including BA-MCTS's most generous corner, is **kept**.
**Purpose:** settle whether the motivation experiment's negative result is about *search budget* or about *the environment* — before anyone rewrites the paper.

---

## 1. Why this run exists

The motivation run ([SERVER_RESULTS](SERVER_RESULTS_motivation_native_run.md)) found native ecological baselines beating general offline MBRL in **860/864 cells (99.5%)**. Two explanations survive, and they lead to different papers:

- **(A) the general methods are under-tuned** → the headline is about search budget, not ecology;
- **(B) the general methods are model-limited** → the environment hands the natives an exact model, and the paper's premise does not hold as built.

A 3-cell spot audit (36 rows, job `58257281`, all COMPLETED) points at **(B)** — but it has a **fatal hole**, which is the reason for this run.

### What the spot audit found

| cell | native | best "tuned" general | gap |
|---|---|---|---|
| Amur tiger / ricker / σ=0.2 | 5.663 | 3.498 | **−2.17** |
| Iberian lynx / allee / σ=0.2 | 9.907 | 8.421 | **−1.49** |
| Puerto Rican parrot / regime / σ=0.4 | 5.770 | −7.538 | **−13.31** |

More planner budget did not close the gap, and for RefPlan it actively **hurt** (Amur tiger: 3.498 at h=5 → 2.737 at h=10 → **1.509** at h=20) — the signature of a worse model compounding over longer rollouts.

### THE HOLE — the knob only reached one of three methods

| method | search-depth knob | reads `planner.horizon`? | evidence |
|---|---|---|---|
| `refplan` | `ParticleMPC` → [planning.py:43](../../src/real_ecology_benchmark/planning.py#L43) | **YES** | returns changed across 5/10/20 |
| `bamcts` | `depth: int = 5` — constructor kwarg ([bamcts.py:28](../../src/real_ecology_benchmark/methods/bamcts.py#L28)) | **NO** | returns **bit-identical** at h=5/10/20 |
| `ogsrl` | `_rollouts(..., horizon: int = 6)` — hardcoded ([ogsrl.py:191](../../src/real_ecology_benchmark/methods/ogsrl.py#L191)) | **NO** | never varied |

"We tuned the general baselines and it didn't help" is therefore **vacuous for two of three**. A negative result cannot rest on a sweep that was a no-op for BA-MCTS and OGSRL.

---

## 2. **[R2]** The code change — corrected implementation

Codex is right that `build_method` constructs policies as `METHODS[method](cfg.environment, cfg.model, cfg.planner, seed=...)` ([pipeline.py:228](../../src/real_ecology_benchmark/pipeline.py#L228)) with **no kwargs pass-through**. So the new settings must be read from `self.planner_cfg` *inside* the policies. And the existing tests pass `simulations=16, depth=3` as **direct kwargs** ([test_methods.py:26,63](../../tests/synthetic/test_methods.py#L26)), so the constructors must still accept them.

**Design: kwarg wins if given, else fall back to `planner_cfg`.** This keeps the direct-kwarg tests working *and* makes the knob reachable from config.

```python
# config.py :: PlannerConfig  -- defaults are the CURRENT LITERALS, so this is a no-op
    horizon: int = 5                  # (existing) refplan / ParticleMPC rollout length
    sequences: int = 96               # (existing) refplan / MPC candidate sequences
    particles: int = 32               # (existing)
    discount: float = 0.95
    pessimism: float = 0.5
    bamcts_depth: int = 5             # NEW -- was the literal in BAMCTSPolicy(depth=5)
    bamcts_simulations: int = 128     # NEW -- was the literal in BAMCTSPolicy(simulations=128)
    ogsrl_rollout_horizon: int = 6    # NEW -- was the literal in OGSRL._rollouts(horizon=6)

# bamcts.py
    def __init__(self, *args, simulations: int | None = None, depth: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = (
            self.planner_cfg.bamcts_simulations if simulations is None else simulations
        )
        self.depth = self.planner_cfg.bamcts_depth if depth is None else depth

# ogsrl.py :: _rollouts -- horizon defaults to the config, not the literal 6
    def _rollouts(self, start_states, start_rho=None, start_kappa=None, horizon: int | None = None):
        horizon = self.planner_cfg.ogsrl_rollout_horizon if horizon is None else horizon
```

**Why new fields and not `planner.horizon`:** BA-MCTS's depth default (5) happens to equal `planner.horizon` (5), but **OGSRL's rollout default is 6, not 5**. Repointing OGSRL at `planner.horizon` would silently change 6 → 5 and *retroactively invalidate the completed motivation run*. Separate fields carrying the current literals make the change provably inert.

### Gate G1 — no-op proof (hard stop)
1. `make test` green, `make docs-check` green, direct-kwarg tests still pass.
2. Re-run `bamcts` **and** `ogsrl` at defaults on a completed cell and assert `operational_return_mean` is **bit-identical** to the finished motivation run. **Not "close" — identical.** If it is not, the change is not a no-op and the run does **not** proceed.

---

## 3. **[R2]** Dataset reuse — explicit, with a hash gate

Codex is right that this was hand-waved. `run_real_manifest_row.py` rewrites dataset paths under whatever `--output-root` it is given ([line 59](../../scripts/run_real_manifest_row.py#L59)), and the spot-audit script generated *fresh* datasets under `outputs/audit_general`. That answers a nearby question, not the one we need.

**Implementation:** add `--dataset-root` to the runner (defaulting to `--output-root`, so every existing caller is unchanged). The audit passes:

```
--dataset-root real_ecology_runs/motivation_native_20260711/outputs/main
--output-root  real_ecology_runs/general_audit_20260712/outputs
```

so `cfg.dataset.output` points at the **motivation run's existing `public.npz`**. `ensure_dataset` finds it, `_validate_dataset_cell` checks it matches the cell, and **nothing is regenerated**.

**Gate G2 — dataset identity (hard stop).** Before the sweep, assert for all **144** audit cells that the `dataset_sha256` equals the motivation run's for the same cell. Any mismatch → **STOP**. (Datasets are deterministic given `(environment, seed=116)`, so this should hold trivially — which is exactly why a violation would mean something is wrong.)

**Consequence:** because no dataset is generated, there is **no `ensure_dataset` lock contention**, so the block size no longer has to align to cell boundaries. That frees the packing to be chosen for load-balance instead (see §5).

---

## 4. **[R2]** The sweep — widened so "we tuned them" is defensible

Codex is right that depth+pessimism alone does not earn the sentence "we tuned the general methods." Each method now gets a genuine **search-budget** axis alongside depth.

### The config grid — **kept whole**, including the most generous corner

| method | axes | configs | new (as-run already in the main run) |
|---|---|---|---|
| `refplan` | `horizon` {5,10,20} × `sequences` {96,256} × `pessimism` {0.5,0.0} | 12 | **11** |
| `bamcts` | `bamcts_depth` {5,10,20} × `bamcts_simulations` {128,256} × `pessimism` {0.5,0.0} | 12 | **11** |
| `ogsrl` | `ogsrl_rollout_horizon` {6,12,24} | 3 | **2** |

The as-run baselines are reused from the completed motivation run — so the comparison is against exactly what was published, not a re-run of it.

### **[R4]** The cell grid — trimmed to **144 cells**, up front, with the reason

Rev-3 used all 288 cells. **At live timings that grid costs 2,139 CPU-h ≈ 8.9 h wall — it does not fit the 6–8 h window** (§ cost, below). Something has to give, and *which* thing gives is a scientific decision, not a scheduling one:

- **The config axis IS the experiment.** It is what answers "does more search budget close the gap?" The most informative points are the *extremes* — BA-MCTS at (depth 20 × 256 sims) is the single configuration most likely to close the gap and falsify my own headline. Cutting it would gut the run (and §9 forbids it).
- **The cell axis is replication.** Cutting it costs statistical power, not the ability to answer the question.

**So the cells are trimmed: σ ∈ {0.0, 0.4} only** (drop σ = 0.1, 0.2) → **144 cells** = 9 populations × 4 families × **2 σ** × 2 reward modes.

Justified directly from the completed motivation run — the general-vs-native gap across σ is:

| σ | gap | |
|---|---|---|
| 0.0 | −0.698 | **kept** (endpoint) |
| 0.1 | −1.119 | dropped — interpolates |
| 0.2 | −1.118 | dropped — interpolates |
| 0.4 | −1.210 | **kept** (endpoint) |

σ=0.1 and σ=0.2 are within 0.001 of each other and sit between the retained endpoints. They carry essentially no information the endpoints do not. Keeping both extremes preserves every claim of the form "more search budget does/doesn't help, at low *and* high observation noise" — while keeping all 9 populations, all 4 families, and both reward modes.

**Rows:** 144 × (11 + 11 + 2) = **3,456**.

### **[R4]** Cost — recomputed from **live** fleet timings (rev-3's numbers were stale)

**Rev-3 used per-row timings from a single contended login-node measurement. They were wrong by up to 1.8×.** Recomputed from the **288 completed rows per method** in the motivation run (`row_seconds` in every `summary.json`):

| method | rev-3 assumed | **live mean** | live p95 | live max |
|---|---|---|---|---|
| `refplan` | 53 s | **96.5 s** (1.8× under) | 132 s | 152 s |
| `bamcts` | 254 s | **430.4 s** (1.7× under) | 622 s | 815 s |
| `ogsrl` | 596 s | **705.0 s** (1.2× under) | 1063 s | 1100 s |

MCTS cost scales ≈ `simulations × depth`; MPC cost ≈ `sequences × horizon`. Summing the cost factors of the *new* configs (the as-run config is reused, not re-run) gives Σfactor = 50.3 (refplan), 41.0 (bamcts), 6.0 (ogsrl).

| method | rows | Σ cost/cell | CPU-h @ 144 cells | *(@ 288 cells — rev-3)* |
|---|---|---|---|---|
| `refplan` | 1584 | 4,857 s | **194** | *(389)* |
| `bamcts` | 1584 | 17,646 s | **706** | *(1,412)* |
| `ogsrl` | 288 | 4,230 s | **169** | *(338)* |
| **total** | **3,456** | | **~1,069 CPU-h** | *(**2,139** → **8.9 h**, over budget)* |

At the 240-CPU working quota the **ideal wall is 4.46 h**. The realised wall is bounded below by the longest single task (§5) at ~5.4 h. **Expected ≈ 5.5 h, inside a 6–8 h window.**

---

## 5. **[R3]** Slurm shape — **concurrency caps are proportional to cost, not equal**

Because §3 removes dataset generation, packing is chosen for **load balance**, not cell alignment. **One array per method**, so each gets an appropriate walltime and the slow BA-MCTS rows cannot strand a task holding cheap RefPlan rows.

### The rev-2 blocker (codex, correct — this would have blown the window)

Rev-2 said `--array=0-287%80` on each of the three arrays ("3 × 80 = 240, under the cap") and claimed ~6 h wall. **That arithmetic is wrong.** A per-array `%` cap is a *reservation*, not a share: when RefPlan and OGSRL finish early, BA-MCTS **cannot borrow their CPUs**. So the wall is set by the slowest array in isolation:

| array | CPU-h | at `%80` | |
|---|---|---|---|
| `refplan` | 214 | 2.67 h | finishes, then idles 40+ cores |
| `ogsrl` | 288 | 3.60 h | finishes, then idles its cores |
| `bamcts` | **833** | **10.41 h** | ← **the real wall.** Blows a 6–8 h window. |

### The fix: size each cap to its array's cost so all three land together

Allocate the 240 concurrent CPUs in proportion to each array's CPU-hours (`cap_i ≈ 240 · CPU-h_i / 1069`):

| array | rows | BS | tasks | CPU-h | **cap** | mean h/task | **worst h/task** | **wall** | `--time` |
|---|---|---|---|---|---|---|---|---|---|
| `refplan` | 1584 | 11 (one cell) | 144 | 194 | **%44** | 1.35 | 2.12 | **4.4 h** | 04:00:00 |
| `bamcts` | 1584 | **3** | 528 | 706 | **%158** | 1.34 | **5.43** | **≈5.4 h** | **12:00:00** |
| `ogsrl` | 288 | 2 (one cell) | 144 | 169 | **%38** | 1.17 | 1.83 | **4.5 h** | 04:00:00 |
| analysis | — | — | 1 | — | — | — | — | — | 01:00:00 |
| **total** | **3,456** | | **817 tasks** | **1,069** | **240** | | | **≈ 5.5 h** | under `MaxSubmit=1000` |

### **[R4]** BA-MCTS walltime and packing — codex's second blocker

Rev-3 gave BA-MCTS `--time=08:00:00` with `BS=11` (one whole cell per task). **That is unsafe.** A cell is all 11 new configs, Σfactor = 41; on the slowest observed cell (`bamcts` max row = 815 s) a whole-cell task costs `41 × 815 s ≈ 9.3 h` — **it would be killed at the 8 h wall**, silently losing that cell.

Two changes:
1. **`BS = 3` for BA-MCTS** (not a whole cell). Worst single row is the `(depth 20 × 256 sims)` config on the slowest cell = `8 × 815 s = 1.81 h`, so a worst-case 3-row task is **5.43 h**.
2. **`--time=12:00:00`** for the BA-MCTS array — **2.2× margin** over that worst case. Walltime is free when unused; a timeout is not.

**Manifest ordering: config-major, most-expensive-config first.** Two reasons:
- a task's rows then share one config, so its cost is *predictable* rather than a lottery over the cheap and expensive corners;
- longest-processing-time-first means the 5.4 h BA-MCTS tasks start in wave 1 and the short ones backfill, instead of a 5.4 h task starting last and adding itself to the tail.

- Partition `comp`, **CPU-only** — no method here has a CuPy path; requesting GPU buys nothing and queues behind fair-share.
- `--cpus-per-task=1`, `--mem=8G`, `OMP_NUM_THREADS=1`.
- Frozen code snapshot + `PROVENANCE.txt`; canary first; then release.

> **Two general lessons, both learned the hard way here:**
> 1. Per-array `%N` caps **do not pool**. If arrays of very different cost share one quota, size each cap to its own cost, or the cheap array will sit on CPUs the expensive one needs.
> 2. Size `--time` against the **worst task**, not the mean. A mean-sized walltime silently kills the tail — and the tail is exactly the expensive corner you most needed.

---

## 5b. **[R4]** Manifest / runner contract — codex's third blocker (a silent-overwrite bug)

Rev-3 never specified how a tuning config reaches the runner. As it stands **it cannot**, and worse, it would *silently destroy data*:

`run_real_manifest_row.py::apply_row_config` ([lines 43–62](../../scripts/run_real_manifest_row.py#L43-L62)) builds the output path from **cell + method + filter only**:

```python
eval_cell = Path(slug(population)) / family / f"sigma_{sigma_slug(sigma)}"
cfg.evaluation.output_dir = str(root / "evaluation" / eval_cell)
# run_method then appends .../reward_{mode}/{method}/{filter}
```

There is **no config dimension in the path**. So all 11 `refplan` configs for one cell would write to the *same* directory and **overwrite each other** — the run would complete "successfully" and leave one arbitrary config's results per cell. It also ignores any tuning columns entirely, so every row would silently run at defaults.

### Manifest schema (new columns)

| column | applies to | blank means |
|---|---|---|
| `config_tag` | **all rows** | — (required, unique per `(cell, method)`) |
| `horizon`, `sequences`, `pessimism` | `refplan` | use config default |
| `bamcts_depth`, `bamcts_simulations`, `pessimism` | `bamcts` | use config default |
| `ogsrl_rollout_horizon` | `ogsrl` | use config default |

`config_tag` is a deterministic slug of the overrides, e.g. `h20_seq256_pess0p0`, `d20_sims256_pess0p5`, `roll24`.

### Runner changes (`run_real_manifest_row.py`)

```python
# 1. dataset reuse (§3): read the motivation run's datasets, never regenerate
parser.add_argument("--dataset-root", default=None)   # default = --output-root -> all existing callers unchanged
dataset_root = Path(args.dataset_root or args.output_root)
cfg.dataset.output         = str(dataset_root / "datasets" / cell / "public.npz")
cfg.dataset.private_output = str(dataset_root / "private"  / cell / "truth.npz")

# 2. tuning overrides -> planner config
overrides = {k: cast(row[k]) for k in TUNING_COLUMNS if row.get(k) not in (None, "")}
cfg.planner = replace(cfg.planner, **overrides)

# 3. UNIQUE OUTPUT PATH -- the fix for the overwrite bug
tag = row["config_tag"]
cfg.evaluation.output_dir = str(root / "evaluation" / eval_cell / tag)
#   -> .../evaluation/<pop>/<family>/<sigma>/<config_tag>/reward_<mode>/<method>/<filter>/summary.json
```

### Gates

- **Manifest generation asserts `config_tag` is unique per `(cell, method)`** — a collision is a hard error at generation time, not a silent overwrite at run time.
- **Post-run assert:** `summary_count == 3456`. If the overwrite bug were still live, the count would come back short — this is the cheap detector for it.
- Every summary records its `config_tag` and the resolved planner fields, so the analysis can never mis-attribute a row to the wrong config.

---

## 6. **[R2]** Model quality — three columns, honestly labelled

Codex is right that the current script calls `predict_ricker_next` even on Allee/regime cells, which makes it a **MOOR/Ricker diagnostic**, not "the native model". The full-grid report will carry **three** separate columns, clearly named:

| column | what it is |
|---|---|
| `learned_log_rmse` | the model the **general methods** learn from 4000 transitions (`LearnedLinearProposal`) |
| `ricker_native_log_rmse` | the **Ricker** mechanistic model — what `moor_native` commits to (misspecified off-Ricker) |
| `true_family_log_rmse` | the mechanistic model of the **cell's true family** — the upper bound a form-correct solver could reach, and what `plus_native`'s candidate bank contains |

The claim to be tested is specifically: *does reading exact `r`/`K` from the public tables buy a materially better model than learning it from data* — and, separately, *how much does Ricker misspecification cost*. Those are two different numbers and were previously conflated into one.

---

## 7. Pre-registered reading (fixed BEFORE seeing numbers)

- **Best-tuned general closes the gap on a material fraction of cells** → explanation **(A)**. The motivation negative result is **withdrawn**; the baselines must be re-run at the tuned settings.
- **Gap persists at every depth/budget for all three methods** → explanation **(B)** confirmed on the full grid with all three baselines genuinely tuned. The negative result stands and the environment-design question becomes unavoidable.
- **More budget makes the generals *worse*** (as RefPlan already does) → positive evidence for **(B)**: model error compounding. Report alongside the three-column model-quality table.

**I will not re-tune until the numbers come out the desired way.** The grid above is fixed in advance and reported whatever it shows.

---

## 8. Launch sequence (only after sign-off)

```
1. PlannerConfig + bamcts + ogsrl change      (§2; kwarg-wins-else-config)
2. make test && make docs-check               (gate)
3. G1 no-op proof: bamcts & ogsrl bit-identical to the motivation run   (HARD STOP)
4. add --dataset-root to run_real_manifest_row.py (default = --output-root: no caller changes)
5. freeze snapshot + generate manifest        (**3456 rows**; assert 144 cells x 24 configs;
                                              assert config_tag unique per (cell, method)  -- §5b)
6. G2 dataset-hash gate: all 144 cells match the motivation run's sha256 (HARD STOP)
7. canary: one cell, verify **24 summaries in 24 DISTINCT config_tag dirs** + hash match  -- §5b
8. release: 3 arrays on comp, **caps %44 / %158 / %38** (refplan / bamcts / ogsrl),
            BS 11 / 3 / 2, --time 04:00 / **12:00** / 04:00, config-major expensive-first  -- §5
9. full-grid model quality (3 columns, §6)
10. assert summary_count == 3456 (the overwrite detector, §5b), then analyse against §7
```

## 9. **[R3]** Resolved: the grid is pre-registered and stays whole

Rev-2 asked whether, if BA-MCTS at (depth 20 × 256 simulations) overruns, the right move is to **drop that corner** mid-run.

**No — and codex was right to push back.** Two reasons:

1. **It is no longer necessary.** The overrun was an artifact of the broken `%80` caps (§5). With proportional caps the whole grid, BA-MCTS corner included, lands at **≈5.5 h** (§4/§5). There is nothing to drop.
2. **It would be results-shopping even if it were necessary.** (depth 20 × 256 sims) is the *most generous* configuration BA-MCTS gets — the single cell most likely to close the gap against the natives. Dropping it reactively, after seeing how the run is going, silently removes the strongest evidence *against* my own headline. A pre-registered grid that gets trimmed under time pressure is not pre-registered.

**Rule for this run:** the grid in §4 is fixed before launch and reported in full, whatever it shows. If the wall time turns out worse than modelled, the correct response is to **extend the window or raise the caps — never to quietly delete the corner that was most likely to falsify the result.** If the grid must shrink, it shrinks *in this document, before submission*, with the reason written down.
