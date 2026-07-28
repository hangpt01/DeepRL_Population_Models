# Real-ecology (Tier-4) setting

A **self-contained** subpackage that re-poses the continuous-state, noisy-observation
ecological POMDP benchmark on **real population data** with a **set-point growth
rate** and **empirically-costed actions**. It implements the E0–E10 checklist of
`../docs/29_6_Real_Ecology_Setting_Implementation_Plan.tex`.

> **Scope rule.** Everything here lives under `real_ecology_cont_obser/`. The
> original Tier-2/3 package (`../src/tier2_benchmark/`) and its data are treated as
> **read-only reference** and are not modified, moved, or deleted.

## What is reused vs. changed

The seven methods, the particle filter + proposal ladder, the ridge dynamics
ensemble, the shared particle-MPC, the unified evaluator, the seed protocol, and
the public/private schema guard are **reused with identical semantics**. Because
the Tier-2/3 package is tightly coupled by relative imports (`from ..planning import
…`), you cannot import those modules and have them transitively use an adapted
dynamics layer without editing the original package. So the package is **vendored**
(copied into `src/real_ecology_benchmark/`) and only the dynamics/data layer is
adapted in place — the copied modules' relative imports then resolve to the
adapted low-level modules, which is exactly the intended reuse.

Categories (see `git diff` against `../src/tier2_benchmark/` for the exact lines):

- **Verbatim** (unchanged): `observation`, `types`, `dataset`, `rollout`,
  `methods/base`, `methods/delphic` (Delphic reads `dataset.rewards`, so it needs
  no internal reward change).
- **Adapted dynamics/data layer**: `realdata` (new), `actions`, `config`,
  `controls`, `envs`, `beliefs`, `collector`, `pipeline`, `cli`, `manifest`.
- **Reward alignment** (state-dependent reward + two `reward_mode` settings; benefit
  input observation→predicted next state, penalty routed through `reward_mode` via
  `build_reward`): `reward`, `planning`, `dynamics` (unchanged logic; used through
  the reward path), `evaluator` (battery + `population`/`reward_mode`), `gate`, and
  the seven methods' reward construction (`mopo`, `refplan`, `bamcts`, `ogsrl`,
  `value`, `moor`, `plus`). `moor`/`plus` additionally specialise for known-r
  (fit K only; K-candidates).

The **method/MPC/evaluator control flow and structure are preserved**; the edits
route the per-population action table (`resolve_actions(cfg)`) and make the reward
state-dependent and mode-aware (`build_reward(cfg)` / benefit on `s_{t+1}`).

## The setting (`control_mode = "real_setpoint"`)

Per episode one of **9 real populations** is instantiated (`s0 = N0`, `K_ref =
K_base`, caps from `species.csv`); population identity is **known** to the agent,
only abundance `s` is partially observed and the structural stress `(C, θ, z)` is
hidden. Per step, for action `a`:

- **Set-point growth** `r_eff = clip(r_setpoint(p, a), r_min(p), r_max(p))` — the
  measured rate of the action in force (Ricker `ln λ` for ricker/allee/regime,
  LGM `λ−1` for theta). a0 → `r_base`. Realised through the existing cumulative-
  control plumbing with `accumulator_decay_r = 1` and `r_base ∈ [0,0]`, so `rho`
  carries the set-point and `r_eff = clip(rho, …)`. Negative set-points (declining
  populations, sinks) decay geometrically via the `r_pos/r_mort` split, exactly
  reproducing the measured λ.
- **Cumulative capacity** `K_eff = clip(K_base(p) + κ, K_base(p), 2·K_base(p))`,
  `κ += dK_step(p, a)`.
- **Translocation (a10)** `s += 0.10·N0(p)` before growth (the only direct-state
  action).
- **Cost** `cost_step(a) ∈ [−0.10, 1.00]` (harvest negative = revenue), portal-grounded.
- **Reward** (spec §"Reward: two state-dependent settings" / E6): benefit on the
  **true next state** `R = α·s_{t+1}/(s_{t+1}+K_ref) − cost(a) − P_m·1[crossed s_safe]`,
  `K_ref = K_base(p)`, `s_safe = 0.1·K_base(p)`. The agent still acts only on `o_t`;
  the simulator logs `R_t=R(s_t,a_t,s_{t+1})` so every method consumes identical
  `(o_t, a_t, R_t, o_{t+1})`. Two settings via `reward_mode`: **`safe`** (`P>0`,
  collapse-aware headline) and **`yield`** (`P=0`, baseline-style, matches
  PLUS/MOOR). A separate agent is trained per mode; raw returns are never compared
  across modes — both are scored on a shared reward-agnostic battery (persistence /
  final abundance, collapse probability, min abundance, fraction of steps `≤s_safe`,
  economic cost). The planner/critic benefit is the belief-expected next-state
  benefit `E_b[α·s'/(s'+K_ref)]`, consistent with the env and free of the
  `E[o|s]=s·e^{σ²/2}` survey bias.

The four dynamics families are retained. Because real `K ∈ [31, 325]`, the Allee
threshold `C` and the regime thresholds are **scaled to `K_base(p)`** (the Tier-3
absolute `C∈[90,150]`, `90/180` are nonsensical at `K=31`); θ∈[3,6] is unchanged.

`r_eff` is **not leaky** here (population identity, hence the whole λ-profile and
caps, is observed), unlike Tier-3 where it leaked `r_base`.

## Data

`../real_ecology_data/` is the authoritative in-repo table.  The benchmark reads
that single copy directly; there is no redundant package-local
`revised_cost_action_table/` folder.
`action_effects_long.csv` is the precomputed `(population, action)` lookup the env
reads; `species.csv` / `actions.csv` / `species_lambda.csv` are the relational
core. Costs are grounded in the Conservation Costing Portal (see
`../real_ecology_data/README.md`).

## Run

```bash
cd real_ecology_cont_obser
# E10 acceptance tests
PYTHONPATH=src python -m unittest discover -s tests -v
# end-to-end smoke (one population, MOPO)
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
# generate + calibrate a cell
PYTHONPATH=src python -m real_ecology_benchmark.cli calibrate \
    --config configs/real_default.yaml --population "Amur tiger" --environment ricker
# run a method / a decision-relevance gate
PYTHONPATH=src python -m real_ecology_benchmark.cli run  --config configs/real_default.yaml \
    --population "Iberian lynx" --environment theta --method mopo --filter learned
PYTHONPATH=src python -m real_ecology_benchmark.cli gate --config configs/real_default.yaml \
    --population "Amur tiger" --environment allee
# full population x family x sigma manifest
PYTHONPATH=src python -m real_ecology_benchmark.cli manifest --output outputs/real/manifest.csv
```

## E10 acceptance results

All conditions pass (`tests/test_real_ecology.py`, 16 tests — 10 E10 + 4 reward
+ 2 cache/P-safe hardening):

1. **Env reproduces `action_effects_long.csv`** for every `(population, action)`:
   set-point `r_eff`, cumulative `dK`, translocation `dN`, and `cost` all match.
2. **Abundance term = 0.5 at `s = K_base`** for every population; `r_eff` within
   caps; `K_eff ≤ K_max = 2·K_base`; capacity accumulates to the ceiling.
3. **Collapse band** (after recalibration with the harvest-tilted
   `REAL_DEFAULT_PROFILE`):
   - The **Amur tiger** — the recoverable population whose heaviest-exploitation
     rate is low enough (a2 λ≈0.62) — lands in `[0.15, 0.24]`.
   - The other recoverable populations are **structurally robust**: their measured
     heaviest-exploitation λ stays ≥0.86 (crab-eating fox a2 λ=0.987, jaguar 0.97,
     lynx 0.96, elephant ≈1.0), so a healthy population cannot be driven to
     collapse within the horizon **regardless of the behaviour mixture**. This is a
     property of the real demographics, reported per population — not a calibration
     bug (cf. the handoff: "if real stocks aren't decision-relevant, that's a
     finding").
   - The two **demographic sinks** (`r_max < 0`: Egyptian vulture, bottlenose
     dolphin) are excluded from the recoverable set and reported separately (the
     vulture collapses heavily under Do-Nothing; only translocation adds individuals).
4. **Leakage**: a generated real dataset passes `assert_public_schema`.
5. **State-dependent reward + two settings** (spec E6/E6′): benefit ≈ α/2 at
   `s_{t+1}=K_base`; reward tracks `s_{t+1}` not `o_t` (invariant to σ_obs);
   `reward_mode="yield"` ⇒ P=0 while `"safe"` ⇒ −P on a crossing; the eval battery
   (min/final abundance, economic cost, collapse, persistence) is identical across
   `reward_mode` for a fixed policy. In smoke runs the two settings visibly diverge
   — under `yield`, RefPlan/OGSRL rationally harvest the stock to collapse
   (persist=0); under `safe` they preserve it — the safety gap E6′ predicts.
6. **Post-audit hardening**: cached datasets reject `reward_mode`/reward-scale
   mismatches, safe/yield output roots are namespaced, pooled return summaries are
   keyed by `reward_mode`, `P_safe=10` exceeds the default discounted healthy
   half-benefit episode, and the runtime CSVs load from the single authoritative
   `../real_ecology_data/` table.

## Key design notes for the next chat

- A *cell* = one `(population, family)`; per-population scale lives in the config,
  filled by `config.real_environment(population, family)`.
- The single choke point for the per-population action table is
  `actions.resolve_actions(cfg)`; every copied module routes through it.
- Method science is intact but lightly adapted where set-point-r-is-known matters:
  MOOR fits only `K` (r known); PLUS candidates span `K∈[K_base,K_max]` instead of
  `r` (degenerate when r is known). Full per-method recalibration (the tex's
  C5/E9-style tuning) and the headline benchmark run are the next step.
