# Handoff: ecological per-timestep trajectory overlay (Ricker-only PLUS) → merge with general-RL

## 0. What the next agent must do

The general-RL side is **complete and frozen**. Your job is the **ecological** counterpart:
produce matched **per-timestep trajectories** for the ecological method, validate them by
reproducing the frozen ecological returns, then **merge** with the general side and produce the
combined **population / action / reward over-time** figures.

You are NOT computing new headline results. The trajectory work is a **report-only regeneration**
that must reproduce the frozen returns; it does not replace them. Keep everything **return-blind
and sealed** until the operator explicitly authorizes comparative inspection.

The producer code is already written, verified (except the roll itself), and gated. Most of the
remaining work is: run it over the ready cells, write a small aggregator, validate, merge, plot.

---

## 1. Two layers — do not confuse them

- **Headline / "general" results (frozen, sealed).** Aggregate registered metrics: mean discounted
  return, collapse-entry, unsafe fraction, MVP, persistence, economic cost, etc. One `summary.json`
  + per-**episode** `episodes.csv` per method-cell. No ordered time series. These exist on both sides
  and stay return-blind.
- **Trajectory diagnostic (this overlay).** Ordered per-**timestep** traces (50 steps × 20 episodes):
  state, observation, action, reward, belief, candidate weights. This is the input to the
  three-panel over-time figures (population / action / reward). It is the ONLY thing you generate.

---

## 2. The general-RL side (FROZEN — read, never rerun)

Worktree (do all general-side reads here):
```
/fs04/scratch2/ce25/general_rl_phase2_iso
```
Package:
```
real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1
```
Methods: `refplan`, `ogsrl`, `bamcts`, `ensemble_value_disagreement_pessimism`.
Design: 9 pops × 4 families {ricker,allee,theta,regime} × σ_obs{0.1,0.2} × modes{safe,yield} × 4 methods = 576 rows.
Eval: 4000 logged transitions, 160×25-step episodes, 80/20 episode split, seeds 7001/7051/7101/7151/7201 × 4 = 20 eps, 50 steps, γ=0.95, 1 CPU thread.

Authoritative general files + SHA-256:
| Object | Path (under package) | SHA-256 |
|---|---|---|
| Canonical results | `analysis/matched_results/MATCHED_GENERAL_ECOLOGICAL_RESULTS.md` | `c2b7b1afa78d62cf0978afd65e8949005faa5e838221dda34356efa2398c148c` |
| Manifest | `manifests/full_general_sigma01_02_576_rows.csv` | `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4` |
| Registration | `manifests/registration.json` | `15fd7aa1c3ddc460fd255e5d66518d49244f81cf3dbbd291a5bdfe52b376b37a` |
| Dataset reuse registry (144) | `manifests/ecological_dataset_reuse_registry_144.csv` | `474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4` |
| Frozen code | (package `code/`) | `f615d363a525422abfb983f0eee7c433b009fff81a4d0ae4925bfdaba0f3aa72` |
| Traj long table (192k rows) | `analysis/general_only_trajectory_results/general_only_trajectories_long.csv.gz` | `067a956edcc7136aa71912147311c12e08928b90ebf278323d505e3bbfd01231` |
| Traj population extrema | `analysis/general_only_trajectory_results/general_only_population_extrema.csv` | `016656ae9da8042ae61d187307adce149d0a5dff80dcda6e3cd1bf904bf38cdd` |
| Traj receipt | `analysis/general_only_trajectory_results/general_only_trajectory_receipt.json` | `b0d840cec5f06c325c0423f194a2bc6584ee65ffde36ac6afdbc0bee0a25f0ca` |
| General template generator | `analysis/capture_general_only_trajectories.py` | (this overlay is the ecological analogue of it) |

The general trajectory diagnostic covers only the **3 selected species** (Amur tiger, Puerto Rican
parrot, Egyptian vulture) × 4 families × 2σ × 2 modes = 48 cells. Your ecological trajectories must
match those keys to merge.

---

## 3. The ecological run this overlay serves

Run root:
```
/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/ricker_only_plus_72_20260720
```
| Item | Path (under run root) |
|---|---|
| Frozen code snapshot (RUN THIS CODE) | `runtime_snapshot_routing_fix/` |
| Production artifacts (READ-ONLY inputs) | `production/` |
| Config | `runtime_snapshot_routing_fix/configs/paper_faithful_hidden_ricker_only_plus_v1.yaml` |
| Plan manifest | `runtime_snapshot_routing_fix/experiments/ricker_only_plus_72/manifests/ricker_only_plus_plan_72.csv` |
| Acceptance program | `runtime_snapshot_routing_fix/scripts/run_ricker_only_plus_acceptance.py` |
| Per-cell completion receipts | `production/completion_receipts/<fit_cell>/<method>/plan_completion.json` |
| Global acceptance (when it lands) | `production/acceptance.json` |

Run identity: **72 rows = 9 pops × 4 families {ricker,allee,theta,regime} × σ_obs{0.1,0.2}**,
method **`plus_adapted_ricker_only_pbvi`** (PLUS only; NO MOOR), reward_mode **safe only**,
`expose_rk=hidden`. seed 116, collapse_penalty 5, 8 Ricker candidates (uniform prior), PBVI planner.
eval seeds/horizon/discount **identical to the general side** (7001..7201 ×4, H=50, γ=0.95).

In-flight Slurm: production array `58396952`; acceptance `58397033` (afterany). As of 2026-07-20
these were completing (62/72 cells done at last check). **No returns opened; no jobs cancelled.**

---

## 4. The producer code (already written)

```
real_ecology_runs/ricker_only_plus_72_20260720/trajectory_overlay/capture_ricker_only_plus_trajectories.py
```
SHA-256: `22b354c0942c7f5c46d45cdaf62e302d7059225d5d4ad7c0362648f12879064b`

What it does / guarantees:
- Runs **only** `plus_adapted_ricker_only_pbvi`; one manifest cell per invocation (`index` 0–71).
- Imports the **frozen** `runtime_snapshot_routing_fix/{src,scripts}` so it reproduces exactly the
  code that produced the run. Reads production datasets/fit-cache/public-surrogate.
- **No refit**: the 8-candidate bank is loaded from the frozen fit cache; reuse is asserted with the
  run's own `validate_adapted_fit_receipt_gate` + `validate_adapted_plan_cache_hit` (raises on any miss).
- **Mirrors the frozen evaluator loop exactly** (seed=block+i; `filt.reset(obs, seed+10000)`;
  `policy.reset(seed+20000)`; per step act→step→observe→`filt.update`; γ starts 1.0, ×0.95 after step).
- **Return-blind**: never opens `summary.json` / `comparative_summary.json` / `ranking.json`, never reads
  any frozen return. It reconstructs `Σ_t γ^t · reward_public[t]` **from its own captured trajectory**
  into a sealed receipt, for a LATER authorized comparison against frozen returns.
- **Per-cell gate (default)**: refuses unless THIS cell's `plan_completion.json` is present with
  `completion_status=="complete"`, `return_fields_opened is False`, matching method/index. Cells capture
  independently as they finish — NO need to wait for other cells, the global acceptance, or the general side.
  `--require-global-acceptance` adds the stricter whole-run gate; `--check` gates+rebuilds (proves
  no-refit) without rolling or writing.
- Writes only under `trajectory_overlay/results/{raw_npz,receipts}/`. Frozen snapshot + production untouched.

Verification status:
- ✅ py_compile; ✅ frozen-code imports; ✅ gate fails closed on missing/INCOMPLETE cells;
  ✅ on a live complete cell (idx 16) `--check` passes with `fit_cache_reuse_gate=passed` (8/8 keys, zero refit).
- ⚠️ The **roll itself (return reproduction) is NOT yet exercised** — it is the first step that
  generates sealed data. The build-in reconstruction + later authorized comparison is the validation.

---

## 5. Per-timestep schema (what each cell's npz stores)

Arrays shaped `(episodes=20, horizon=50)` unless noted:
```
state_pre, state_post, observation_pre, observation_post,
belief_mean, belief_low, belief_high, action,
reward_true, reward_public, danger, unsafe, mvp, terminated,
candidate_entropy, selected_candidate_index,
candidate_weights            # shape (20, 50, 8)
```
Plus a `meta` JSON blob (schema `ricker_only_plus_trajectory_v1`, `report_only=True`,
`replaces_frozen_results=False`, `sealed=True`, seeds, thresholds, reconstructed return, provenance).

Field → merge-schema mapping (matches the general long table): `state_post`→population panel;
`action`→action panel; `reward_public`→reward panel; `belief_*`→belief band; `candidate_weights`→
misspecification diagnostic. `danger` uses pre-action private state: `safety_threshold < state_pre <= 4×safety_threshold`.

---

## 6. Exact commands

Gate-only sanity check on a ready cell (no data written):
```bash
cd .../ricker_only_plus_72_20260720/trajectory_overlay
python3 capture_ricker_only_plus_trajectories.py 16 --check
```
Capture one cell (rolls 20 eps, writes sealed npz+receipt):
```bash
python3 capture_ricker_only_plus_trajectories.py 16
```
Capture all ready **selected-species** cells (23/24 ready at handoff; idx 43 pending):
```bash
for i in 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 40 41 42 44 45 46 47; do
  python3 capture_ricker_only_plus_trajectories.py "$i" || echo "skip $i (not ready)"
done
```
Enumerate currently-ready indices (return-blind):
```bash
python3 - <<'PY'
import csv; from pathlib import Path
run=Path(".."); prod=run/"production"
M=run/"runtime_snapshot_routing_fix/experiments/ricker_only_plus_72/manifests/ricker_only_plus_plan_72.csv"
for r in csv.DictReader(open(M)):
    rc=prod/"completion_receipts"/r["fit_cell"]/r["method"]/"plan_completion.json"
    if rc.is_file(): print(r["index"], r["population"], r["environment"], r["sigma_obs"])
PY
```

### Selected-species index map
| idx | species | family | σ |
|--:|---|---|--|
| 0–7 | Egyptian vulture (SINK) | ricker,allee,theta,regime × 0.1/0.2 | ready |
| 16–23 | Amur tiger (recoverable) | ricker,allee,theta,regime × 0.1/0.2 | ready |
| 40–47 | Puerto Rican parrot (recoverable) | ricker,allee,theta,regime × 0.1/0.2 | 43 (allee σ0.2) pending |

---

## 7. Remaining work (in order)

1. **Capture.** Run the producer over the ready selected-species cells (§6). Optionally all 9 pops
   (indices 0–71) for a complete run, but the 3 selected species are what the general trajectory
   diagnostic covers. Everything lands sealed in `results/`.
2. **Write the aggregator** (not yet written; ~40 lines). Concatenate `results/raw_npz/*.npz` into:
   - `ecological_only_trajectories_long.csv.gz` — long form, one row per (cell, method, episode, seed,
     timestep) using the general long-table columns (`population, population_scope, environment,
     sigma_obs, reward_mode, method, episode, seed, timestep, state_pre, state_post, observation_pre,
     observation_post, belief_mean, action, reward_true, reward_public, danger, unsafe, mvp, terminated`).
     Keep the extra columns `belief_low, belief_high, candidate_entropy, selected_candidate_index`.
   - `ecological_only_population_extrema.csv` — per cell max/min `state_post` + return-reproduction gap.
   - a coverage/validation receipt.
3. **Authorized validation** (operator go required — this opens frozen returns). For each cell compare
   the reconstructed `Σ γ^t reward_public` mean over 20 eps to the frozen `operational_return_mean` in
   `production/evaluation/.../summary.json`. Require a tiny gap (general side hit 1.1e-13). Only then unseal.
4. **Dataset-alignment check** (fairness confounder). Confirm each cell's `dataset_sha256` (recorded in
   the overlay receipt) matches the general side's `ecological_dataset_reuse_registry_144.csv`. If a cell
   diverges, flag it — the comparison for that cell is confounded.
5. **Merge & plot** (§8).

---

## 8. Merge & combined figures

Join cell-level tables on:
```
population, environment, sigma_obs, reward_mode
```
Keep `method` as the comparison axis — do NOT collapse the four general methods to a "best general".
Per-timestep tables facet on `(population, environment, sigma_obs, reward_mode, episode, seed, timestep)`,
then group by `method`. Episodes are paired by seed — retain identities for bands / paired diffs.

For each selected cell, one three-panel figure across every available method:
1. **Population over time** — mean/median `state_post` by timestep, paired-episode bands, mark the
   private safety threshold.
2. **Action over time** — categorical IDs 0–10; prefer per-method stacked action-frequency bands.
3. **Reward over time** — mean signed `reward_public` by timestep with bands; keep negative sink rewards.
Also: species contact sheets; separate safe/yield views (eco is safe-only here); **Egyptian vulture
sink shown individually**, not only pooled; population max/min tables; danger-zone action composition
over all 11 actions; collapse/unsafe/MVP/persistence comparisons; method-by-method general-vs-ecological.

---

## 9. Fairness — required framing (operator asked "is it fair?")

Fair on: identical eval protocol, hidden r/K on both sides (the fix for the earlier near-oracle
unfairness), same public reward. **Load-bearing asymmetry = model form:** this eco method fits ONLY
8 Ricker candidates but is evaluated on allee/theta/regime truth too — deliberate misspecification vs
the general methods' form-flexible surrogates. Therefore:
- Report **per-family**; do NOT pool ricker with non-ricker.
- The defensible claim is *"a form-committed ecological planner degrades off-form vs form-agnostic
  general MBRL,"* NOT *"general plans better."* On ricker cells the eco method has a legitimate
  correct-form advantage; on non-ricker cells it is expected to degrade.
- Coverage limits (not unfairness): PLUS-only (no MOOR), safe-only (no yield) → covers 24 safe cells of
  the general 48-cell diagnostic. Compare method-by-method; never write "ecological methods" plural for
  this run. (Operator's stated preference was `moor_native`+`plus_native`; the in-flight run is the
  adapted-PBVI method, so this overlay targets the running method. A MOOR-inclusive / yield-mode
  comparison is future work under separate authorization.)

---

## 10. Return-blind discipline & obsolete artifacts

- Never open `summary.json` / `comparative_summary.json` / `ranking.json` or any frozen return until
  step 7.3 is explicitly authorized. Keep `results/` sealed.
- DO NOT reuse the obsolete `real_ecology_runs/hidden_rk_comparison_20260716` MOOR/PLUS values.
- DO NOT use the cancelled `genrl-traces` workflow (`58398441`/`58398450`).
- DO NOT use `scripts/capture_trajectories.py` (synthetic-tier: abundance/action/reward only, wrong
  cells/config) — it is unrelated to this overlay.

## 11. Action semantics
0 do-nothing · 1 sustainable harvest · 2 aggressive harvest · 3 predator/disease control ·
4 breeding/recruitment support · 5 moderate restoration · 6 intensive restoration ·
7 integrated conservation (light) · 8 adaptive conservation trial · 9 flagship programme · 10 translocation.

## 12. Current stopping point
Producer written, gated (per-cell), and validated up to but not including the roll. 62/72 cells done;
23/24 selected-species cells ready (idx 43 pending). Nothing rolled yet — the roll awaits operator go
because it is the first step that generates sealed trajectory data. Next agent: capture ready cells →
write aggregator → (authorized) validate returns + dataset hashes → merge with the general side → plot.
