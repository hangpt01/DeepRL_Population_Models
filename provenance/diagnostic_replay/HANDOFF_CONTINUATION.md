# Handoff: continue the P=10 diagnostic-replay work

> **Completion note (2026-07-27 22:09:55 AEST):** The work described below is complete. Tier A
> job `58569801` and Tier B job `58577745` passed every requested seven-field parity check exactly
> (`max_abs_diff=0.0`, `recomputed_fits=0`). Use `DIAGNOSTIC_REPLAY_RECEIPT.json` and `SUMMARY.md`
> as the final record; the remaining text is retained as the pre-run handoff/audit trail.

Date: 2026-07-27 ~14:30 AEST. Read this + `DIAGNOSTIC_REPLAY_RECEIPT.json` + `SUMMARY.md` + `manifest.json`
in this same root. This is a **REPLAY of accepted policies with logging** — it supersedes nothing and
re-ranks nothing.

## 0. What to do next (in order)
1. **Wait for the Tier-A array `58569801` to finish** (~4 h from 14:20 AEST). Then verify all 10 cells'
   7-field parity gate (script below). PASS ≤1e-9. **Any INVESTIGATE/FAIL is instrumentation, not
   architecture** (the same-hardware MOOR baseline passed 0.0) — halt that cell, diagnose, do NOT
   interpret its logs.
2. **Build Tier-B logging** (RefPlan/OGSRL/BA-MCTS/EVD) — the main remaining implementation (§5).
   Must NOT gate the Tier-A fleet. Then launch B1–B3 and parity-check them.
3. **Write the extended receipt/SUMMARY** once all cells pass.

Per-cell parity check (run when the array is done):
```bash
cd .../three_species_diagnostic_replay_20260726_v1
python3 - <<'PY'
import glob,json
for f in sorted(glob.glob("parity/PARITY_*diagnostic_replay.json")):
    d=json.load(open(f)); p=d["parity"]; print(d["cell"],d["side"],p["verdict"],p["max_abs_diff"])
PY
```
Flagged diagnostics to report (operator predictions): **A2** M2 switch_vs_MAP (predicted >0, harmful) + M5 centred/raw; **A1** M4 margin mass below 1e-3; **A5** return_sd (predicted 0.000 despite per-episode θ redraw). Per accepted cell log M1/M2/M3/M5/M7/M13/M15.

## 1. Immutable constraints (never violate)
- Accepted `real_ecology_runs/three_species_ecological_p10_correction_20260723_v1/analysis/MATCHED_P10_144_METHOD_CELLS.csv` (sha `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`) and its receipt: **untouched**.
- `recomputed_fits` must be 0 (reuse the 216 demographic fit-cache slots by hash) — applies to PLUS/MOOR. (General methods fit their own cheap public ensemble at build; that's not a demographic-slot recompute.)
- **Return-blindness NOT claimed** (parity gate uses accepted returns as the invariant).
- **threads=1** (`OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=1`) and **`--constraint=xenon-8452Y`** are parity-critical: m3h also has AMD `EPYC9454` nodes whose BLAS kernel breaks 1e-9 parity.
- Side-effect-free logging (rule 2): copy already-computed arrays; never recompute in a way that alters the decision path or RNG order. The shadow rollout must assert env-state + all RNG bit-generator states unchanged (criterion 7).
- Constant-action rows (S1) are **evaluator-only reference policies** — never a seventh method; never change any method/reward/hyperparameter in response to any log.

## 2. Current live state
- **Tier-A array `58569801`** (10 tasks = A1–A5 × {plus,moor}) submitted, **RUNNING** as of 14:28 (started ~14:20 after a long queue on m3h behind GPU jobs). ETA: MOOR cells ~30 min, PLUS ~4.25 h. Only `A6 moor`/`A6 plus` parity files exist (checkpoint). Array script: `scripts/replay_array.sbatch` (PAIRS index→cell/side map).
- No waiter is currently armed (previous session torn down). Re-arm one, or poll `squeue -j 58569801`.

## 3. Completed and PASSED (parity exact 0.0, all 7 fields)
- **A6-MOOR** on login 6548Y+ (pilot) AND on 8452Y (`58548481`) — arch baseline, both 0.0.
- **A6-PLUS** on 8452Y (`58548480`, capture-path code) — 0.0. `recomputed_fits=0`.
- A6 signature matched predictions: M7=1.000, M3 unanimity=1.00, M2=0; **M1 posterior concentrates (TV 0.83, ‖Δw‖₁ 1.75) while M2=0** ("inference succeeds, decisions unaffected"); M5 raw 0.905 vs centred 0.006; M13 surrogate over-prices reward +4.70 over all visited unsafe states.
- **S1 constant-action sweep** (`derived/S1_constant_action_sweep.csv`, 264 rows) + **a0 baseline** done. Findings in §7.
- **CANCELLED:** job `58540799` (first A6-PLUS on the ~2× recompute path, rule-2 non-compliant) — superseded by `58548480`.

## 4. Harness design (`scripts/run_diagnostic_replay.py`, sha in receipt)
Imports the FROZEN `ricker_only_plus_72_20260720/runtime_snapshot_routing_fix` code. Per cell:
rebuild the exact accepted policy from the frozen fit-cache (no refit; asserted via `validate_adapted_fit_receipt_gate`+`validate_adapted_plan_cache_hit`), then run the **unmodified** `ContinuousEvaluator.run`. Logging is non-invasive: (a) env `__class__`-swap wrapping `super().step()` + an 11-action shadow via pure `transition_value` (asserts state/RNG unchanged); (b) policy `act` wrapper copying `last_diagnostics`. **PLUS q_cand is CAPTURED** from the frozen act()'s own `action_values` calls (wrapped in `run_cell`), not recomputed. Outputs `.npz` + `.scalars.csv.gz` (no pyarrow in venv). Summarizes the 7 fields and compares to the accepted CSV row.

## 5. Tier-B build spec (MAIN REMAINING WORK)
Tier-B cells: **B1 fox/regime/0.2 (=A3), B2 tiger/ricker/0.1 (EVD collapse cell), B3 tiger/allee/0.2 (OGSRL-wins)**; four general methods each (refplan, ogsrl, bamcts, ensemble_value_disagreement_pessimism). ~1.1 core-h total.

**CRITICAL — the general methods do NOT match the current harness path:**
- **Different code.** The general methods (`refplan.py, ogsrl.py, bamcts.py, ensemble_value_disagreement.py, public_models.py`) in the ricker_only snapshot **DIFFER by hash** from the general run's code. The accepted general returns were produced by the general run's code, so Tier-B replay **must import the general run's code**:
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/code/src`
- **Different config / data / surrogate**: config `code/configs/general_phase2e_full_sigma01_02.yaml`; datasets + public surrogate under `general…/quarantine/datasets/regime_hidden/reward_safe/<species>/<family>/sigma_<s>/` (the surrogate npz `public.regime_hidden.public_surrogate.v2.seed20116.npz` is present). **Note the safe-mode cells are under `reward_safe/`** (there is also `reward_yield/`); the P=10 matched table uses safe.
- **Different build path**: general methods use `run_method`'s NON-faithful branch — `split_train_holdout(dataset, cache, cfg)` (80/20) + a public dynamics ensemble (5 ridge-linear bootstrap members) fit at build (deterministic; this is NOT a demographic-slot recompute) + `ContinuousEvaluator.run`. Filter is `learned`, not `faithful_internal`. Rebuild = load config+dataset+surrogate → `_hidden_method_context` → `make_filter_factory(...,"learned",ctx)` → `build_method(method, train, factory, train_cache, holdout_dataset=…, holdout_cache=…, split_info=…, method_context=ctx)`.
- **RNG**: BA-MCTS consumes RNG in `act` (256 sims); the unmodified evaluator + `policy.reset(seed+20000)` makes it deterministic. Logging (reading `last_diagnostics`) must not consume RNG. EVD/refplan/ogsrl: check before wrapping.

**Per-method log schema + exposed `last_diagnostics` keys (already reconnoitred):**
- **RefPlan** (`refplan.py:97 plan_marginalized`, hidden path): keys `action_scores, model_entropy, posterior_max, public_extinction_risk, return_std`. Log the **5-member posterior `w_m`** (from `policy.posterior`, 5 members), best-sequence posterior-weighted mean return + return SD, argmax under λ_ref=0; belief columns VALID (uses σ_o).
- **OGSRL** (`ogsrl.py`, offline actor): keys include `action_probabilities, guardian_threshold, guardian_override, behavior_normalized_cost, cost_horizon…, actor_norm_*`. Log predicted C_25 for chosen action, safety budget, slack, **both duals**, OOD/guardian flag, fallback; NO belief columns (σ_o unused). **M12**: fraction of steps constraint binds; mean slack; s_low(35.9/31.9/22.9) vs true s_safe(25/10.25/81.25).
- **BA-MCTS** (`bamcts.py:267/295`): keys `tree_nodes, root_q, simulations, members`. Log root visit counts per action, max depth, node count, root MODEL belief; NO state-belief columns.
- **EVD** (`ensemble_value_disagreement.py:189`): keys `q_ensemble_mean, q_ensemble_disagreement*, pessimistic_q_score, q_ensemble_members…`. Log Qbar[a] (11), Var_m Q[a] (11), argmax under λ_V=0; NO belief cols; **NO surrogate_reward_t** (EVD trains on raw `dataset.rewards`, not the surrogate — confirmed). **M11**: switch frac between argmax(Qbar−0.1·Var) and argmax(Qbar); on B2 rank the recovery actions under Qbar.
- **M14** (all three of OGSRL/BA-MCTS/EVD): record explicitly that no belief/filtering columns exist — do not present them beside PLUS/MOOR/RefPlan filtering diagnostics.

Parity target for each Tier-B cell/method = the accepted MATCHED_P10_144 row (which came from the general run). Verify the general dataset hash matches the accepted `dataset_sha256` before trusting a cell.

## 6. Key paths / jobs / hashes
- Replay root: `real_ecology_runs/three_species_diagnostic_replay_20260726_v1/` (scripts, logs, derived, parity, receipt, manifest, SUMMARY).
- Frozen faithful code (PLUS/MOOR): `ricker_only_plus_72_20260720/runtime_snapshot_routing_fix`.
- General code (Tier B): `general_rl_phase2_iso/…/general_phase2e_full_sigma01_02_20260720_v1/{code,quarantine}`.
- Accepted P=10 root: `three_species_ecological_p10_correction_20260723_v1` (datasets/fit_cache/manifests under `{plus,moor}/`, configs+manifests under `launch_package/`).
- Jobs: checkpoint MOOR `58548481`, PLUS `58548480`; Tier-A array `58569801`; cancelled `58540799`.
- Slurm: `-p m3h -A ce25 -q m3h -c 1 -t 8:00:00 --constraint=xenon-8452Y` (mirrors accepted arrays `58493916`/`58493918`).
- HW/SW: node m3h101 Xeon Platinum 8452Y (Sapphire Rapids); numpy 2.2.6, OpenBLAS 0.3.29 DYNAMIC_ARCH.

## 7. Findings so far (reference-only; not in the 6-method tables)
- **a0 do-nothing (vulture/ricker/0.1) = −183.80**, beats best learned (RefPlan −186.768) by +2.96; a0≡a5 trajectory (r_pos=0 ⇒ dK inert), a5 wastes 3.46 discounted cost.
- **S1: 14/24 cells a constant reference matches/beats best learned** — vulture 8/8 (best const a2 harvest beats by +4.4…+4.8), tiger 5/8 headroom 0.000 (learned==a10 translocation), fox 0/8 (only value-adding species, +0.23…+1.56).
- **Dominated actions** (r_setpoint≤0 group ⇒ dK inert ⇒ lowest-cost dominates): tiger {a5,a6}, fox {}, vulture {a5,a6,a7,a8,a9}. **PLUS & MOOR deploy dominated a5 on all 8 vulture cells.**

## 8. Guardrails / obsolete
- Do not reuse the CANCELLED `58540799` recompute-path outputs (none written).
- Constant-action sweep is reference-only. No method/reward/hyperparameter change in response to any diagnostic.
- Memory: `project_diagnostic_replay_p10.md` (index in MEMORY.md) mirrors this state for future sessions.
