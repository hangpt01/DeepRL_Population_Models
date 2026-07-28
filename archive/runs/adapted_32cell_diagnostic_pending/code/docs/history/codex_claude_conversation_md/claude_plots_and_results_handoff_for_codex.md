# Handoff for Codex: verify the plots and the results report

Date: 2026-06-21. Author: Claude (Opus 4.8). Repo of record: `discrete_action_cont_obser/`.

## What I'm asking Codex to verify
After the full Tier-2 matrix completed, I (1) generated per-cell trajectory plots and (2) wrote an empirical results LaTeX report. Please re-check three things:
1. **Are the run experiments legit?** (matrix actually completed, calibrated, gated, no leakage, no stale artifacts.)
2. **Are the plots reasonable?** (capture faithfully reproduces the evaluation; figures are sensible.)
3. **Does the report correctly describe what the code and experiment actually are?** (every number traceable; method descriptions match the code; claims supported.)

I made several judgment calls (listed under "Decisions to scrutinize") — please challenge them.

---

## Context recap (state before this work)
The 384-row matrix finished after a calibration fix. Jobs:
- Gate `57056985` (completed), first matrix `57056986` (128 COMPLETED / 256 FAILED at the enforced calibration stop), rerun `57087408` (**256/256 COMPLETED, 0 failed**).
- Final: **384/384 `summary.json`, 24/24 cells × 16 rows, all 24 cells in `[0.15,0.24]`, all gates pass.** Aggregation in `outputs/aggregate.json`.
- The 256 failures were 4 out-of-band families (allee\_5a, regime\_5a, theta\_5a, theta\_10a) fixed by collector-only calibration profiles (`src/tier2_benchmark/collector.py` `PROFILES`); allee\_10a/regime\_10a kept the default profile and were never re-run.

---

## Part A — Trajectory plots

### Problem
Per-step traces were **not** saved by the pipeline (only per-episode aggregates in `episodes.csv`), fitted policies were **not** saved, and **matplotlib was not installed**. So trajectories had to be regenerated.

### What I did
1. Installed matplotlib (`pip install --user matplotlib` → 3.9.4). Login-node python only; capture jobs are NumPy-only.
2. Wrote `scripts/capture_trajectories.py` (+ `scripts/slurm/capture_trajectories.sh`). For each of the 24 cells it:
   - rebuilds the cell config exactly as `run_manifest_row.py` does (`full.yaml` → `environment_with_kind_defaults(kind)` → set `num_actions`, `observation_noise_sigma`);
   - reuses the **existing** dataset (`ensure_dataset(regenerate=False)`), the saved **belief cache** (`public.learned.beliefs.npz`), and the saved **learned filter** — only the per-method `fit` is recomputed;
   - re-fits all 7 methods via `build_method(...)` (same defaults/seeds as the matrix: `seed=cfg.seed+40000`);
   - rolls **5 paired episodes** (seeds `7001..7005`) capturing per-step **true abundance** (`evaluator_info["state"]`), **action**, **immediate reward**; loop mirrors `evaluator.py` (act → `env.step` → `policy.observe(PublicTransition(...))` → `filter.update`); nan-padded to horizon 50;
   - saves `outputs/trajectories/<cell>.npz`.
3. Smoke-tested capture inline (allee\_10a, fast methods) and synthetic-tested the plotter before launching.
4. Submitted capture array `57141768` (`--array=0-23%24`); **24/24 COMPLETED**, 24 npz files.
5. Wrote `scripts/plot_trajectories.py` → `outputs/plots/<cell>.png` (3 stacked panels: true abundance, mean action id, immediate reward — one line per method, mean over the 5 episodes; safety=50 and K=500 markers) + `outputs/plots/_contact_sheet_abundance.png`. I visually inspected the contact sheet and `allee_10a_sigma0.2.png`.

### Decisions to scrutinize (plots)
- **Learned filter only** (the primary config); raw/ricker not plotted.
- **5 paired episodes**, and I plot the **mean over episodes** for all three panels (a single line per method). Mean can mask individual collapses; single-episode lines would show them more starkly.
- **True latent abundance** is plotted (from the evaluator/private channel) — appropriate for an ecological readout, but it is the hidden state, not what the agent sees. Observed $o_t$ is not overlaid.
- **Mean action id** as a continuous line (loses categorical structure); an alternative is per-method small-multiples.
- The re-fit assumes the captured policy == the matrix policy. Fits are seeded deterministically (`cfg.seed+40000`), so they should reproduce the matrix policies, **but I did not bit-check captured vs. matrix policies** — worth confirming (e.g., that a captured method's mean return over 50 episodes matches its `summary.json`).

### What Codex should check (plots)
- `capture_trajectories.py:roll()` matches `evaluator.py`'s loop (sanitized `PublicTransition`, no `evaluator_info` reaching the policy, same filter reset/update, same fallback-to-0 on exception).
- Re-fit config equals the matrix config (it loads `full.yaml` and the same per-cell overrides; belief cache/learned-filter reused, not recomputed differently).
- Spot-check 1–2 cells: do plotted mean returns / abundances agree with the corresponding `summary.json` (within episode-count sampling)?
- Are the abundance traces ecologically sensible (stay roughly in the 250–550 band, above the 50 floor; collapses show as dips toward 0 then episode end)?

---

## Part B — Results report

### What I did
1. Duplicated the spec into the repo: `docs/22_6_Continuous_Observation_New_Baselines.tex` (exact copy of the parent `docs/…` file).
2. Extracted all numbers programmatically from `outputs/aggregate.json`, the 384 `outputs/evaluation/*/summary.json`, `outputs/calibration/*`, `outputs/gates/*` (one extraction script; printed tables).
3. Wrote `docs/22_6_Continuous_Observation_Experiment_Results.tex` — a **standalone-compilable** companion that **shares the spec's preamble macros** so its `\section{}` blocks can be appended before `\end{document}` of the duplicated spec after approval (header comment explains the merge). **Verified it compiles** with `pdflatex` (clean); removed the generated PDF/aux.

### Report contents
Realized configuration; hyperparameters **with meaning** (env/data/reward, particle filter, dynamics+MPC, per-method, collector profiles); calibration + gate outcomes; results tables (operational return, beats-both rate, collapse/unsafe, filter RMSE, **filter ablation** learned−raw, **PLUS/MOOR Ricker-fidelity**); analysis (6 findings); limitations + 8 improvements; pointers to plots/artifacts.

### Key numbers in the report (for Codex to cross-check against outputs)
- **Overall mean operational return**: moor 7.11, mopo 6.64, refplan 6.43, plus 5.94, ogsrl 5.59, bamcts 4.78, delphic 4.54 (= `aggregate.json` `model_return_mean`).
- **Operational return by σ (learned)** e.g. RefPlan 7.33/7.23/7.14/5.14; MOOR 6.97/6.94/7.22/**7.80**; OGSRL 4.37/5.95/7.39/7.48; Delphic 5.58/3.74/5.98/5.77.
- **Beats-both rate (learned)**: BA-MCTS 0.67/0.83/0.50/0.00; RefPlan 0.67/0.50/0.33/0.00; MOPO 0.50/0.33/0.33/0.00; all learned ≈0 at σ=0.4 (OGSRL 0.17).
- **Filter ablation (learned−raw, mean op return)**: delphic +1.45, ogsrl +1.40, bamcts +1.30, refplan +0.56, mopo +0.28, plus +0.09, moor −0.05.
- **Ricker-fidelity**: PLUS learned 5.97 / raw 5.88 / ricker 5.99; MOOR learned 7.23 / raw 7.28 / ricker 6.83.
- **Filter RMSE (mean over methods)** ≈ 0 / 43 / 80 / 160 at σ=0/0.1/0.2/0.4.
- **Calibration**: allee\_10a 0.184, allee\_5a 0.212, regime\_10a 0.196, regime\_5a 0.201, theta\_10a 0.209, theta\_5a 0.209.
- **Gate**: allee/regime hard-pass (reward gap 2.9–7.3, collapse gap 0.25–0.80); theta diagnostic-pass (1.5–5.0, 0.15–0.40).

### Decisions to scrutinize (report)
- **Aggregation convention**: per-method×σ tables are the **mean over the 6 (family) cells** at that σ (matches `model_return_mean`'s all-cell averaging). Beats-both is taken straight from `aggregate.json` (paired seed-level vs `max(PLUS,MOOR)`). Filter ablation = mean over all cells.
- I report **operational return** (and state $R^{\mathrm{true}}$ tracks it within ≤0.1, so no leakage) — but the report **warns operational returns are not comparable across σ** (log-normal mean bias).
- Hyperparameter values are transcribed from `configs/full.yaml` + method `__init__` defaults (e.g. bamcts sim=128/depth=5; delphic worlds=10/λ=0.1/α=0.5; ogsrl budgets 0.02/0.05/iters=30; plus candidates=21; refplan members=15/plan-subset=5; ensemble=15; MPC h=5/seq=64/part=16/λ=0.5; filter particles=1024). **Please confirm these match the code paths actually executed** (build_method does not override method defaults, so the `__init__` defaults are what ran).
- Analysis claims (esp. "MOOR best at high noise", "Delphic doesn't show the predicted σ-signature", "filter helps learners not baselines") are my interpretations of the numbers — please confirm they're supported and not overstated.
- The report repeats two audit-level caveats as improvements (Delphic worlds are random-feature-linear; OGSRL deployment safety limit reuses the discounted-budget value as a one-step threshold). Confirm these still hold in the current code.

### What Codex should check (report)
- Every table value matches the outputs (re-run my extraction or your own).
- The **method descriptions** match the code: Delphic compatible-worlds + `u_Δ=Var_w` + CQL penalty (`methods/delphic.py`); OGSRL guardian + dual + hard fallback + posterior-particle risk (`methods/ogsrl.py`); PLUS Rao-Blackwellized per-candidate bank (`methods/plus.py`); MOOR LS-on-filtered-means + continuous fitted-Q (`methods/moor.py`); dynamics per-action quadratic ridge predicting latent then composing emission (`dynamics.py`).
- The hyperparameter "meaning" column is accurate.
- Nothing in the report contradicts the locked design (action tables, equations, safety\_threshold=50, reward/collapse semantics, priors, σ-shared physics).

---

## Known caveats I want Codex to challenge
1. **Plots re-fit rather than replay** the matrix policies (policies weren't saved). Deterministic by seed, but unverified against the matrix runs.
2. **Plots use 5 episodes / mean lines** (illustrative); the statistical results use 250 episodes. Don't read significance off the plots.
3. **Learned filter only** for plots; raw/ricker not visualized.
4. **Delphic and OGSRL** carry the prior audit caveats (random-feature worlds; OGSRL low-noise over-conservatism + safety-limit scale mismatch) — the report flags these as improvements, not as faithful-to-paper guarantees.
5. **matplotlib installed `--user`** on the login node; not added to `requirements.txt` (it's analysis-only, not a runtime dep).

## Files to review
- Plots: `scripts/capture_trajectories.py`, `scripts/slurm/capture_trajectories.sh`, `scripts/plot_trajectories.py`, `outputs/trajectories/*.npz`, `outputs/plots/*.png`.
- Report: `docs/22_6_Continuous_Observation_Experiment_Results.tex` (new), `docs/22_6_Continuous_Observation_New_Baselines.tex` (duplicate of spec).
- Data the report draws on: `outputs/aggregate.json`, `outputs/evaluation/*/summary.json`, `outputs/calibration/*`, `outputs/gates/*`.
- Calibration fix (context): `src/tier2_benchmark/collector.py` (`PROFILES`), `scripts/calibrate_profiles.py`, `scripts/regenerate_failed_cells.py`, `tests/test_calibration_profiles.py` (32/32 tests pass).
