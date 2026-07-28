# Session Handoff — Deep RL Population Models (2026-06-13)

Detailed context dump so a new chat can continue seamlessly. Covers the whole
working session: the selected-method benchmark, the POMDP stress models, the
Phase-116 danger-zone benchmark, and the clean GitHub repo build.

---

## 0. Environment & how to run

- **Working dir:** `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models`
  (also reachable as `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`).
- **Main implementation:** `claude_build/` (the messy working repo).
- **Clean GitHub-ready repo:** `DeepRL_Population_Models/` (built this session; see §6).
- **Python env:** `module load miniforge3 && eval "$(conda shell.bash hook)" && conda activate pytorchrl`
  (has torch, hydra, pandas, pytest; the bare login `python` does NOT).
- **Cluster:** SLURM. GPUs seen: NVIDIA L40S (fast), A40, Tesla T4 (slow).
- **Config:** Hydra. **Tracking:** Weights & Biases (`wandb.mode=disabled` to skip).
- Everything is registry-driven: `env=` / `active_env=`, `active_model=` + `model=`,
  `active_planner=` + `planner=`.

---

## 1. Project & the four methods

Offline, model-based RL for **adaptive management of a harvested population**.
Central question: can *general* model-based offline RL (learns dynamics from data)
match/beat *ecology-specific* mechanistic baselines (assume a Ricker family) —
especially under **model misspecification**?

| Key | Method | Type |
|---|---|---|
| `ensemble` + `pessimistic` | **MOPO** (categorical dynamics ensemble + uncertainty-penalized planner) | general MBRL |
| `refplan` | **RefPlan** (Reflect-then-Plan; history-conditioned, uncertainty-aware) | general MBRL |
| `bioconserv18_plus` | **PLUS** (BioConserv'18 finite hidden-model Bayesian planner) | ecology baseline (Ricker) |
| `moor` | **MOOR** (ExpertSys'23 least-squares mechanistic offline RL) | ecology baseline (Ricker) |

**Problem setting (POMDP):** observation = discrete abundance bin
`x = floor(s/10)`, 100 bins, max abundance 1000; hidden per-episode `r_base~U`;
reward `R = α·x/x_max − cost(a)` (α=1, x_max=100); 5-action default + 10-action
(`ricker_full`). Stress envs add hidden `C`/`θ`/`z_t` and optional direct action
effects `s ← s·(1−harvest_fraction) + stocking_delta` (applied before growth).

---

## 2. Experiment 1 — Ricker vs Schaefer selected-method benchmark

`quick_selected_methods.sh`; seed 42, DATA_N=25000, 50 eval episodes, horizon 50,
all four methods sharing one offline dataset per setting. **Full-budget** results:

| Method | Ricker | Schaefer |
|---|---:|---:|
| MOOR-Ricker | 16.436 | 16.323 |
| PLUS | 16.436 | 16.323 |
| MOPO | 16.013 | 15.595 |
| RefPlan | 14.947 | 14.688 |

**Key findings:**
- **PLUS ≡ MOOR exactly** (same greedy policy under exact-observation finite MDP).
- The Ricker↔Schaefer mismatch is **decision-irrelevant** (both share equilibrium
  K; reward depends on abundance/cost → optimal policy family-invariant), so the
  ecology baselines aren't harmed and lead on reward.
- **MOPO degrades the MOST under mismatch** (−0.42 vs −0.11 baselines): its
  uncertainty rises → pessimism penalty → conservative, lower-reward behaviour.
  RefPlan trails but is best-calibrated (uncertainty triples under shift).
- **Compute lesson (important):** full-budget MOPO eval is ~2.8h on an L40S
  (~4 s/step; ~6.25M batch-1 ensemble predicts) — it is **slow, not a hang**.
  An earlier claim that it "cannot finish / ~42h" was **WRONG** (extrapolated a
  CPU microbenchmark to GPU; idle-looking GPU ≠ stalled). Schaefer MOPO timed out
  only on a slow **Tesla T4** (4h wall), then completed on an L40S (~2.9h).
- Report: `docs/latex_files_reports/quick_selected_methods_report.tex` (+ PDF).

---

## 3. Experiment 2 — POMDP stress models

Three new envs (`src/environments/stress_pomdp_envs.py`):
`allee_ricker_pomdp` (hidden critical-depensation threshold C),
`theta_logistic_pomdp` (hidden curvature θ), `regime_switch_pomdp` (hidden
two-regime Allee). POMDP contract enforced (hidden vars only in `info`/diagnostics,
never in obs or saved dataset). Two reward modes: `base`, `collapse_sensitive`
(one-time −10 entry penalty, absorbing bin-0). Collection: `mixed_danger_zone`.

**Issues found & fixed:**
- Original defaults → **degenerate data** (~100% episodes collapse, 83–96% bin-0)
  because the spec's high growth priors crashed everything.
- The collapse-rate test divided by `n_transitions` not `n_episodes` (so it passed
  under near-total collapse). **Both fixed** by codex (lowered `r_base`, fixed test
  denominator, entry-only penalty, coverage guards). Now collapse 0.15–0.24.
- But the smoke benchmark then showed **no RefPlan advantage**: with low growth the
  envs are too gentle / decision-irrelevant; only theta showed a learned win, and
  it was **MOPO** not RefPlan.

---

## 4. Experiment 3 — Phase 116 danger-zone benchmark (current focus)

Designed to make misspecification **decision-relevant**. Setting tag `116`;
configs `config/env/{allee,theta,regime}_ricker..._116.yaml` and `..._116_10a.yaml`
(5- and 10-action). Key machinery added:
- **Separate eval start distribution** (`eval_start` config + `reset_for_eval()`):
  episodes start in the danger zone so the hidden threshold matters.
- **`mixed_danger_zone_116`** collection (better live-danger coverage).
- **MOPO vectorized planner** (`predict_batch` + `_rollout_values_vectorized`,
  `use_vectorized: true` default) — makes full-budget MOPO tractable; numerically
  verified (test passes).
- **Control-gap gate** (`scripts/analysis/stress116_control_gap.py`): oracle MPC
  (true dynamics) vs Ricker MPC (same reward, wrong dynamics). Hard pass/fail on
  `reward_gap ≥ 1` and `collapse_gap ≥ 0.05`. **This is the go/no-go.**
- **Reward "option (a)"**: PLUS/MOOR now apply the collapse penalty under their OWN
  (misspecified) transition model — so a learned win is attributable to dynamics
  modeling, not reward blindness.
- **Action authority**: `ActionSpec` gained `harvest_fraction` / `stocking_delta`
  (in `reward.py`), applied before growth in the true env, PLUS, MOOR, AND the
  control-gap Ricker MPC. This was the fix that made the threshold controllable.

**Gate status (PASSES):** Allee 2.96/0.125, regime 3.20/0.125 (hard-gated);
theta diagnostic 0.84/0.0 (no Allee threshold → collapse-gap N/A, intentionally
non-blocking). 10-action cells have stronger gaps (allee 5.86, regime 7.05,
theta 2.79). Tests: 19–20 pass.

---

## 5. CURRENT RUNNING EXPERIMENT (re-check before acting)

A **reduced one-day TUNING-ONLY run** (the user shortened the 5–7 day plan):
- Job `56433936` — manifest `outputs/manifests/stress116_tune_methods_1day.tsv`,
  216 rows (108 MOPO + 108 RefPlan), 6 env cells × 3 tuning seeds (1160–1162),
  6 configs each, `ROWS_PER_TASK=2`, `--array=0-107%4`, collapse_sensitive, 50 eps.
  At last check **~80% done** (tasks ~87–92 running, 93–107 pending).
- Job `56433939` — dependent **freeze-only** selector (afterok); writes frozen
  configs to `outputs/stress_pomdp_116/frozen/` — does **NOT** launch the final
  benchmark.

**CRITICAL:** this run is **tuning only**. It has **no PLUS/MOOR, no final
comparison, no held-out seeds, no report** — so it does NOT yet answer "do
RefPlan/MOPO beat PLUS/MOOR." **Next step after it finishes:** launch the final
benchmark with frozen configs — all 4 methods incl. PLUS/MOOR, **held-out seeds
7001–7005**, 100 eval episodes, collapse_sensitive (+ optional base), then
aggregate the report (`stress_pomdp_116_after_tune.sh` / `aggregate_stress116_phase116.py`
exist but are NOT queued). Watch MOPO compute (use L40S/A40, long walls). User
wants a **meaningful** multi-seed result (no near-term deadline) across **both**
5-action and 10-action sets.

---

## 6. The clean GitHub repo — `DeepRL_Population_Models/`

Built this session for the user's professor. **4 methods only**; pruned the other
6 model adapters (combo/cql/iql/bamcts/romi/hmmdp) + MOReL planner and their
configs/tests; rewrote registries; stripped hmMDP blocks from `run_pipeline.py` /
`evaluate.py`. **All `claude`/`codex` strings scrubbed** from every text file.
Structure: `src/` (4 models + infra), `config/`, `scripts/` (core/analysis/
experiments/slurm/single_runs), `tests/` (9 suites), `docs/`. **112 tests pass.**

`docs/`: `problem_setting.tex`, `references.md` (cites ALL source papers — see §7),
`EXTENDING.md` (how to add a new algorithm/env/planner — repo is registry-based and
extension-friendly per the user's request), and `reports/`:
- `ricker_schaefer_selected_methods.tex` (+pdf)
- `stress_pomdp_benchmark.tex`
- `mopo_ablation_report/` (main.tex/pdf + EXPERIMENT_REPORT.md + 9 figures) — **kept
  at user request**, with a **scope note** that the hmMDP comparison baseline is
  outside the 4-method scope and not shipped.

Codex did a follow-up cleanup pass (caught real leftovers I'd missed:
`run_general_mbrl_single.sh` bamcts/romi defaults, stale docstrings) — verified
safe (docstring-only; 112 tests still pass).

**Remaining before GitHub upload (user does this):**
- Recompile the two report PDFs from the scrubbed `.tex`.
- Decide on bundling paper PDFs (currently referenced, not included — copyright).
- Optionally: full citations + a `docs/references.bib`.
- SLURM scripts are cluster-specific (BASE/partition need editing) — noted in README.

---

## 7. Papers / citations

All source PDFs are cited in `DeepRL_Population_Models/docs/references.md`:
PLUS/BioConserv'18, MOOR/ExpertSys'23, MOPO (Yu et al. 2020), RefPlan (ICML'25);
context: AAAI'12 MOMDPs, AAAI'21 universal solver (the hmMDP baseline), ICLR'26
Bayes-Adaptive MCTS, ICLR'26 Robust Value-Aware. PDFs are not bundled. Papers are
also referenced in the reports and `problem_setting.tex`.

---

## 8. Audit / review docs produced this session (in `docs/codex_claude_conversation_md/`)

- `claude_check_flow_codex.md` — review of the selected-method run flow (note: its
  §2 "MOPO cannot finish" was later CORRECTED — MOPO does finish; see §2 above).
- `claude_check_codex_code.md` — audit of the POMDP stress implementation (+ the
  verified-fixes recheck section).
- `claude_check_116_plan.md` — review of the Phase-116 plan.
- `claude_check_116_implementation.md` — review of the 116 implementation (the gate
  initially failed → action-authority fix → gate passes).

---

## 9. Memory files (persist across chats)

In the auto-memory dir: `project_deeprl_pipeline`, `project_mopo_planner_eval_hang`
(MOPO eval slow-not-hang), `project_stress_pomdp_degenerate_data` (was degenerate →
fixed → action-authority gate now passes). Check `MEMORY.md` index.

---

## 10. Immediate next actions for the new chat

1. **Re-check** the tuning run: `squeue -u $USER`; `sacct -j 56433936 -X`; look for
   `outputs/stress_pomdp_116/frozen/`.
2. When tuning + freeze finish: **launch the final benchmark** (frozen configs +
   PLUS/MOOR + held-out seeds 7001–7005 + 100 eps + report), 5a **and** 10a, with
   MOPO compute budgeted on fast GPUs. This is what actually answers the headline
   question.
3. Help finalize `DeepRL_Population_Models/` for upload (recompile PDFs, optional
   BibTeX) — the user reviews/refines and uploads it themselves.
