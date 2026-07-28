# Handoff — General-RL Baselines Phase 2C (take over from here)

You are taking over an in-progress, multi-phase effort to correct the four **hidden-`r,K`
general-RL baselines** (RefPlan, OGSRL, BA-MCTS, Delphic; MOPO archived/excluded) in a conservation
POMDP benchmark. Phases 1 → 2B are **done**. Your job is **Phase 2C: implement the approved Phase 2B
plan** in the isolated worktree, then produce a Phase 2C report. **Do not skip the "Hard
constraints" section.**

---

## 0. TL;DR of what happened in this session

1. **Phase 1 (audit)** — inspected the four baselines; found the hidden implementations were paper-
   *inspired* linear/numpy re-implementations with defects. Report:
   `PHASE1_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`.
2. **Phase 1B (deep paper verification)** — read the full ICML25/NeurIPS25/ICLR26/ICLR24 papers,
   reached OGSRL official code, produced a corrective plan. Key discovery: the offline **behavior
   policy is privileged** (`collector.py:211`, acts on true hidden state) so the data carries
   **genuine hidden confounding** → Delphic is applicable. Report:
   `PHASE1B_GENERAL_RL_DEEP_PAPER_VERIFICATION_AND_CORRECTIVE_PLAN.md`.
3. **Phase 2 (implementation, isolated)** — in a git worktree, fixed all four methods + tests +
   canaries. Report: `PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION_REPORT.md`.
4. **Review** — user reviewer raised 5 findings: `REVIEW_PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION.md`.
5. **Phase 2B (plan-only reconciliation)** — accepted the review, ran return-blind measurements, and
   wrote the Phase 2C plan. **This is the document you implement.** Plan:
   `PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md`. **Verdict: NO-GO until Phase 2C lands.**

**The single most important finding to internalize:** Delphic's cross-world Q variance was measured
to be **misspecification-dominated** (observed-vs-counterfactual variance ratio ≈ 1, should be ≫1),
so the current "delphic uncertainty" is NOT validated. Phase 2C must either fix the world
construction (support-gated counterfactual divergence) so it passes the strengthened gate, **or**
honestly downgrade Delphic's name. Do not present misspecification variance as delphic uncertainty.

---

## 1. Where the code lives (READ THIS)

- **Corrected code is in an ISOLATED WORKTREE, NOT main:**
  `/fs04/scratch2/ce25/general_rl_phase2_iso` (detached `git` HEAD at `5f9cf32`).
- **Main repo** `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models` (== `~/ce25_scratch2/…`) is a
  **dirty working tree at `5f9cf32-dirty`** with unrelated ecology work; the **docs/reports live in
  main** at `docs/fix_implement_general_RL/`. Do **all code work in the worktree**, read reports
  from either (they are the same files).
- The hidden-`r,K` infrastructure is **entirely uncommitted** (HEAD `5f9cf32` = "known r,K, before
  hiding"); the worktree imported the current pipeline on top of that base.

### Files changed in Phase 2 (worktree, general-RL only)
Modified vs `5f9cf32`: `src/real_ecology_benchmark/methods/{refplan,ogsrl,bamcts,delphic}.py`,
`src/real_ecology_benchmark/public_models.py`.
New: `src/real_ecology_benchmark/delphic_compat.py`, `tests/real/test_general_paper_mechanisms.py`,
`tests/real/test_general_privacy.py`, `scripts/general_adequacy_probe.py`,
`scripts/make_general_corrected_manifest.py`.
Provenance/digests: `docs/fix_implement_general_RL/PHASE2_provenance/` (pre/post sha256, canary JSONs,
registration). Prepared-but-unsubmitted run dir (in worktree):
`real_ecology_runs/general_corrected_prepared/` (1,152-row manifest + `code/` snapshot + registration).

---

## 2. Hard constraints (do not violate)

- **Do NOT merge the worktree into main; do NOT reset/clean/checkout the main dirty tree.**
- **Do NOT touch the running PLUS/MOOR jobs** (`adapt32-plus-plan`, `adapt32-accept-v2` in
  `squeue`); they run from their own frozen code.
- **Do NOT launch/submit jobs** and **do NOT submit the 1,152-row manifest**.
- **Do NOT inspect performance returns** (`operational_return`, `true_return`, survival return,
  rankings, comparative summaries). Model-internal diagnostics (behavior likelihood, Q-variance,
  cost prevalence/spread, timing, RMSE) are allowed — that is how all measurements were done.
- **Do NOT tune anything by policy return.** Delphic gate thresholds and the OGSRL cost/budget are
  **preregistered before returns**; select by interpretability/non-degeneracy/privacy/stability only.
- Keep the matched **4,000-transition** budget and **shared ensemble size 5** (7 only as a shared
  sensitivity). MOPO stays archived, not deleted.

---

## 3. What Phase 2C must implement (from PHASE2B, §3–§9)

Priority order (blocking first):

1. **Delphic (BLOCKING).** Calibrate the reference behavior model
   (`delphic_compat.py::fit_reference_behavior` → standardize, 400/lr0.3/L2 1e-2). Replace the
   marginal-TV-only gate with the **4-part gate G-D1..G-D4** (conditional NLL vs calibrated
   reference; per-region TV ≤ 0.20; observed-support value agreement `V_obs`; counterfactual/observed
   ratio ≥ 3). Redesign `delphic.py::_fit_world` so latent divergence is **support-gated** (shared
   public-feature Q base + `Δ_w(s,a)=g_w·u(s,a)·κ_amb(s)`, `u`=1−kNN-guardian support). If the
   full-4000 ratio ≥ 3 does not hold for healthy **and** sink pops → **downgrade the name** to
   "ensemble value-disagreement pessimism (Delphic-motivated)" and drop the compatible-worlds claim.
   Add toy tests: matches `P(a)` but not `P(a|history)` → fails; compatible-but-counterfactual → passes.
2. **OGSRL (BLOCKING for final method).** Replace the inert extinction risk with a **public
   low-abundance shortfall cost** `c(o')=max(0,(s_low−o')/s_low)`, `s_low`=20th pct of positive
   **training** observations (public; not private K/threshold). Preregister budget = mean discounted
   `c` under the behavior policy on train, floored 0.02 / capped 0.10. Add a test that the safety
   dual **binds** on a constructed low-abundance case. Keep the OOD guardian.
3. **RefPlan (paper-closeness).** Add a **public conservative policy prior** (calibrated logistic on
   train public features, ε=0.10 uniform floor) used to sample/weight candidate first-actions in
   `plan_marginalized`; keep it a **distinct object** from the model posterior. Toy test: swapping the
   prior changes plan selection with the model belief fixed.
4. **Privacy (verification).** Add `test_private_value_intervention_invariance`: hold the public
   dataset+surrogate fixed, intervene on **one private field at a time** (`K_base, r_min/r_max, kind,
   safety_threshold, population, …`), rebuild `MethodContext`, assert it is **byte-identical** and the
   fits are byte-identical. This catches transformed leaks (`log K`, `K/2`, family branch) that the
   exact-equality scan cannot.
5. **Threads (engineering, resolved in plan).** Pin `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
   MKL_NUM_THREADS=1`, 1 core/task; run a 1-thread-vs-default parity canary; report task-hours,
   CPU-hours, core-hours, wall separately.

Exact file/function targets are tabulated in **PHASE2B §8**; acceptance gates in **§9**.

---

## 4. How to run things (worktree)

```bash
cd /fs04/scratch2/ce25/general_rl_phase2_iso
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl_$USER && mkdir -p "$MPLCONFIGDIR"

python -m pytest tests/ -q                                   # full suite (was 187 passed)
python -m pytest tests/real/test_general_paper_mechanisms.py tests/real/test_general_privacy.py -q
python scripts/general_adequacy_probe.py --population "Amur tiger" --transitions 4000 \
    --episode 25 --act-steps 25 --out real_ecology_runs/general_adequacy_canary   # return-blind
```

- Shared test fixture: `tests/real/test_general_paper_mechanisms.py::_hidden_fixture(sigma_obs=…)`
  builds a real hidden dataset + `MethodContext` + belief cache. Reuse it.
- Populations (K_base): Spotted turtle 31 … Jaguar 325; sinks/depleted: **Egyptian vulture**
  (K325,N0=41), Bottlenose dolphin (K35), Puerto Rican parrot (N0/K=0.31). Use Amur tiger (healthy)
  + Egyptian vulture (sink) as the two canary cells.
- numpy backend = scipy-openblas 0.3.29; general methods are **Python-loop-bound** (BA-MCTS act ≈600
  ms/step at 256/8 is 85% of compute), so threads=1 barely changes wall.

---

## 5. Key facts you must not re-derive wrong

- **OGSRL extinction risk is degenerate benchmark-wide**: `terminated`(state==0) prevalence 0.0,
  `constant_single_class`, risk identical across all 11 actions → the old safety-cost channel cannot
  bind. That is *why* Finding 4 requires the public low-abundance cost. Measured on Amur + vulture.
- **Delphic obs-vs-cf Q-variance ratio ≈ 1** (1.06 Amur / 2.42 vulture on full 4000) = misspecification,
  not delphic uncertainty. A behavior-only gate does not fix this.
- **Delphic NLL reconciliation**: crude reference 27.5 (uncalibrated), standardized reference 2.13,
  worlds 1.86–1.92 (calibrated). The behavior channel is fine; the **value channel** is the problem.
- **Delphic ambiguity correctly vanishes at σ_obs=0** (diversity ≈4e-23, gate fails) — keep this.
- **RefPlan** hidden `act` now marginalizes the posterior (`public_models.py::plan_marginalized`);
  **BA-MCTS** now does the in-tree Eq-4 belief update (`bamcts.py::_public_member_loglik`,
  `_belief_from_logweights`). These Phase-2 fixes STAND — do not regress them.
- Papers (local PDFs): `docs/references/general_model_based_offline_RL_papers/` (RefPlan ICML25,
  BA-MCTS ICLR26) and `docs/references/discrete_action_continuous_obser_state/` (Delphic ICLR24,
  OGSRL NeurIPS25). OGSRL official code: `github.com/Runz96/SafeRL-OGSRL` (guardian = Gaussian KDE,
  ConOpt = CPO, transitions = kNN).

---

## 6. Files to read, in order

1. `docs/fix_implement_general_RL/PHASE2B_GENERAL_RL_REMAINING_ISSUES_PLAN.md` — **the spec you implement.**
2. `docs/fix_implement_general_RL/REVIEW_PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION.md` — the 5 findings + accepted decisions.
3. `docs/fix_implement_general_RL/PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION_REPORT.md` — what Phase 2 already built.
4. `docs/fix_implement_general_RL/PHASE1B_…PLAN.md` — full-paper mechanism cores per method.
5. This file.
6. Code (worktree): `src/real_ecology_benchmark/methods/{delphic,ogsrl,refplan,bamcts}.py`,
   `delphic_compat.py`, `public_models.py`; tests `tests/real/test_general_{paper_mechanisms,privacy}.py`;
   `scripts/general_adequacy_probe.py`; `src/real_ecology_benchmark/{collector,public_surrogate,config,beliefs,pipeline}.py` for the hidden pipeline.
7. `docs/fix_implement_general_RL/PHASE2_provenance/` — canary JSONs + digests (evidence).

---

## 7. Definition of done (Phase 2C)

All PHASE2B §9 gates green; full `pytest tests/` passing; return-blind canaries (Amur + vulture)
showing: Delphic gate decision (pass with ratio ≥3, or documented downgrade), OGSRL dual binds on a
low-abundance case, RefPlan prior changes selection, private-value intervention invariance passes,
thread parity confirmed. Regenerate pre/post sha256 + registration. **Manifest stays unsubmitted.**
Write `PHASE2C_GENERAL_RL_FINAL_IMPLEMENTATION_REPORT.md` and **stop for review** — do not launch the
experiment or inspect returns.
