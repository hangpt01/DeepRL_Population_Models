# Full-Session Audit Brief — Motivation Experiment + Search-Budget Audit

**For:** codex
**Date:** 2026-07-12
**Base commit:** `e74526e` (working tree dirty — see §5)
**Two runs completed:** `real_ecology_runs/motivation_native_20260711/` and `real_ecology_runs/general_audit_20260712/`

This is the complete trail: what the PI asked, what I did, what came out, and where to attack it.

---

## 1. What I was told, and what I did

| # | PI instruction | What I did |
|---|---|---|
| 1 | Read the handoff. Produce a gate doc: does native discrete-POMDP PLUS/MOOR exist? Code paths, conflicts, plan, first smoke test. **Read-only. No code. No runs.** | Wrote `SERVER_CONTEXT_motivation_native_baselines_gate.md`. Gate: **build required**. Found `PLUS-GPU` is a CuPy routing flag, not a solver; no POMDP library installed; **but a compiled SARSOP + POMDPX writer exists in `baseline_original/`** (bound to the published hmMDP problem, so it doesn't close the gate). Touched nothing. |
| 2 | (codex audit relayed) | Folded in 3 load-bearing corrections. Two were **my errors**: the K lattice is population-scaled (`dK = {0.1, 0.3}·K_base`, not a fixed 25 — my claim broke 7 of 9 populations), and the aggregator keys on `filter`, so renaming baselines would leave the headline table silently **empty**. |
| 3 | (codex implemented the native build) | Audited it. Found 3 blockers: 4 new test errors exposing a missing method↔filter guard; new tests invisible to `unittest` (pytest-style, pytest not installed); and **nearest-bin transition snapping creating artificial fixed points** — the model believed a declining population stops declining and that extinction is unreachable. |
| 4 | (codex fixed) | Verified empirically. Barycentric interpolation now reproduces drift to machine precision; self-loops 13–47 → 1–2; **resolution sensitivity now converges** (was non-monotone). |
| 5 | **Chose: PLUS-native's bank = the 4 mechanistic forms** | Implemented. Required augmenting the solver's hidden state to `(abundance, regime)` for the regime candidate — pinning regime=0 would have been a crippled model, which §4 of the handoff forbids. Also **retracted my own §3 corollary**: I had said the bank should span `K`; `K` is unidentifiable here (6/11 actions have `r ≤ 0`, which cancels `K` out of the Ricker exponent), which made PLUS ≡ MOOR byte-for-byte. |
| 6 | **"run the complete acceptance battery"** | Built and ran an 8-check battery (288 rows). **8/8 PASS.** |
| 7 | **"plan the experiment for 12–16h using GPU and CPU; check server limits"** | Wrote `SERVER_PLAN_motivation_native_run.md`. **Key finding: the GPU is unusable** — `GPU_SUPPORTED_WORKLOADS` covers only `mechanistic_transition` / `method:plus` / `oracle_ablation:plus`, and this run excludes adapted PLUS. Limits: `MaxSubmit=1000`, `cpu=250`. |
| 8 | (codex verified, then "don't run yet") | Held. Patched the canary index (task **32**, not 0 — task 0 is the vulture sink). |
| 9 | **"When acceptance_report.json lands"** | It landed 8/8. Ran canary (task 32) → 5 summaries, `fallback_count=0`, one shared dataset hash. Then released the sweep. **It completed: 1440 + 1152 rows, 0 failures.** |
| 10 | *"I slept through it, I can only run in the next 6–8h, modify the plan"* | **Corrected the premise: the experiment had already run and finished.** Did not fabricate a new plan. |
| 11 | **"give me output for codex to audit"** | Wrote `SERVER_AUDIT_REQUEST_motivation_results.md` with 6 claims to attack, flagging my own weakest reasoning (C4/C5). |
| 12 | (codex: the *general-method audit* plan is not launched; concurrency math is wrong) | Correct. Equal `%80` caps **do not pool** → BA-MCTS alone implied a **10.4h** wall. Patched to cost-proportional caps. **Withdrew** the "drop the BA-MCTS corner if it overruns" contingency — that corner is the evidence most able to falsify my own headline. |
| 13 | (codex: rev-3 has 3 new blockers) | All real. Timings were stale (1.6–1.8× under), BA-MCTS walltime would have **killed tasks**, and the runner had **no config dimension in its output path** → 11 configs per cell would have silently overwritten each other. Fixed → rev-4. |
| 14 | (codex signed off rev-4) | Implemented the code, ran the hard gates: **G1** (bit-identical no-op) and **G2** (dataset identity), then the canary. |
| 15 | **"how long will they finish?"** | Found I'd run the BA-MCTS canary's 11 configs **sequentially** (3.5h). Cancelled, reran parallel (41 min). |
| 16 | **"so you're running 144 right? it's ok for me"** | Submitted the 144-cell sweep. |
| 17 | **"no — I want 3–4 hours"** | Cancelled. **Cut cells 144 → 72 (yield dropped); kept ALL 24 configs.** Recorded in the plan *before* resubmitting (§R6). Relaunched → finished in ~2.9h. |

---

## 2. RESULT 1 — Motivation experiment: **NEGATIVE**

`real_ecology_runs/motivation_native_20260711/` · **1440 main + 1152 resolution rows, 0 failures**

> **Native ecological baselines beat general offline MBRL in 860/864 paired cells (99.5%).**
> This is the **opposite** of the paper's premise.

| beats-both rate (recoverable pops) | |
|---|---|
| `refplan` | **0.000** at every σ, both reward modes |
| `bamcts` | **0.000** at every σ, both reward modes |
| `ogsrl` | 0.000–0.04 |

Gap is **widest** on allee/theta/regime (−1.13 / −1.11 / −1.24) — the families where naive Ricker was supposed to fail.

**Integrity:** 288/288 cells share one `dataset_sha256`; resolution check **0/288 flips**; acceptance battery **8/8**.

Doc: `SERVER_RESULTS_motivation_native_run.md`

---

## 3. RESULT 2 — Search-budget audit: **MODEL-LIMITED**

`real_ecology_runs/general_audit_20260712/` · **72 safe cells × 24 configs = 1728 rows, 0 failures**

> **More search budget does NOT close the gap. The general methods are not under-tuned.**

| | |
|---|---|
| as-run general beats native | **2.8%** (2/72) |
| **tuned** (best of 24 configs) | **4.2%** (3/72) |
| mean gap as-run → tuned | −1.216 → **−0.962** (only 21% closed) |

**The mechanism is the finding: deeper search is strictly WORSE.** Mean return by depth, on **recoverable populations** (sinks excluded, so they cannot drive it):

| method | shallow | mid | deep |
|---|---|---|---|
| `refplan` | `h5` **7.024** | `h10` 6.220 | `h20` **5.646** |
| `bamcts` | `d5` **6.848** | `d10` 6.490 | `d20` **6.176** |
| `ogsrl` | `roll6` **7.209** | `roll12` 7.187 | `roll24` **7.113** |

Monotone for all three. The fingerprint of a **wrong model**: extra rollout steps compound model error, so a deeper plan is a more *confidently wrong* plan. A search-limited agent improves with depth; these do the opposite. The small gains that exist come from **breadth** (256 sequences / 256 simulations).

**[CORRECTED]** An earlier revision claimed "every winner has `pessimism = 0`". **That was wrong** — best-by-mean gives `pess0p5`, mode-of-argmax gives `pess0`, and the marginal is 0.01–0.11 on returns of ~6. **Pessimism is a wash.** Caught by codex; the depth claim is unaffected.

Doc: `SERVER_RESULTS_general_method_audit.md`

---

## 4. Claims to attack (ranked by how likely I am to be wrong)

### C1 — **The verdict's coverage gate.** *(I found a bias in my own favour here — check I removed it fully.)*
The first analysis reported **141 cells / 2.8%**. It pooled in **69 `yield` cells left over from the sweep I cancelled**, which have *partial* config coverage. A truncated config set understates the tuned general → **biased toward my own "tuning doesn't help" conclusion**. Clean figure on the 72 fully-covered cells: **4.2%**.
```bash
cd real_ecology_runs/general_audit_20260712 && python3 analyze_budget.py
# must print: cells with FULL 24 cfg: 72 | cells DROPPED (partial): 69
```
**Attack:** is the coverage gate correct? Are there other partial cells? Does the conclusion hold if you *include* the yield leftovers with their partial sets (it should get *worse* for the generals, not better)?

### C2 — **"Deeper search hurts"** — *codex checked this; it SURVIVES on recoverables only.*
The worry was that the sinks (safe-mode occupancy penalties) drive the means. Recomputed on recoverable populations only: `refplan` h5 7.024 > h10 6.220 > h20 5.646; `bamcts` d5 6.848 > d10 6.490 > d20 6.176; `ogsrl` roll6 7.209 > roll12 7.187 > roll24 7.113. **Monotone, all three.** Claim stands.
**However codex also found a real overclaim:** "every winner has pessimism = 0" was false (see §3). Corrected; pessimism is a wash.

### C3 — **G1 no-op proof.**
`bamcts` and `ogsrl` must be **bit-identical** to the motivation run at defaults, or "we tuned them and it didn't help" is vacuous (the knob previously did not reach either method).
`gates/G1_no_op_proof.json` — 6/6 exact float equality.

### C4 — **G2 dataset identity.** `gates/G2_dataset_identity.json` — 144/144 hashes match. Nothing regenerated.

### C5 — **The motivation run's own diagnosis (weakest link, carried over).**
I claim planning depth isn't why the natives win (truncating native VI to a matched 5-step lookahead costs only ~4%). But a VI *iteration* is a full Bellman backup and an MPC *step* is a rollout — **these may not be commensurate**, and it was one cell. The search-budget audit now supports the same conclusion by an independent route, which strengthens it — but the original proxy is still shaky.

### C6 — **Scope.** 72 cells, **safe mode only**. Reduced power, not reduced validity: all 9 populations, 4 families, both σ endpoints, **all 24 configs**. Yield was dropped because the motivation run showed `refplan`/`bamcts` at exactly 0.000 in *both* modes at every σ.

---

## 5. Files

### New source
| file | purpose |
|---|---|
| `src/real_ecology_benchmark/discretize.py` | native grid; K lattice normalized by `K_base`; σ=0 point-mass emission; `(abundance, regime)` hidden state |
| `src/real_ecology_benchmark/native_solver.py` | tabular MOMDP solver; **barycentric** transition (the fixed-point fix); 4 mechanistic forms |
| `src/real_ecology_benchmark/methods/moor_native.py` | single fitted Ricker |
| `src/real_ecology_benchmark/methods/plus_native.py` | **posterior over the 4 forms** (not over `K` — unidentifiable) |
| `tests/real/test_native_baselines.py` | 14 unittest tests |

### Modified source
`config.py` (PlannerConfig + `bamcts_depth`/`bamcts_simulations`/`ogsrl_rollout_horizon`, defaults = old literals) · `methods/bamcts.py`, `methods/ogsrl.py` (kwarg-wins-else-config) · `beliefs.py` (`DiscreteGridFilter`) · `pipeline.py` (`native_discrete` route, method↔filter guard, belief-cache key carries grid size) · `manifest.py` (cross-filter aggregation) · `cli.py` · `collector.py`/`dataset.py` (`dataset_sha256`) · `scripts/run_real_manifest_row.py` (**`--dataset-root`, planner overrides, `config_tag` in output path**) · `scripts/check_docs.py`

### Scripts
`make_motivation_native_manifest.py` · `make_general_audit_manifest.py` · `run_motivation_acceptance.py` · `general_audit_20260712/analyze_budget.py`

### Docs
`SERVER_CONTEXT_…_gate.md` (rev 3) · `SERVER_PLAN_motivation_native_run.md` · `SERVER_STATUS_…_launch_readiness.md` · `SERVER_RESULTS_motivation_native_run.md` · `SERVER_PLAN_general_method_audit.md` (**rev 6**) · `SERVER_RESULTS_general_method_audit.md` · `SERVER_AUDIT_REQUEST_motivation_results.md`

### Verifiable artifacts
```
outputs/acceptance_20260711/acceptance_report.json                     8/8 PASS
motivation_native_20260711/analysis/aggregate_main.json                864 paired cells
motivation_native_20260711/analysis/resolution_check.json              0/288 flips
general_audit_20260712/gates/G1_no_op_proof.json                       6/6 bit-identical
general_audit_20260712/gates/G2_dataset_identity.json                  144/144 hashes
general_audit_20260712/analysis/budget_verdict_clean.json              72 cells, 4.2%
<run>/PROVENANCE.txt                                                   frozen commit + tree state
```

**Gates now:** `make test` **105 OK** · `make docs-check` **PASS**

---

## 6. Mistakes I made (so you can check I actually fixed them)

1. **K lattice** — claimed a fixed 25-unit lattice from one population; it's `{0.1, 0.3}·K_base`. Would have broken 7/9 populations.
2. **PLUS bank over `K`** — unidentifiable; made PLUS ≡ MOOR byte-for-byte. Retracted in the context doc §3.
3. **Nearest-bin transition** (codex's build, my catch) — artificial fixed points; the model thought extinction was unreachable.
4. **Cost model** — 1.38× under overall, and **qualitatively wrong for OGSRL** (its cost is training-bound, not rollout-bound). The canary caught it.
5. **Worst-task sizing** — I wrote "size against the worst task, not the mean" into the plan, then packed refplan at `BS=11` by the mean anyway. A task held 11 expensive rows ≈ 5h.
6. **`path.exists()` ≠ complete** — a `scancel` left a valid-but-truncated summary; my resume check skipped it.
7. **Verdict contamination** — partial cells from the cancelled sweep biased the result **toward my own conclusion**. Caught and gated out; the corrected number is *less* favourable to my hypothesis and the conclusion still holds.

---

## 7. Where this leaves the paper

Option **(c)** — "the generals are just under-tuned" — is **closed. They are not.** Two remain, and it is the PI's call:

- **(a)** Report the negative result and reframe.
- **(b)** Redesign the environment so it stops handing solvers exact per-action growth rates (breaks comparability with every prior audited run).

I have not tuned anything toward the desired answer, and I am not recommending which framing to take.
