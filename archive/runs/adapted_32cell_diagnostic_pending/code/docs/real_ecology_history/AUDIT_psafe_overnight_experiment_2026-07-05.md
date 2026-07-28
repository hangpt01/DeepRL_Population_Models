# Audit handoff — P_safe overnight experiment (2026-07-05)

Audience: Codex, to audit a completed real-ecology experiment run and advise on
further runs. Scope rule unchanged: modify only inside `discrete_action_cont_obser/`.

This documents (A) exactly what was run and where, (B) the results, (C) issues
found + fixes, (D) what to audit, (E) further possible experiments. It is a
handoff/record — do not treat its numbers as source of truth; re-derive from the
artifacts and code before asserting.

---

## A. What was run

**Goal:** lock `collapse_penalty` (P_safe) under the new per-step **occupancy**
safety penalty (Codex switched crossing→occupancy earlier), and produce the full
real-ecology reward-mode benchmark.

**Run directory (all artifacts):**
`discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705/`

- `code/` — a **frozen rsync snapshot** of the package + `real_ecology_data/`,
  taken when 36 tests passed. Jobs ran `ROOT=code/real_ecology_cont_obser` so a
  concurrently-editing agent could not corrupt in-flight jobs.
- `code/.../configs/real_experiment_p{2,5,10,20}.yaml` — per-penalty configs
  (only `environment.collapse_penalty` differs; `safety_penalty_mode: occupancy`
  explicit; plots on). Base = `real_experiment.yaml` (4000 transitions, 256
  particles, ensemble 5, planner H5×96×32, eval 5 seeds × 4 eps × H50).
- `manifests/` — `manifest_full_learned*` (2592 rows, both modes) for p10;
  `manifest_safe_only*` (1296 safe rows) reused for p2/p5/p20. Sharded into
  ≤1000-row re-indexed parts; run 16 rows/array-task (`run_block.sh`).
- `outputs/p{2,5,10,20}/` — datasets, `evaluation/.../summary.json`, per-cell
  `training_history.{json,png}`, `summary.txt`.
- `analysis/` — `all_metrics.csv` (6480 rows), `rollup.csv`, `DECISION_psafe.md`,
  `convergence_p{2,5,10,20}.png`.
- `README.md`, `submit_blocks.sh`, `run_block.sh`, `run_analysis.sh`,
  `analysis/analyze_psafe.py`, `watch_run.sh`, `job_ids.txt`, `monitor_status.txt`.

**Grid (all CPU, partition `comp`):** `collapse_penalty ∈ {2,5,10,20}` as an axis.
p10 = full grid (safe+yield); p2/p5/p20 = safe-only (penalty is inert in yield).
Coverage: 9 populations (incl. both sinks) × 4 families (ricker/allee/theta/regime)
× 4 σ_obs (0.0/0.1/0.2/0.4) × 7 methods (mopo, refplan, bamcts, plus, moor,
delphic, ogsrl) × filter fronts (learned; + ricker-mechanistic for plus & moor).

**Outcome:** 409/409 array blocks COMPLETED, **6,480/6,480 rows, 0 failures**,
~2.8h wall. Per-penalty summary counts: p10=2592, p2/p5/p20=1296. PLUS = 1,440
runs, all complete on CPU.

**Why CPU (not GPU):** measured per-row wall at this budget — OGSRL ~6–8min (tall
pole), PLUS **90s**, refplan 52s, mopo 46s, bamcts ~60s, moor 13s, delphic 8s. At
90s PLUS-on-CPU, GPU adds allocation overhead + depended on then-in-progress
PLUS-GPU code for zero benefit. GPU deliberately unused.

---

## B. Results (headline)

From `analysis/DECISION_psafe.md` (safe mode, decision populations):

- **Recoverable pops** (Amur tiger, Puerto Rican parrot): as P rises they get
  safer (final_state↑, collapse_entry↓) but return falls; **P=20 over-penalises**
  (safe true_return goes negative), P=10 borderline (parrot already negative),
  P≤5 stays non-negative. Heuristic pick = **collapse_penalty 5**.
- **Egyptian vulture (sink):** `unsafe_frac = 1.0` at every P; safe return scales
  ~linearly worse with P (−37 → −370 from P=2→20); `min_true_state` barely moves
  (1.7→2.0). Penalty-dominated regardless of P — only translocation a10 adds
  individuals. This is the intended occupancy science, **not** a bug.
- The penalty axis is fully materialised: the full sweep exists at every P, so the
  final P choice is already backed by complete results (no re-run needed to pick).

Machine-readable: `analysis/all_metrics.csv` (per-cell) and `analysis/rollup.csv`
(per penalty × reward_mode × method means).

---

## C. Issues found + fixes (during the run)

1. **`run_real_row.sh` ROOT resolution** — Slurm spools the batch script to
   `/var/spool`, so its `SCRIPT_DIR` fallback computed a bogus ROOT. Fixed by
   exporting `ROOT` explicitly in `--export`. (Caught by a 2-row canary.)
2. **Account MaxSubmitJobs=1000** — a single 1000-task array maxes it. Fixed by
   packing 16 rows/array-task (`run_block.sh`, ~409 tasks). MaxArraySize=1001.
3. **Convergence overview parsing** — `training_history.json` is long-format
   (`{metric,phase,split,step,value}`); the first analysis pass looked for wide
   keys and produced no plots. Fixed in `analyze_psafe.py`; regenerated 4/4 PNGs.
   Per-cell `training_history.png` (5041 files) were always present.
4. **`cli aggregate` is BROKEN (pre-existing bug, still in the live tree).**
   `aggregate_summaries` raises `AttributeError: 'NoneType' object has no
   attribute 'append'` at `src/real_ecology_benchmark/manifest.py:201`
   (`grouped_wins.setdefault(...)`) on this data, so `outputs/p*/aggregate.json`
   were not produced. Worked around with `analysis/rollup.csv` +
   `all_metrics.csv`. **Please fix this in the package** — see §D.

---

## D. What to audit (requests for Codex)

1. **`cli aggregate` bug (priority).** Reproduce and fix `manifest.py:201`
   `grouped_wins` NoneType — likely an unseeded dict key or a group with no
   winner. Add a regression test over a small multi-cell evaluation tree.
2. **Occupancy penalty consistency in the executed code.** Re-confirm
   `safety_penalty_indicator` (occupancy default) is applied identically in env,
   MPC (`planning.py`), BA-MCTS, fitted-Q (`value.py`), OGSRL rollouts — i.e. the
   reward the agents optimised matches the reward evaluated. The frozen snapshot
   under `code/` is the exact code that ran; audit that copy.
3. **Reward non-leak under these runs.** Verify realized reward never entered
   belief/policy inputs during the grid (dataset offline targets are fine). The
   `PublicTransition`-has-no-reward guarantee should still hold in the snapshot.
4. **Sink accounting.** Confirm Egyptian vulture + bottlenose dolphin are the only
   `r_max<0` sinks and are reported separately everywhere; nothing should pool
   them with recoverables. Sanity-check the vulture `unsafe_frac=1.0` result is a
   true-dynamics consequence, not a floor/threshold artefact.
5. **P_safe decision logic.** Audit `analyze_psafe.py`'s heuristic (largest P
   keeping recoverable decision-pops' safe return ≥0 while reducing collapse_entry
   vs P=2 → P=5). Is that the right economic criterion, or should P_safe be fit
   from a stated "one collapse outweighs a healthy episode" target (spec E6)?
6. **Data table + backend labels.** Confirm every summary records
   `data_table=…/real_ecology_data` and `compute_backend_effective=numpy` (no
   silent CPU/GPU pooling). Snapshot froze the vendored-removed, single-source
   table.
7. **Filter scope.** The run used the **learned** filter (+ ricker-mechanistic for
   plus/moor), not the `raw` filter. Confirm that is the intended headline
   front-end and that omitting `raw` is acceptable for the main result.
8. **Calibration.** Runs used `--allow-uncalibrated` (no gate artifacts). Judge
   whether the headline needs per-cell behavior-policy collapse-band gates, or
   whether the in-summary calibration metrics suffice.

---

## E. Further possible experiments (not yet run)

Ordered by likely value; none block the P_safe decision.

1. **Headline full sweep at the locked P, all filters.** Once P_safe is chosen
   (likely 5), run `manifest_full_all_filters` (4608 rows) at **that one penalty**
   to add the `raw`-filter ablation ("does the learned filter matter?"). ~1–2h CPU
   with the existing snapshot + `run_block.sh`. Cheap and paper-relevant.
2. **Calibration gates** (`manifest_gates`, 288). Produce per-cell behavior-policy
   collapse-band artifacts if the writeup needs dataset-quality QA.
3. **Finer / re-centred P_safe grid.** If P between 5 and 10 matters, add
   {3,4,6,8} on the decision cells only (cheap) to pin the recoverable
   break-even more precisely.
4. **Sink translocation study.** For the two sinks, vary translocation
   availability/cost (a10) to quantify how much repeated translocation can lift
   min-abundance under `safe` — the only lever that moves them.
5. **P_safe re-fit vs spec target.** Instead of the ≥0-return heuristic, fit
   P_safe so one true-state below-safe episode outweighs a healthy episode's
   discounted benefit (spec E6), and check consistency with P=5.
6. **Seed/CI robustness.** Current eval = 5 seeds × 4 eps. If tighter CIs are
   needed for the headline table, widen eval seeds for the chosen P only.
7. **PLUS-on-GPU (deprioritised).** Only worth it if the per-row budget grows a
   lot; at 90s CPU it is unnecessary. If Codex finishes honest method-level GPU
   PLUS, benchmark it separately and label the backend distinctly — never pool.
8. **Pooled-agent architecture (deferred).** The spec mandates one pooled agent
   over all 9 populations; this run is per-population cells. Still the largest
   open architectural item.

---

## F. Reproduce / extend

```bash
# from the frozen snapshot
cd discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705/code/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests   # sanity

# re-run analysis only (fast, reads the 6480 summaries)
PYTHONPATH=src MPLBACKEND=Agg python ../../analysis/analyze_psafe.py ../..
```

Submission pattern for a new penalty/ablation: copy `submit_blocks.sh`, point it
at the desired manifest + config, keep `ROOT` exported and rows-per-task ≤ what
keeps total tasks < 1000.
