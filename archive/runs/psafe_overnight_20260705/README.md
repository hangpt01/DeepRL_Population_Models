# P_safe overnight run — 2026-07-05

Autonomous overnight run set up by Claude while the user slept. Goal: produce the
full real-ecology reward-mode benchmark **and** the data needed to lock
`collapse_penalty` (P_safe) under the new per-step occupancy safety penalty.

## Key decisions (data-backed)

- **All CPU.** Measured per-row wall at the real_experiment budget: delphic 8s,
  moor 13s, mopo 46s, refplan 52s, PLUS 90s, bamcts ~60s, **OGSRL ~6–8min (tall
  pole)**. PLUS on CPU is 90s, so **the GPU question is moot** — GPU adds
  allocation overhead and depends on Codex's still-in-progress PLUS-GPU code for
  zero benefit at this budget. Nothing here uses GPU.
- **Penalty is a real axis, not an auto-decision.** Because rows are cheap, the
  grid runs at `collapse_penalty ∈ {2,5,10,20}` so the full sweep exists at every
  P. The morning P_safe choice is a *read* off complete results, not a fragile
  automated decision that could waste the night.
- **Frozen code snapshot.** `code/` is an rsync snapshot of the package taken when
  36 tests passed, so Codex's concurrent PLUS/GPU edits to the live tree cannot
  corrupt in-flight jobs. Jobs run `ROOT=code/real_ecology_cont_obser`.

## Layout

- `code/` — frozen snapshot (`real_ecology_cont_obser/` + `real_ecology_data/`).
- `configs/` (inside snapshot) — `real_experiment_p{2,5,10,20}.yaml` (occupancy
  penalty mode explicit; plots on).
- `manifests/` — `manifest_full_learned*` (p10, both modes, 2592 rows, 3 shards);
  `manifest_safe_only*` (1296 safe rows, 2 shards) reused for p2/p5/p20.
- `outputs/p{2,5,10,20}/` — datasets, evaluation summaries, per-cell training
  plots, `aggregate.json`, `summary.txt`.
- `analysis/` — `all_metrics.csv`, `DECISION_psafe.md`, `convergence_p*.png`.
- `logs/` — Slurm `.out/.err`.
- `job_ids.txt` — submitted array + analysis job ids.

## What runs

9 CPU arrays on partition `comp` (%32 concurrency each), 6,480 row-runs total:
p10 full grid (safe+yield) + p2/p5/p20 safe-only. A `psafe-analyze` job runs
`afterany` all arrays: per-penalty `cli aggregate` + `summarize_real_outputs.py`,
then the cross-penalty decision table and convergence plots.

Expected wall: ~2–4h. Everything sized to finish well inside 10–12h.

## RESULT (run finished 2026-07-05 09:33, ~2.8h wall, 0 failures)

409/409 blocks COMPLETED, 6,480/6,480 rows, zero failures. Headline read from
`analysis/DECISION_psafe.md`:

- **Heuristic P_safe pick = collapse_penalty 5.** Recoverable decision-pops (Amur
  tiger, Puerto Rican parrot) keep non-negative safe return at P<=5 and get safer
  (final_state up, collapse_entry down); **P=20 over-penalises** (their safe
  returns go negative). P=10 is borderline (parrot already negative).
- **Egyptian vulture (sink):** unsafe_frac=1.0 at every P, safe return scales
  ~linearly worse with P — penalty-dominated regardless; only translocation a10
  moves it. Confirms the occupancy-penalty science.

This is an AID — the full grid exists at every P, so any final choice is already
backed by complete results.

## Morning checklist

1. `cat analysis/DECISION_psafe.md` — decision table + P_safe pick (P=5).
2. Open `analysis/convergence_p*.png` — training convergence per method (4/4).
   Per-cell curves also exist as `training_history.png` in every learned/model cell.
3. `analysis/rollup.csv` — per (penalty, reward_mode, method) means (the machine
   -readable rollup). `analysis/all_metrics.csv` — full 6,480-row tidy table.
4. `cat outputs/p*/summary.txt` — per-penalty benchmark summaries.
5. Sinks (Egyptian vulture, bottlenose dolphin) are reported separately — never
   pool them with recoverable populations.
6. Lock the chosen `collapse_penalty`, update the default + the calibration test,
   then this grid already is the full sweep for that P.

## Known issue (non-blocking)

The built-in `cli aggregate` failed on all 4 penalties with a **pre-existing bug**
in the snapshot code (`manifest.py:201`, `grouped_wins` NoneType.append) — so
`outputs/p*/aggregate.json` is absent. Fully covered by `analysis/rollup.csv` +
`analysis/all_metrics.csv` (both complete). Worth fixing in the live tree before
the next run, but it cost no data here. `summary.txt` (from a different code path)
was produced normally for all four.

## Not done (by design)

- PLUS-on-GPU: unnecessary at this budget (90s CPU) and depends on Codex's
  in-progress code. Revisit only if the budget grows.
- Pooled-agent architecture: deferred per earlier decision; this is the
  per-population-cell benchmark.
