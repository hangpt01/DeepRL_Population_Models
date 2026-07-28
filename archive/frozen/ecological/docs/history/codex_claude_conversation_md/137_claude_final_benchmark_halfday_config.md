# Final stress-116 benchmark — half-day reconfig (2026-06-13)

Follow-up to `136_claude_ask_codex_modify.md`. The user asked whether the
"additional experiment" (the final 4-method comparison on frozen configs) would
fit in **half a day / three-quarter day max**. Answer + the edits I made.

## Time assessment (grounded, not the earlier hand-wave)

The final script `scripts/slurm/stress_pomdp_116_after_tune.sh` regenerates a
**240-row** method manifest (confirmed against `stress116_final_methods_template.tsv`):

- 60 MOPO + 60 RefPlan + 60 PLUS + 60 MOOR
- × **2 reward modes** (`collapse_sensitive` **and** `base`) ← doubles work
- × **100 eval episodes**, 6 env cells, held-out seeds 7001–7005

**RefPlan is the expensive method, not MOPO** (corrected 2026-06-14 from the
completed tuning metadata for the *frozen* configs — my earlier reading was
backwards; the long ~1–2 h tuning tasks were RefPlan, not MOPO). The half-day
final run executes, per frozen config:
- frozen RefPlan (`ref_h8_n512_l12_k10p0_u0p2`) ≈ **54 min/run** at 50 eps
  (~108 min at 100 eps) — CPU-bound: 512 sequences × 12 latents × horizon 8 +
  value iteration over 15 ensemble members; cost scales ~linearly with episodes.
  Rock-steady ~53–56 min across all 18 tuning cells (5a and 10a alike).
- frozen MOPO (`mopo_h5_r25_lam0p5_ep50`, vectorized planner) ≈ **2.6 min/run**.
- PLUS / MOOR are analytical minutes; the `dataset` stage is just
  `generate_dataset.py` (no gate) → cheap.

GPU QOS caps the throttle: `normal` = **4 GPU** (7-day wall), `rtq` = **6 GPU**
(2-day wall), `shortq` = **10 GPU**. The old `%4` was the `normal`-QOS ceiling,
not a choice. The `gpu` partition (where the L40S nodes live) `AllowQos=ALL`, so
`--qos=rtq` is valid and gives 6 concurrent. (RefPlan barely uses the GPU it is
allocated — a known inefficiency, not a blocker.)

**As written:** 60 RefPlan × ~108 min ÷ 4 GPU ≈ **~27 h** + data/report ≈ **~1.5 day** → does NOT fit.

| Config | RefPlan runs | eps | GPU | RefPlan wall | total |
|---|--:|--:|--:|--:|--:|
| As-is | 60 | 100 | 4 | ~27 h | ~1.5 day ❌ |
| collapse-only, 50 eps, rtq6 | 30 | 50 | 6 | ~4.5 h | ~5 h |
| **both modes, 50 eps, rtq6** (chosen) | 60 | 50 | 6 | ~9.1 h | **~10–11 h** ✅ half (tight) |
| both modes, 100 eps, rtq6 | 60 | 100 | 6 | ~18 h | ~18–19 h (¾ day) |

Biggest, scientifically-cheapest lever: **100 → 50 eval episodes** (halves RefPlan;
5 seeds × 50 = 250 eps/cell, plenty for mean±SE). `rtq` 6-GPU bump is free now (the
`fedcmoo` contention job was CANCELLED). The chosen total earlier read as "~9–10 h"
only because I had the wrong method as the bottleneck at a coincidentally similar
per-run time; the conclusion (fits half a day) is unchanged, margin is just tighter.

## User's choice

**Both reward modes (collapse_sensitive + base), 50 eval episodes, rtq 6-GPU → ~half day.**

## Edits made to `stress_pomdp_116_after_tune.sh` (claude_build; file is git-untracked)

All parametrized as overridable env vars with the new defaults:

- `FINAL_EVAL_EPISODES=50` (was hardcoded 100) — used in both manifest-gen calls.
- `FINAL_REWARD_MODES=collapse_sensitive,base` (unchanged value, now a var).
- `FINAL_QOS=rtq` — added `--qos="$FINAL_QOS"` to **both** sbatch array calls
  (dataset + method). **Required:** without it, `%6` is moot — `normal` QOS caps
  at 4 GPU.
- `FINAL_METHOD_CONCURRENCY=6` (was 4). Dataset concurrency was already 6.
- **Fixed a manifest mismatch:** defaults now point at the `*_1day` names that the
  reduced tuning run actually produced —
  `TUNE_MANIFEST=…stress116_tune_methods_1day.tsv`,
  `FROZEN_ENV/JSON=…phase116_frozen_1day.{env,json}`. The selector
  (`select_stress116_frozen.py`) silently skips rows with no eval CSV, so the old
  full-grid default would have selected on only the run subset implicitly.

`bash -n` passes.

## Prerequisites verified

- All 6 control-gap gate summaries exist and **passed=True** (allee/regime/theta ×
  5a/10a) at `outputs/stress_pomdp_116/<env>_collapse_sensitive_seed116/control_gap/`,
  so the `REQUIRE_GATE_PASS=true` method jobs won't exit-4.
- Frozen configs not produced yet — the freeze-only job `56433939` is `afterok` on
  the tuning array `56433936` (still running, ~88% done at last check).

## How to launch (do NOT run until tuning 56433936 is COMPLETED)

`after_tune.sh` re-derives the freeze itself (from the `_1day` tuning CSVs) then
submits dataset → method → report. Two options:

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/claude_build

# (a) hands-off: auto-fire after tuning finishes
sbatch --dependency=afterok:56433936 scripts/slurm/stress_pomdp_116_after_tune.sh

# (b) manual after tuning COMPLETED (lets you eyeball frozen configs first)
BASE=$PWD bash scripts/slurm/stress_pomdp_116_after_tune.sh
```

The standalone freeze-only job `56433939` becomes redundant (after_tune re-freezes
to the same `_1day` paths). Optional `scancel 56433939` to avoid a duplicate
write; harmless to leave.

## Caveats to carry into reporting

- **Report on the held-out seeds (7001–7005), not the tuning seeds 1160–1162.** The
  freeze selects on the tuning seeds; reporting on those would be optimistic for
  the learned methods.
- **theta-5a is the diagnostic control** (gate reward-gap 0.64). Lead with
  allee/regime and the 10-action cells. theta-10a (2.79) is real.
- `base` reward mode is the secondary/sanity story; `collapse_sensitive` is the
  headline (where dynamics modeling matters for avoiding collapse).
