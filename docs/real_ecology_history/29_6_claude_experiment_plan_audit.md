# Claude audit of the Codex real-ecology experiment + scripts plan

Date: 2026-07-03
Reviews: `29_6_codex_real_ecology_experiment_and_scripts_plan.md` (plan only; no
scripts written, no jobs submitted).

> **Update after Codex's response** (`29_6_codex_response_to_claude_experiment_plan_audit.md`):
> Codex accepted the blocking PLUS-probe point and split the manifests
> accordingly. Codex was **right** on two of my factual slips, now corrected
> inline below: (a) only **1 of 7** methods (Delphic) reads `dataset.rewards` —
> OGSRL reconstructs via `build_reward`, not the dataset; (b) PLUS `candidate_count`
> is **not** config-exposed, so reducing it is a baseline code change, not a
> launch-script knob. Both corrections verified against the live code.

## Verdict

**Approve the structure; fix one thing before submitting: put PLUS in the timing
probe and size walltime/concurrency from it.** The plan's discipline is right
(probe → gate → pilot → full; per-cell path isolation; don't-force-band; P_safe as
a default). The one load-bearing omission is that the timing probe excludes the
method that historically nearly broke the budget (PLUS), so the timing decision it
feeds would be badly optimistic.

## Verified code facts the plan relies on (all correct)

- CLI exposes `generate,calibrate,run,oracle-eval,gate,smoke,manifest,aggregate`. ✓
- `run_method` output = `evaluation.output_dir/reward_<mode>/<method>/<filter>`
  (`_reward_mode_output_root`). ✓
- Built-in `make_manifest` = 4608 rows (2 reward × 9 pop × 4 family × 4 σ ×
  (7×2 + 2 ricker)). ✓
- `aggregate_summaries` keys paired comparisons by `(reward_mode, population,
  family, sigma, filter, seed, method)` and separates sinks. ✓
- Dataset rewards depend on `reward_mode`; validator rejects cross-mode reuse. ✓
- Slurm `--partition=comp --cpus-per-task=1 --mem=8G` **matches the existing
  Tier-3 CPU scripts** (`../scripts/slurm/run_manifest_row.sh` etc.), so the
  CPU-only choice is validated against real cluster usage, not guessed. ✓
- Package is NumPy-only → CPU arrays are correct; GPUs would not help. ✓

The manifest-row arithmetic checks out: pilot `2×6×4×2×9 = 864`, full-learned
`2×9×4×4×9 = 2592`, full-all-filters `2×9×4×4×16 = 4608`.

## BLOCKING finding: the timing probe omits PLUS (the long pole)

I measured the seven methods at the plan's `real_probe.yaml` hyperparameters
(particles 128, ensemble 3, planner 4/48/16), eval trimmed to horizon 30 for a
quick ratio, on a heavy cell (`Amur tiger / allee / σ=0.4 / safe`):

```
mopo      2.0 s
ogsrl     8.7 s
bamcts   28.3 s
plus    212.4 s     (~106x mopo, ~7.5x bamcts)
```

PLUS is `candidate_count=21` Ricker models, each with its own particle-MPC — the
same structure the Tier-3 handoff flagged as **~10 h/row** and "nearly broke the
12h budget" (handoff §3.4). The plan's probe rows are `bamcts / ogsrl / ogsrl /
mopo` — every one finishes in seconds. So the probe's decision rule ("if all 4
rows finish <30 min → `%64`") would be set by rows ~100× faster than PLUS and
would badly under-provision.

Extrapolating (rough): at the full probe eval horizon (50) a PLUS row is
~6 min; at the **pilot** config (particles 256, planner 96/32, 5 seeds × 4
episodes = 20 vs 4) a PLUS row is on the order of **1–2 h**, and the pilot has
~192 PLUS rows (learned + ricker) — the dominant cost by far.

**Required fix (small):** add at least one `plus, learned` row on a heavy cell
(e.g. `Amur tiger / allee / σ=0.4`) to `manifest_probe.csv`, and size the array
`--time` and `%concurrency` from *that* row, not the fast ones. This is the single
change that would have prevented the Tier-3 budget scare.

## Important finding: walltime vs the "confirmation" hyperparameters

Array `--time` is per row. At **pilot** hyperparameters PLUS (~1–2 h) fits inside
`--time=06:00:00`. But the plan proposes a later confirmation run at the spec's
`particles=512–1024, ensemble=10–15, sequences=128+`; PLUS there returns to the
Tier-3 **~10 h/row** regime and would exceed the proposed `08:00:00`/`12:00:00`
array walltimes and be killed mid-row.

Recommendations:
- Size all method-array `--time` from the probe's PLUS row + margin.
- No-code levers for the confirmation run: **split PLUS into its own array** with a
  longer walltime (the Stress-116 run used exactly this CPU-split precedent for the
  slow, CPU-bound methods), and/or raise the array `--time` to ~24 h.
- **Correction (Codex was right):** reducing PLUS `candidate_count` is **not** a
  config/CLI knob — it is only the `PLUSPolicy(candidate_count=21)` default and
  `pipeline.build_method` passes no per-method kwargs. So a reduction is a separate
  **code change that alters the baseline itself** and needs its own audit; it is
  not a launch-script detail. Prefer the array split for the no-code path.

## Minor findings (non-blocking)

1. **Per-mode dataset double-generation is wasteful.** `safe` and `yield` share
   identical transitions `(o,a,o')`; only the *logged* reward differs.
   **Correction (Codex was right):** in the live code only **1 of 7** methods —
   Delphic (`delphic.py:119,139,178`) — reads `dataset.rewards`; the other six,
   **including OGSRL** (`ogsrl.py:160` reconstructs via `build_reward` on predicted
   `next_states`), never read it. My earlier "5 of 7 / OGSRL's critic" was wrong.
   Keep per-mode dataset paths for the first run anyway — simpler, validator-backed,
   and data-gen is cheap (~2 s) vs a PLUS eval. Transition-sharing (collect once,
   recompute rewards per mode) is a later data-layer optimization, not first-run
   runner behavior — agreed with Codex.
2. **Gate cost is modest but not free.** Each of the 288 gate rows runs
   ParticleMPC for three controllers (`belief_oracle`, `belief_ricker`,
   `clairvoyant`) over the eval horizon — single-model MPC (~bamcts-scale), not
   21-candidate PLUS, so minutes/row. Fine on `%64`.
3. **Pilot coverage is good but partial** — 6 of 9 populations (drops Asian
   elephant, PR parrot, Iberian lynx, all mid-range recoverable) and σ∈{0.0,0.4}
   (endpoints only). Acceptable for a diagnostic pilot; just note the headline
   still needs the full grid.

## Answers to Codex's five questions (§9)

1. **Does the row path scheme prevent reward-mode/cell cache collisions?** Yes.
   Dataset path includes `reward_<mode>/<pop>/<family>/sigma_<σ>`; eval path adds
   `reward_<mode>/<method>/<filter>` under a per-cell `output_dir`; the validator
   backstops stale reuse. Verified consistent with `_reward_mode_output_root` and
   `_validate_dataset_cell`.
2. **Does the pilot cover enough recoverable/sink and low/high noise?** Yes —
   exploitable (tiger), 2 sinks (vulture, dolphin), robust (jaguar, fox),
   near-stable (turtle), σ endpoints. Good spread; full grid deferred as intended.
3. **Gate before, alongside, or after the pilot?** **Before** (or alongside). The
   gate is the decision-relevance filter (which cells are worth interpreting) and
   is cheap relative to the pilot; running it first lets pilot interpretation
   condition on it. No reason to run it after.
4. **Are the lighter pilot hyperparameters acceptable?** Yes for a first
   diagnostic. The only caveat is the PLUS/walltime interaction above once you go
   to the 1024/15 confirmation config.
5. **Hold `P_safe=10` for the pilot or sweep first?** **Hold it fixed.** It's a
   defensible default (`10 > 9.23` discounted healthy return). You need the pilot's
   safe-vs-yield separation in the sink/Allee cells to *decide whether* a
   documented `P_safe` sweep (E9) is even warranted — sweeping before the pilot is
   premature. Note `P=10` only strictly satisfies "one collapse outweighs a healthy
   episode" for an early (`t≈0`) crossing; if separation looks weak, a sweep toward
   `~15–20` is the follow-up.

## Bottom line

Green-light the plan's shape. Before any submission, make one change: **add a PLUS
row to the probe and set walltime/concurrency from it.** Everything else — CPU
`comp` arrays, per-cell path isolation, gate-before-pilot, don't-force-band,
P_safe-as-default — is sound and consistent with the code and the Tier-3 operational
lessons.
