# Run Analysis Report

I've verified the run end-to-end. Here's the analysis.

> **Update 2026-06-14 — configured for a half-day final run.** Tuning + freeze are
> now COMPLETE and the final comparison has been set up to run in **~½ day** (not
> the ~1 day first estimated): `stress_pomdp_116_after_tune.sh` was edited to keep
> **both reward modes** (`collapse_sensitive` + `base`) at **50 eval episodes** on
> the **6-GPU `rtq`** queue. Full timing breakdown + the exact edits are in
> `137_claude_final_benchmark_halfday_config.md`. The plan section below reflects
> this.

**The 1-day run is well-built — but it's tuning only, not the experiment.**
The engineering is sound, and I confirmed the things that actually matter:

* **Vectorized MOPO planner works** — `test_pessimistic_planner_vectorized.py` passes (codex never managed to run it), it's the default (`use_vectorized: true` with a safe fallback), and it's clearly effective (heaviest MOPO tasks finish in ~2h instead of timing out). This was my biggest technical worry and it's resolved.
* **All six gates passed**, with stronger decision-relevance for the 10-action cells (reward-gap 5.9 / 7.1 / 2.8 for allee/regime/theta-10a vs 2.5 / 2.9 / 0.64 for 5a). The 10-action axis is the more promising one.
* **Freeze-only really is freeze-only** — no sbatch/final submission in it; it won't launch anything heavy.
* **Tuning + freeze are now COMPLETE** — array `56433936` finished all tasks (last 2026-06-14 00:23), and freeze job `56433939` wrote the frozen configs to `outputs/stress_pomdp_116/frozen/phase116_frozen_1day.{env,json}` (+ `phase116_tuning_scores_1day.csv`). The final comparison can launch immediately.

But the decisive finding: **the 1-day manifest is 108 MOPO + 108 RefPlan and zero PLUS/MOOR, and the dependent job only freezes configs.** So when this finishes you'll have tuned MOPO/RefPlan configs + their tuning-seed scores — and no head-to-head against the baselines, no final eval, no report. By itself it does not answer "do RefPlan/MOPO beat PLUS/MOOR," and tuning-seed numbers shouldn't be presented as the result (that's selecting and reporting on the same seeds). The reduction to one day cut out exactly the comparison phase.

That's a reasonable phasing decision (tune first, inspect, then run final) — as long as you know a second phase is still required.

## The final comparison — configured for a half-day run (ready to launch)

* **Run the final comparison with the frozen configs** — this is the actual benchmark: all four methods (MOPO, RefPlan, PLUS, MOOR) on **held-out seeds 7001–7005**, **both** reward modes (`collapse_sensitive` + `base`), then aggregate the `.tex`/CSV report. `stress_pomdp_116_after_tune.sh` (freeze → dataset → method → report) + `aggregate_stress116_phase116.py` already exist; **the script has now been edited to fit a ½-day budget** (it was ~1.5 days as originally written — both reward modes × 100 eps × only 4 GPUs). The changes:
    * **50 eval episodes** (was 100) — `FINAL_EVAL_EPISODES=50`. ×5 held-out seeds = 250 eps/cell aggregate, plenty for mean±SE. This is the dominant cost lever.
    * **6-GPU `rtq` queue** — `FINAL_QOS=rtq`, `FINAL_METHOD_CONCURRENCY=6`, and `--qos=rtq` added to **both** sbatch arrays. **Required:** the default `normal` QOS caps a user at 4 GPUs, so `%6` is moot without it. `rtq` = 6 GPU / 2-day wall; the `gpu` partition (where the L40S nodes live) allows it.
    * **Both reward modes kept** — `FINAL_REWARD_MODES=collapse_sensitive,base`.
    * **`_1day` manifest fix** — defaults now point at the manifest/frozen names the reduced tuning run actually produced (`stress116_tune_methods_1day.tsv`, `phase116_frozen_1day.{env,json}`); the selector silently skips rows with no eval CSV, so the old full-grid default would have selected on only the run subset implicitly.
    * **Expected wall ≈ 10–11 h.** Per the completed tuning metadata for the *frozen* configs, **RefPlan is the bottleneck, not MOPO**: frozen RefPlan (`ref_h8_n512_l12_k10p0_u0p2`) ≈ **54 min/run** (CPU-bound: 512 sequences × 12 latents × horizon 8), while frozen MOPO (`mopo_h5_r25_lam0p5_ep50`, vectorized) ≈ **2.6 min/run**; PLUS/MOOR are analytical minutes; the `dataset` stage is just `generate_dataset.py`, no gate. So 60 RefPlan × 54 min ÷ 6 ≈ 9.1 h drives the wall, + the cheap methods/datasets ≈ ~10–11 h total — within half a day but tighter than first stated. (Halving episodes 100→50 still halves RefPlan, so it remains the right lever; dropping `base` → ~5.5 h.) Launch (tuning + freeze are done, so no dependency needed):
      ```bash
      cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/claude_build
      BASE=$PWD bash scripts/slurm/stress_pomdp_116_after_tune.sh
      ```
* **Report on the held-out seeds, not the tuning seeds.** The freeze step picks configs on 1160–1162; the comparison uses fresh seeds 7001–7005 — keep it that way or it's mildly optimistic for the learned methods.
* **Base reward mode is included** in the ½-day plan (the secondary/sanity story); `collapse_sensitive` is the headline (where dynamics modeling matters for avoiding collapse). To shave to ~5 h, drop `base` (`FINAL_REWARD_MODES=collapse_sensitive`).

## Minor things to watch

* **theta-5a is barely decision-relevant** (gate reward-gap 0.64) — don't expect a meaningful learned-vs-baseline gap there; it's the diagnostic control. theta-10a (2.79) is real. Lead your presentation with allee/regime and the 10-action cells.
* **Tuning afterok is moot now** — all 108 tuning tasks finished COMPLETED and the freeze job fired, so there are no partial-results to handle from this phase. (The same `afterok` fragility now applies one level down, inside the final phase — see below.)
* **GPU quota:** the earlier contender (`56445062 fedcmoo`) was CANCELLED, so your GPU quota is free. The final phase runs under `rtq` (6 GPU); if `rtq` is congested on the day you can fall back to `FINAL_QOS=normal FINAL_METHOD_CONCURRENCY=4` (≈14 h) or trim eps/modes further.
* **afterok still applies inside the final phase:** `after_tune.sh` chains dataset → method → report with `afterok`. If any method-array task fails/times out, the report won't auto-fire — `sacct` the method array for non-COMPLETED tasks before reading results.