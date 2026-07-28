# Claude verification of the real-ecology runner + launch readiness

Date: 2026-07-03
Reviews the runner layer Codex implemented (configs, manifest generator, row/gate
runners, Slurm wrappers, summarizer) against
`29_6_codex_real_ecology_experiment_and_scripts_plan.md`.

## Verdict

**The probe, gate sweep, and the full pilot are launch-ready** — I ran the actual
scripts end-to-end (below). **One blocker remains for the *full* run only:** the two
full-**fast** arrays exceed this cluster's `MaxArraySize=1001` and will be rejected
by `sbatch`. I added `scripts/shard_manifest.py` to fix that; the pilot does not
need it.

## What I verified end-to-end (not just `bash -n`)

Generated all manifests and ran the real scripts (probe config), exit 0 each:

- `make_real_experiment_manifests.py` → counts exactly as claimed: probe 5,
  gates 288, pilot 864 (fast 672 / plus 192), full-learned 2592 (fast 2016 /
  plus 576), full-all-filters 4608 (fast 3744 / plus 864).
- **Each manifest file is re-indexed contiguously 0..N-1** (`_write` uses
  `enumerate`), so `SLURM_ARRAY_TASK_ID → index` mapping is correct for every
  split file including the `_fast`/`_plus` shards.
- `run_real_manifest_row.py MANIFEST 3` (mopo, Jaguar/regime/σ0.4) → trained,
  evaluated, wrote `summary.json`.
- `run_real_gate_row.py MANIFEST 0` → wrote gate JSON.
- `cli aggregate` + `summarize_real_outputs.py` → produced `aggregate.json` +
  `summary.txt`.
- **Path isolation is correct** (no safe/yield or cell collisions):
  - `datasets/reward_safe/jaguar/regime/sigma_0p4/public.npz`
  - `private/reward_safe/jaguar/regime/sigma_0p4/truth.npz`
  - `calibration/reward_safe/jaguar/regime/sigma_0p4.json`
  - `evaluation/jaguar/regime/sigma_0p4/reward_safe/mopo/learned/summary.json`
  - `gates/reward_safe/egyptian_vulture/ricker/sigma_0.json`
- Probe manifest includes the mandatory `plus, learned` heavy row (index 4).
- Slurm wrappers: correct `SLURM_ARRAY_TASK_ID → INDEX`, `PYTHONPATH`, BLAS
  threads pinned to 1, `--partition=comp --mem=8G --cpus-per-task=1` (matches the
  Tier-3 CPU scripts), config auto-selected by manifest name.
- Unit suite still green (16 tests), CLI smoke `status: ok`.

## Blocker (full run only): `MaxArraySize = 1001`

`scontrol show config` on this cluster reports `MaxArraySize = 1001`, i.e. the
maximum job-array index is 1000. The plan's full-fast launch commands:

```
sbatch --array=0-2015%48 ... manifest_full_learned_fast.csv     # 2016 rows -> REJECTED
sbatch --array=0-3743%48 ... manifest_full_all_filters_fast.csv # 3744 rows -> REJECTED
```

will fail at submission ("Invalid job array specification"). All other arrays fit:
probe 5, gates 288, pilot_fast 672, pilot_plus 192, full_learned_plus 576,
full_all_filters_plus 864 — all `≤1001`.

### Fix (added, non-intrusive): `scripts/shard_manifest.py`

Splits any oversized manifest into `<stem>_partNNN.csv` shards (each re-indexed
0..len-1, ≤1000 rows) and prints the exact `sbatch --array` lines. Verified on the
2016-row manifest → 3 shards (1000/1000/16), contiguous indices, all rows covered.
It does not modify Codex's generator; run it only for the two full-fast manifests
before the full run:

```
python scripts/shard_manifest.py outputs/.../manifest_full_learned_fast.csv
# submit each printed shard command
```

(Alternatively, add a `--max-array-size` option to the generator to emit shards
directly — Codex's call, since it owns that file.)

## Non-blocking notes

- **Walltime for PLUS is handled correctly.** The plan overrides `--time` to
  12–24 h for `_plus` arrays; the CLI `--time` beats the 6 h `#SBATCH` header. The
  probe's PLUS row (index 4) is still the row to size real timings from.
- **Concurrency within a cell is safe.** Multiple method tasks in the same
  `(reward_mode, pop, family, σ)` cell share the dataset/learned-filter/belief-cache
  paths; `ensure_dataset` file-locks and the proposal/cache writers use
  atomic `os.replace`, so concurrent tasks are correct (occasional redundant
  belief-cache recompute is the only waste).
- **Recommended launch order (all fit MaxArraySize):** probe (`0-4`) → size PLUS →
  gates (`0-287`) → pilot_fast (`0-671`) + pilot_plus (`0-191`) → aggregate → audit
  → (only then) full, sharding the two full-fast manifests first.

## Bottom line

Yes — it can run the **probe, gate, and pilot** via Slurm as-is (verified
end-to-end, correct isolation, correct array indexing, correct CPU/walltime
settings). The **full** run needs the two full-fast manifests sharded to ≤1000 rows
first (`shard_manifest.py` added). No jobs were submitted.
