# analysis/ — replay log analysis pipeline

Built 27 July 2026, **before** the fleet reported, against the log schema in
`../DIAGNOSTIC_REPLAY_RERUN_SPEC.md` §1.5.  All 32 unit tests pass on synthetic
fixtures with analytically known values.

## Run it

```bash
python3 run_analysis.py --self-test                 # 32 assertions, no data needed
python3 run_analysis.py --root /path/to/replay_root --out ./analysis_out \
                        --accepted-csv /path/to/MATCHED_P10_144_METHOD_CELLS.csv
```

`--root` is the replay root containing `logs/`.  The driver writes
`schema_report.csv` (what arrived, what is missing), `metrics.json`,
`metrics_table.csv` and the figures.

**Run `schema_report.csv` first.**  It tells you which fields the replay
actually produced.  Metrics whose inputs are absent report
`{"available": false, "reason": ...}` rather than failing, so a partial run
still yields a complete report.

## Files

| File | Purpose |
|---|---|
| `constants.py` | Registered constants from the audits: `s_safe`, `K_ref`, cost table, per-species `r_setpoint`, plus `dominated_actions()` and `max_reachable_abundance()` |
| `loader.py` | Tolerant loader for `.npz` / `.csv.gz` / `.csv`; normalises wide columns (`w_0..w_7`, `q_cand_j_a`) into arrays; reports missing fields |
| `metrics.py` | M1–M15 |
| `figures.py` | The four figures |
| `synthetic.py` | Fixtures with known metric values |
| `test_metrics.py` | 32 assertions |
| `run_analysis.py` | CLI driver |

## The two metrics that carry the argument

**M5, raw vs centred disagreement.** `raw` is cross-candidate variance of action
values; `centred` is the same after removing each candidate's own mean over
actions.  A large `raw` with a small `centred` means the candidates disagree
about the world but not about *what to do* — the cleanest explanation for a
parameter-diverse bank producing a constant policy.  The synthetic fixture
constructs `centred == 0` exactly, and the test asserts it.

**M2, paired switch fractions.** `switch_vs_MAP` isolates whether *averaging*
over candidates changes the action; `switch_vs_uniform` isolates whether
*updating* the weights does.  Both near zero means PLUS is behaviourally a
point-model method along this trajectory distribution — which explains equal
PLUS/MOOR returns mechanically, without claiming algorithmic equivalence.

## Two results the constants module reproduces independently

```
max_reachable_abundance("egyptian_vulture") = 42.937   vs s_safe = 81.25
dominated_actions("egyptian_vulture")       = {5:0, 6:0, 7:3, 8:3, 9:4}
dominated_actions("amur_tiger")             = {5:0, 6:0}
dominated_actions("crab_eating_fox")        = {}
```

Both derive from the audited action tables alone, with no reference to any
result.  **`dominated_actions` found `a8` as well, which two prose documents had
omitted — vulture has five dominated actions, not four.**

## Adapting to the real schema

The loader is deliberately permissive because the replay's serialisation is not
fully pinned.  If field names differ, edit `VECTOR_SPECS` / `MATRIX_SPECS` in
`loader.py` — the regexes are the only place naming is assumed.  If a metric
reports `available: false`, check `schema_report.csv` before changing anything.

## Pre-registered predictions

`../NEXT_WORK_QUEUE.md` §1 records what these metrics are expected to return,
sealed before the fleet reported.  Compare against it before interpreting.
