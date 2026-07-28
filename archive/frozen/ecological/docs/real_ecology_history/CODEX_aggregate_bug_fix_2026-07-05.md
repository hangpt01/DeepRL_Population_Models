# Codex Aggregate Bug Fix

Date: 2026-07-05

Purpose: document the live package fix for the `cli aggregate` crash that showed up during the P_safe overnight analysis.

## Why It Was Not Fixed In The Prior Cleanup

The previous Codex pass was intentionally scoped to no-compute analysis cleanup over the completed run artifacts. I left live package code changes out of that pass so Claude could audit the analysis artifact changes separately from implementation changes.

This fix is now applied in the live package.

## Bug

`real_ecology_benchmark.cli aggregate` crashed inside:

- `src/real_ecology_benchmark/manifest.py`

The failing line was:

```python
grouped_wins.setdefault(
    (backend, reward_mode, method, sigma, filter_name)
).append(win)
```

Because no default list was passed to `setdefault`, the first missing key returned `None`, then `.append(win)` raised:

```text
AttributeError: 'NoneType' object has no attribute 'append'
```

This affected aggregate outputs with paired episode rows that reach the `beats_both_rate` computation.

## Fix

Changed the call to provide an empty list default:

```python
grouped_wins.setdefault(
    (backend, reward_mode, method, sigma, filter_name), []
).append(win)
```

## Regression Test Added

Added:

- `tests/test_real_ecology.py::TestComputeBackend::test_aggregate_computes_beats_both_rate`

The test creates a tiny synthetic `episodes.csv` tree with:

- one recoverable population: Amur tiger
- baselines: `plus`, `moor`
- general methods: `mopo`, `refplan`
- one seed and matched scenario metadata

It asserts:

- aggregate no longer crashes
- `mopo` beats both baselines with rate `1.0`
- `refplan` does not beat both baselines with rate `0.0`
- the aggregate JSON is written

## Verification

Targeted tests:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest \
  tests.test_real_ecology.TestComputeBackend.test_aggregate_computes_beats_both_rate \
  tests.test_real_ecology.TestComputeBackend.test_aggregate_separates_backends -v
```

Result: 2 tests passed.

Full real-ecology test suite:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: 37 tests passed.

Live aggregate smoke against the frozen P10 evaluation tree:

```bash
cd discrete_action_cont_obser/real_ecology_cont_obser
PYTHONPATH=src python -m real_ecology_benchmark.cli aggregate \
  --root ../real_ecology_runs/psafe_overnight_20260705/outputs/p10/evaluation \
  --output /tmp/live_p10_aggregate_check.json
```

Result: command completed successfully and wrote aggregate JSON.

## Claude Audit Checklist

Please audit:

1. The `setdefault(..., [])` fix is the minimal correct fix for the crash.
2. The regression test genuinely covers the paired `beats_both_rate` path.
3. The aggregate smoke path points at the intended frozen P10 evaluation tree.
4. No experiment outputs were modified by this live package bug fix, except the temporary `/tmp/live_p10_aggregate_check.json` smoke output.

