# Zero-process-noise RNG evidence correction

This bounded correction implements the semantics authorized by the sealed audit
`stageb_zero_noise_rng_semantic_audit_20260812_59020454_codex_a` without changing a scientific
setting or any file under `src/tracks/**`.

The registered environment has process noise `0.0` and observation noise `0.2`. A zero process
standard deviation permits no process-noise distribution call and requires byte-identical process
generator state. Positive noise requires exactly one actual distribution invocation per evaluator
step and changed generator state. Missing draws, false draws, count/state disagreement, unexpected
draw APIs, non-finite or negative values, and registration/DRIVER_INPUTS mismatch all fail closed.

`driver.py` now wraps only the evaluator environment's process and observation generators after
each reset. The wrapper exposes the underlying bit-generator state, counts successful registered
distribution invocations, and rejects other RNG APIs. It does not count environment steps. Frozen
policy/filter RNGs remain separate and cannot satisfy evaluator receipts.

The serialized contract changes are intentionally incompatible with the former representation:

- scientific registration v3 contains a required explicit RNG contract;
- DRIVER_INPUTS v3 and its registration binding v2 contain the identical contract;
- RNG receipt, step evidence, exact-return reconstruction, paired evidence and bound task evidence
  use v2 schema identities;
- former `*_calls_*` fields are removed and replaced by `*_draw_invocations_*` plus explicit
  draw-required and state-advancement-applicability fields.

The failed scientific attempt and all prior sealed evidence remain historical and unchanged. A
future scientific execution requires a new commit-bound registration, canonical DRIVER_INPUTS,
keys, output namespace, independent audit and explicit authorization. The prior registration and
keys cannot be reused.

The protected projection/replay closure excludes this evidence layer. Carry-forward of the 36
fit/replay receipts is permitted only after exact post-commit verification of both frozen-track
aggregates, `artifacts.py`, `fit_probe.py`, `real_artifacts.py`, all 12 frozen objects, every
`FRESH_RELOAD_REPLAY.json`, and all three 12-receipt aggregates.
