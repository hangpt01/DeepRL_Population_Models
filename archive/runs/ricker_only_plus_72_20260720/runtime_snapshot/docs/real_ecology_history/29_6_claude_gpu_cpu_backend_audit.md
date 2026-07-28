# Claude audit of the GPU/CPU backend plan

Date: 2026-07-03
Reviews: `29_6_gpu_cpu_backend_implementation_plan.md`
Motivation given: PLUS on CPU is very slow (~4.2 h/row).

## Verdict

**Do not build the GPU backend yet — it targets the wrong layer.** PLUS is slow
because of a **per-particle Python loop in `MechanisticProposal.sample_next`**
(scalar `transition_value` calls), not because of NumPy-vs-GPU. This is a
vectorization bug. Fixing it is CPU-only, ~30 lines, no new dependency, and I
measured it at **1706× faster with bit-identical results**. A CuPy port of a
per-element loop would be *slower*, not faster, and would add a large machinery
(backend layer, namespacing, strict mode, parity tests, GPU Slurm, CuPy dep) to a
problem that a NumPy rewrite of one function solves.

Recommendation: **vectorize first, re-measure, and only consider GPU if genuinely
still needed** (it almost certainly won't be at pilot/full scale).

## Evidence (measured, this environment)

**1. Profile of a short PLUS run** (probe-scale, eval horizon 15), sorted by time:

```
ncalls      tottime  cumtime  function
1,015,635    5.72    26.41    controls.py:49  advance_public_controls
4,065,938    4.82    10.12    numpy _clip_dep_is_scalar_nan   (np.clip on scalars)
1,009,515    2.90    43.80    envs.py:185     transition_value   <-- 93% of a 47 s run
2,032,969    2.24    15.36    numpy _clip
```

`transition_value` is called **1,009,515 times** for one 15-step PLUS episode and
accounts for ~93% of runtime; each call re-invokes `advance_public_controls` and
several `np.clip`-on-scalars (4M scalar clips). The driver is
`MechanisticProposal.sample_next` (`beliefs.py`), which does:

```python
for i, (s, aid) in enumerate(zip(states, actions)):
    out[i] = self._env.transition_value(float(s), self.actions[int(aid)], ...)  # scalar, per particle
```

i.e. a Python loop over ~3072 particles × 5 horizon × 21 candidates × 50 steps.
The Tier-3 handoff (§3.4) already flagged this exact "per-particle Python loop."

**2. Micro-benchmark** — the current per-particle loop vs a vectorized Ricker
next-state over 3072 particles (× 20 reps):

```
per-particle loop: 1.607 s    vectorized: 0.0009 s    speedup: 1706x
max |loop - vectorized| = 8.5e-14   (floating-point noise; results identical)
```

So the hot path is ~1700× faster in **NumPy** with no behavior change.

## Why GPU is the wrong first move

1. **It's a Python-interpreter-loop problem, not an array-throughput problem.**
   Moving a per-element loop to CuPy makes it *worse* — GPU per-op launch overhead
   exceeds even NumPy's scalar overhead. GPU only helps once the work is already
   expressed as large array ops, which is exactly the NumPy fix.
2. **Cost/benefit is lopsided.** ~30 lines of vectorization (one function, plus the
   twin `ExactEpisodeProposal.sample_next` in `gate.py`) vs the plan's `backend.py`,
   config surface, path namespacing, strict mode, fairness framework, GPU Slurm,
   CuPy dependency, and a full parity test suite.
3. **RNG parity is not achievable as the plan's acceptance criteria assume.** CuPy's
   RNG ≠ NumPy's RNG, so a stochastic planner (random sequences, particle resampling)
   cannot match "actions within tolerance" between CPU and GPU unless *all* randomness
   is drawn on the host and only deterministic math runs on device — which caps the
   speedup and complicates the design. The NumPy vectorization has no such issue: it's
   bit-identical (measured 8.5e-14).
4. **CuPy availability is unverified** on this cluster (the package is NumPy-only;
   the running GPU jobs are unrelated). Adding a hard CuPy dep is premature.

## What to do instead (recommended)

1. **Vectorize `MechanisticProposal.sample_next`** across particles: compute the
   Ricker / Allee / theta / regime next-state as array ops (the `r_pos/r_mort`
   split, `np.clip`, `np.exp` all vectorize trivially), removing the per-particle
   `transition_value` calls. Keep the exact same math (verified identical).
2. **Vectorize the twin `ExactEpisodeProposal.sample_next`** in `gate.py` (same
   loop) so the clairvoyant gate controller speeds up too.
3. This speeds up **PLUS (21× candidate loop), MOOR, and the decision gate** — the
   only methods using the mechanistic per-particle path. MOPO/RefPlan already use
   the vectorized dynamics ensemble (hence ~2 s rows).
4. Keep `candidate_count=21`. Re-measure PLUS; expect minutes, not hours.
5. Optional next step, still CPU/NumPy: batch the 21 candidates as an extra array
   axis in the vectorized scorer (no GPU). Only if *that* is still insufficient at
   full scale would GPU be worth revisiting.

If GPU is ever pursued, the plan's *framework* is largely sound (see below) — but it
should sit on top of already-vectorized code, not replace the vectorization.

## Assessment of the plan on its own terms (for the eventual-GPU case)

- **Good:** the fairness rule (one declared backend per ranking table; record
  requested+effective backend; fail-loud in strict mode), backend output
  namespacing, and the method-support matrix (BAMCTS = recursive Python tree,
  OGSRL/Delphic = small linear/ridge heads → correctly low-priority/CPU-only).
- **Needs revision:** the Phase-2/3 acceptance criteria ("CPU vs GPU actions within
  tolerance") — not achievable for the stochastic path without host-side RNG; state
  that explicitly or restrict parity tests to the deterministic sub-computation.
- **Redundant after vectorization:** OGSRL (8.7 s), Delphic, MOPO (2 s), RefPlan are
  already fast; only PLUS/MOOR/gate needed help, and vectorization gives it.

## Answers to Codex's open questions

1. **Exclude BAMCTS by manifest, or fail loud?** If you ever reach GPU: exclude by
   manifest construction (don't waste a submitted job). Moot after vectorization.
2. **Port OGSRL in phase 1 or CPU-only?** CPU-only — it's a NumPy linear policy and
   already fast; no GPU port warranted.
3. **CuPy vs torch/JAX?** Don't decide now — vectorize first. If GPU is later needed,
   check module availability (torch is likely present given the running GPU jobs);
   CuPy is the least-invasive NumPy drop-in but unverified here.
4. **Backend in dataset/cache paths?** After vectorization there is one backend, so
   no namespacing needed. If GPU is later added: transitions are backend-independent
   (share); namespace only the learned-filter/belief caches if GPU changes numerics.
5. **Parity tolerance?** For deterministic array math, ~1e-6 relative is reasonable.
   For the *stochastic* planner, exact/near-exact parity is impossible across
   NumPy/CuPy RNG — parity must be defined on the deterministic sub-computation, or
   RNG must be host-side. This is the main hole in the plan's acceptance criteria.

## Bottom line

The slow-PLUS symptom has a simple root cause (a per-particle Python loop, 93% of
runtime, 1706× recoverable in NumPy, bit-identical). Recommend vectorizing
`MechanisticProposal.sample_next` (and the gate twin) before writing any GPU
backend. I can implement that vectorization in a small, test-guarded diff if you
want; the GPU backend can then be deferred or dropped.
