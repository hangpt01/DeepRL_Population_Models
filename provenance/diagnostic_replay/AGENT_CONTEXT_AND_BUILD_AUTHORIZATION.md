# Context and build authorization — the five tonight jobs

Relay this to the code-server agent. It resolves the preflight stop.

---

## 0. TL;DR

Your reconnaissance is correct and expected: **no launchers exist for these five
jobs, because they are new experiments, not reruns.** You are **authorized to
construct them** by extending the existing frozen replay code, subject to the
per-job validation anchor and submission gate below. One real gap you found
(Job 2's per-step constant-policy states) is resolved in §4. `NEXT_WORK_QUEUE.md`
is a local planning doc on my side — it is *supposed* to be absent on the server;
do not look for it.

## 1. What these jobs are

They are the next experiments in the programme, specified from my side. There is no
pre-built package for any of them. Four of the five are **extensions of code you
already built**; one (S2) is a small new planner. Nothing here reruns an existing
launcher.

| Job | Build target = extend… | New code |
|---|---|---|
| 1 S2 regret/VPI | a NEW small module using `envs.py`'s `transition_value` and the true family maps | backward-induction oracle over the 41-bin grid |
| 2 reward screening | `constant_action_sweep.py` + `ContinuousEvaluator` | log per-step true states, then score under 3 reward fns |
| 3 H14 re-instrument | `run_tier_b_replay.py` | capture one field `surrogate_reward_t` |
| 4 S6 fit probe | the PLUS faithful build path (the code that produced the 216 fit-cache slots) | run it with new seeds, cache OFF |
| 5 H12 dose-response | `run_diagnostic_replay.py` PLUS/MOOR path | an `observation_scale` override knob, re-plan |

## 2. Authorization

You are authorized to write these implementations and submit them under account
`ce25`, constraint `xenon-8452Y` where parity applies, **provided** that for each
job you check its validation anchor (§3) and honour the submission gate. This is
new research code, so the anchors are how you (and I) trust the outputs — they play
the same role the 7-field parity gate played for the diagnostic replay. If a job
has no clean anchor and you are unsure, build it, run the anchor check, and report
the check **without** submitting the full run — do not invent an unvalidated result.

## 3. Per-job validation anchor + submission gate

| Job | Validation anchor (must hold) | Submission gate |
|---|---|---|
| 1 S2 | diagonal `R[F,F] = 0` exactly; every oracle's value ≥ the best constant-action value in its own family; 41- vs 81-bin argmax-flip fraction reported | cheap — build, check anchor, submit, report. Flag the oracle solver as new code so I review before the G1/G2/G3 verdict is final |
| 2 screening | under the **Current** reward, your regenerated constant-policy returns must reproduce `derived/S1_constant_action_sweep.csv` to 1e-9 (that validates the regenerated states + scorer). Only then score Options A and C | build, confirm the Current-reward reproduction, then score A/C, submit, report |
| 3 H14 | the 7 accepted fields still reproduce at `max_abs_diff = 0.0`; logging consumes no RNG (assert generator states unchanged) | parity-gated — build, confirm 0.0 on one cell, then the rest |
| 4 S6 probe | none needed (it is a timing measurement); sanity: returns land in the plausible fox range | build, submit, report FIT/PLAN/EVAL split |
| 5 H12 | the **m = 1.0** arm reproduces the accepted return at `max_abs_diff = 0.0` | **submit the m=1.0 arm ALONE first; report its parity; run the other 8 arms only if it is 0.0.** If m=1.0 ≠ accepted, HALT — bug, not result |

The H12 gate is the important one: it bounds the ~460 core-h run behind a cheap
single-arm parity check, so a plumbing bug cannot silently burn the night.

## 4. Job 2 — resolving the stored-state gap you found

You are right that `derived/` holds only aggregate S1 rows, not per-step
constant-policy states, and that the old `constant_action_sweep.py` reads accepted
values. Do **not** use it as-is. Instead:

1. **Regenerate the constant-policy trajectories fresh.** Run each of the 11
   constant actions through the unmodified `ContinuousEvaluator` on the 20
   registered seeds, logging per-step true state `x_true_t`/`s_{t+1}`, `cost`, and
   `action`. Constant policies are reward-independent, so these trajectories are
   exactly reproducible from the environment alone — no accepted value is read to
   *produce* them.
2. **Validate**, then score. Scoring those states under the **Current** reward must
   reproduce the accepted `S1_constant_action_sweep.csv` aggregates to 1e-9. That
   single check is the only place accepted values are touched, and it is a
   *parity check*, not a substitution — exactly the model the diagnostic replay
   used (accepted returns as the invariant).
3. Then score the same states under Option A and Option C and report the §Job-2
   questions.

**Guardrail clarification, since it caused the stop:** reading accepted values *for
a parity/validation check* has always been permitted and is how the replay was
gated. What is forbidden is **modifying** the accepted CSV/receipt or **presenting
accepted numbers as new results**. Regenerating trajectories and checking them
against the accepted aggregates is compliant.

## 5. Guardrails (unchanged, all five)

- Accepted `MATCHED_P10_144_METHOD_CELLS.csv` (sha `7431318803e468…`) and its
  receipt: never modified. Read only for parity/identity, never re-ranked.
- `recomputed_fits = 0` for PLUS/MOOR **except Job 4**, which fits new seeds by
  design (report it fits from scratch; it touches no accepted cell).
- `--constraint=xenon-8452Y`, threads=1, where parity or accepted-comparison
  applies (Jobs 3 and 5; Job 1's oracle and Job 2's scoring are architecture-
  independent). Job 4 needs no parity.
- Constant-action policies stay evaluator-only references; nothing is re-ranked.
- Reply `NOT FOUND` rather than inferring a path; if a job cannot meet its anchor,
  report the anchor result and do not submit the full run.

## 6. If you would rather stage it

If building all five before any submission is too much for one pass, this priority
order preserves the most value per unit build effort: **Job 4 (probe) → Job 3
(H14) → Job 5 (H12, gated on m=1.0) → Job 1 (S2) → Job 2 (screening).** Submit each
as it is built and validated rather than waiting for all five.
