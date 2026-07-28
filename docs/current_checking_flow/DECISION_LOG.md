# Decision log

A running record of *decisions* — what was decided, when, on what evidence, and
what would reverse it. The project has strong provenance for artifacts (hashes,
receipts, sealed predictions) but had none for decisions; this file closes that
gap. Newest entries at the top. This log records choices, not results — results
live in `NEXT_WORK_QUEUE.md` §1-OUTCOME and the receipts.

Format per entry: **Decision** · **Date** · **Evidence** · **What would reverse it**.

---

## 2026-07-27

### D-006 — S2 preregistered gate thresholds locked
- **Decision:** Confirm the five S2 thresholds — ρ* = 1.0, lnB* = ln 100 ≈ 4.6,
  δ_R = 0.10, k_cells = 2 of 4, δ_VPI = 0.10 — before S2 runs.
- **Evidence:** fox learning gain over the best constant action is +0.23…+1.56
  (S1), so 0.10 return-unit margins are a fraction of the effect in play; ρ = 1 is
  the standard "separable within one observation-noise SD" line.
- **Reverses if:** nothing — preregistered and locked. A later decision to change
  them requires a new dated entry and voids the preregistration for that run.

### D-005 — Proceed with all four post-fleet work items in parallel
- **Decision:** Record outcomes, draft the S2 spec, and relay the analysis-pipeline
  run to the code-server agent, together.
- **Evidence:** Fleet complete and clean (D-004); outcomes recorded; S2 needs no
  logs; the pipeline run is delegable to the agent.
- **Reverses if:** the agent reports the `.npz`/`.csv.gz` logs are unusable or the
  `analysis/` pipeline cannot run on the server — then the pipeline step falls
  back to a local download.

### D-004 — Accept the diagnostic replay as complete
- **Decision:** Treat the P=10 diagnostic replay as finished; move to interpretation.
- **Evidence:** 24/24 parity receipts PASS at exactly 0.0 (Tier A 10, Tier B 12,
  A6 checkpoints 2); `recomputed_fits=0`; accepted CSV `7431318803e468…` verified
  unchanged; Tier B dataset hashes matched accepted (12/12); M14 schema separation
  held. Every sealed §1.1–§1.3 prediction confirmed.
- **Reverses if:** a re-audit of the logs shows a parity field was computed against
  the wrong accepted row, or a side-effect leak is found in the shadow rollout.

### D-003 — Record the M12/OGSRL prediction miss rather than bury it
- **Decision:** Log the missed §1.4 M12 prediction openly and carry it into the
  §9 write-up.
- **Evidence:** Predicted the OGSRL constraint binds on fox / rarely on tiger.
  Actual: formal constraint never binds (bind_fraction 0.0, B1–B3); the kNN
  guardian carries safety and fires on **tiger** (override 0.995 / 0.929), not fox
  (0.0). Mechanism and species both inverted. Predicted s_low figures were also off
  (fox 15.15 vs predicted 31.90).
- **Reverses if:** the pipeline shows the bind-fraction definition used in the log
  differs from the one the prediction assumed (i.e. the miss is definitional, not
  substantive) — still recorded, but reinterpreted.

### D-002 — Open item 19 closed: PLUS averaging buys no consistent advantage
- **Decision:** Treat the PLUS posterior-averaging mechanism question as answered
  for the claim-supporting species.
- **Evidence (M1–M15 pipeline, 27 Jul):** averaging is decision-active on all four
  fox cells (`switch_vs_MAP`: A1 0.141, A2 0.709, A3 0.222, A4 0.675) but its sign
  vs MOOR is **family-dependent** — harmful on Allee (A2, PLUS 10.967 < MOOR 11.611,
  −0.64), helpful on regime (A3, +0.45) and theta (A4, +0.17), neutral on the true
  Ricker family (A1, ≈0); it nets ≈0 across fox. So the extra machinery yields no
  consistent, bankable advantage over a single misspecified model — not a uniform
  harm. (Supersedes the earlier "harmful where it engages" reading, which
  over-generalized the A2 loss.)
- **Reverses if:** S6 replication shows the fox orderings (margin 0.0005) do not
  survive independent fit/collection seeds — then the per-cell signs are not
  established and D-002 weakens to "averaging engages; per-family direction
  unresolved."

### D-001 — Launch the full Tier-A fleet, no trim
- **Decision:** Full 10-cell Tier-A array + Tier B, no cut-protection trim.
- **Evidence:** A6-PLUS checkpoint PASS 0.0 on capture-path code; all sealed §1.1
  predictions confirmed at the checkpoint; manifest confirmed the revised cell list
  (A2 = fox/Allee/0.2) was queued, so the load-bearing A2 test would run at full
  strength.
- **Reverses if:** (moot — fleet already complete) would have reversed on a
  checkpoint parity failure.

---

## Open decisions still owed to Hang (from taxonomy §10)

Recorded here so they are not lost; they are **not yet decided**.

1. **[Blocking] Planner/evaluator objective mismatch** — keep-and-reframe (a),
   symmetric threshold-aware reward (b), or enrich the surrogate basis (c). Gates
   the reward axis (S9) and the family-uncertainty gate.
2. **Primary scientific claim** — "ecological structure helps" vs "latent-state
   modelling helps" vs "family uncertainty is not the binding constraint."
   Recommendation: let S2 and S5 choose it.
3. **Fully-observed PLUS/MOOR convention** (Variant A point-mass vs Variant B MDP).
4. **Whether "oracle dynamics" may use the true non-Ricker family.**
5. **Replication budget split** — more seeds on fewer cells (recommended) vs more
   cells at one seed.
6. **Reward axis timing** — now (coupled to blocking decision 1) or later.
7. **Environment redesign** (§6.5) if S1/S2 confirm degeneracy — preregister the
   ecological justification before seeing which redesign favours which method.
