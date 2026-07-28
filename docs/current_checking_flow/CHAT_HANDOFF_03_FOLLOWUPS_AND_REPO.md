# Chat handoff 03 — post-fleet diagnostics, followups, and repo cleanup

Written 28 July 2026 (Australia/Melbourne), to continue in a fresh chat. Supersedes
the live-state parts of `CHAT_HANDOFF_02`; the conventions and guardrails there
still hold. The accepted 144-cell result is unchanged and remains controlling.

---

## 0. How to start the next chat

The new chat has the same folder mounted. Send only:

```text
Read 00_START_HERE.md, then CHAT_HANDOFF_03_FOLLOWUPS_AND_REPO.md completely, then
NEXT_WORK_QUEUE.md §1-OUTCOME and DECISION_LOG.md. Do not skim. Then tell me the
live job state, what still needs collecting, and the pending decisions — and wait
for me before analysing or writing anything.
```

Reading order: `00_START_HERE.md` (§2, updated) → this file → `NEXT_WORK_QUEUE.md`
§1-OUTCOME (the fleet scorecard) → `DECISION_LOG.md` → `RUN_PLAN_AND_COMPUTE_BUDGET.md`
→ `S2_CROSS_FAMILY_REGRET_SPEC.md`. For code-server job state, read the latest
receipts and `HANDOFF_CONTINUATION.md` under the followups run dir.

---

## 1. Live state — in flight as of this handoff

Everything relayed to the code-server agent (Codex) via copy-paste blocks; I have no
direct cluster shell. All new work lives under
`three_species_diagnostic_followups_20260727_v1/`; accepted artifacts untouched.

| Item | Status |
|---|---|
| **S2** (family regret/VPI) | Finished; grid-convergence re-run through 321 bins requested. **Collect the converged fox VPI + R[Ricker,·] at 41/81/161/321 bins** — decides if the family-uncertainty motivation survives |
| **Reward screening** (Job 2) | **Done, clean.** Option A moves vulture winners off harvest, moves tiger off a10; surrogate R² 0.42–0.99. Green-lights S9 when wanted |
| **H14** re-instrument (Job 3) | 9/9 arms exact parity; **collect the full M13 read.** Early signal runs AGAINST "safe by accident" (tiger visited-state surrogate error large +ve, RefPlan ~+3.0) |
| **S6 fit probe** (Job 4) | PLUS fit finished but MOOR never ran + no receipt. Agent instructed to salvage PLUS timing, run MOOR, write receipt — **collect the FIT/PLAN/EVAL cost** (prices the full S6 grid) |
| **H12 dose-response** (Job 5) | m=1.0 gate passed 12/12 at 0.0; 48 MOOR arms done; **48 PLUS arms failed on a false cache-miss (tuple-vs-list) assertion** — agent instructed to fix as a TYPE bug only (keep recomputed_fits=0), rerun the 48 PLUS arms (~204 core-h), aggregate |
| **Repo cleanup** | `CODE_REPO_CLEANUP_COMMAND.md` relayed: non-destructive, copy-only, Phase-1 audit STOPS for review. **Expect `REPO_AUDIT.md` + a KEEP/ARCHIVE/EXCLUDE manifest to approve** |

**Immediate next actions for the new chat:** (1) collect converged S2 numbers and
decide whether fox VPI clears δ_VPI=0.10 above the solver's numerical floor;
(2) collect the H12 PLUS rerun + S6 receipt + full H14 M13; (3) review the repo
Phase-1 audit and approve the manifest; (4) fold all final numbers into the docs.

---

## 2. What changed since handoff 02

No accepted number changed. The interpretation was refined by the M1–M15 pipeline
and the followups:

1. **PLUS averaging is family-dependent, not uniformly harmful (correction).** It is
   decision-active on all four fox cells (switch_vs_MAP 0.14–0.71) but its sign vs
   MOOR depends on family: harmful on Allee (−0.64), helpful on regime (+0.45) and
   theta (+0.17), neutral on Ricker; nets ≈0. Reading: the extra machinery buys no
   consistent advantage over a single misspecified model. (An earlier "harmful"
   over-generalization from the A2 cell was corrected across all docs.)
2. **A5 is the clean demonstration:** tiger candidates never unanimous (M3=0),
   disagree hugely (centred 2.657), yet switch=0 and SD=0 — r_pos=0 deletes the term
   they disagree about.
3. **M12/OGSRL sealed prediction missed** (recorded, not buried): the constraint
   never binds; the guardian carries safety and fires on tiger (0.995/0.929), not
   fox — mechanism and species both inverted. Likely why OGSRL wins the tiger cells.
4. **S2 (provisional):** fox families DO induce different optimal policies (regret
   −0.45…−0.55, 11/16 cells over δ_R; tiger flat) — motivation preliminarily alive,
   but fox VPI 0.112 sat close to the gate and the 41-bin solver had a numerical
   floor (~0.006), hence the convergence re-run to confirm.
5. **Reward screening:** a graded reward (Option A) would change behaviour — S9 is
   justified when wanted.

---

## 3. New documents this session (index)

| File | What it is |
|---|---|
| `NEXT_WORK_QUEUE.md` §1-OUTCOME | The fleet + pipeline scorecard vs the sealed §1 predictions (sealed part untouched) |
| `DECISION_LOG.md` | Dated decisions (D-001…D-006) with evidence + reversal conditions; plus the open decisions owed |
| `RUN_PLAN_AND_COMPUTE_BUDGET.md` | Remaining experiments (S2/S4/S5/S6/S7/S8/S9 + re-instrument), spec status, measured compute, account capacity, wall times |
| `S2_CROSS_FAMILY_REGRET_SPEC.md` | The S2 spec; thresholds confirmed (ρ*=1.0, lnB*≈4.6, δ_R=0.10, k=2/4, δ_VPI=0.10) |
| `TONIGHT_RUN_BATCH.md` | The overnight batch + Block 0 consolidated relay (S2, screening, H14, S6 probe, H12) |
| `AGENT_CONTEXT_AND_BUILD_AUTHORIZATION.md` | Authorized the agent to BUILD the five (new experiments, no launchers), with per-job validation anchors + the Job-2 state-gap fix |
| `CODE_REPO_CLEANUP_COMMAND.md` | The non-destructive, retention-biased repo reorganization command (CORE list, KEEP/ARCHIVE/EXCLUDE, reproduce-to-1e-9 verify) |

Controlling/method/audit docs and the taxonomy are as indexed in `00_START_HERE.md`.

---

## 4. Pending decisions (nothing decided yet)

From taxonomy §10 and `DECISION_LOG.md`:
- **[Blocking] Planner/evaluator objective mismatch** — keep-and-reframe / symmetric
  threshold-aware reward / enrich the surrogate. Coupled to the reward axis (S9),
  now green-lit by the screening.
- Primary scientific claim (let S2 + S5 choose it).
- Fully-observed PLUS/MOOR convention; whether oracle dynamics may use the true
  non-Ricker family; replication budget split; environment redesign (§6.5).
- **Repo:** license, repo/package name, final clean-repo path, and which docs are
  marked public vs internal in `docs/INDEX.md` — surfaced at the Phase-1 stop.

---

## 5. Conventions & guardrails (unchanged — see handoff 02 §6, §8)

Relay copy-paste blocks to the code-server agent; predictions sealed before results;
every re-run carries a parity/anchor gate; accepted CSV (sha 7431318…) and receipts
never modified; constant-action policies are evaluator-only references; P=5 never
enters the P=10 table; "adapted" PLUS/MOOR vs "inspired" general vs "Delphic-
motivated" EVD; species-specific K_ref/s_safe (250/25, 41/10.25, 325/81.25). The
agent builds new code when authorized, but every new implementation needs a
validation anchor before its numbers are trusted (the S2 oracle especially).
