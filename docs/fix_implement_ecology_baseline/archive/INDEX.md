# archive/ — Claude↔Codex working history (provenance only)

These files are the review-and-revision trail that produced the controlling documents one level up.
Their conclusions are already distilled into `../STATUS_AND_REMAINING_WORK.md`,
`../DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`, and `../CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md`.
Kept for provenance; **not needed to continue the work.** Safe to delete.

Chronological order:

| # | File | What it was |
|---|---|---|
| 1 | `PAPER_FAITHFUL_PLUS_MOOR_IMPLEMENTATION_PLAN.md` | **Superseded** first plan (SARSOP/DESPOT headline). Replaced by the corrected plan after SARSOP was deferred. |
| 2 | `CLAUDE_AUDIT_PAPER_FAITHFUL_PLUS_MOOR_PLAN.md` | Claude audit of plan #1. |
| 3 | `CLAUDE_AUDIT_REQUEST_PAPER_FAITHFUL_PLUS_MOOR_IMPLEMENTATION.md` | Codex's request for Claude to audit the first implementation. |
| 4 | `CLAUDE_AUDIT_PAPER_FAITHFUL_IMPLEMENTATION_AND_SMOKE.md` | Claude audit of the first (v4) implementation + smoke; found the equations fixed but 1-candidate/form. |
| 5 | `CODEX_RESPONSE_TO_CLAUDE_AUDIT_PAPER_FAITHFUL_IMPLEMENTATION_AND_SMOKE.md` | Codex response to #4. |
| 6 | `SERVER_IMPLEMENTATION_AND_SMOKE_HANDOFF_PAPER_FAITHFUL_PLUS_MOOR.md` | Codex handoff for the first implementation/smoke. |
| 7 | `EXTERNAL_REVIEW_PROMPT_PAPER_FAITHFUL_PLUS_MOOR.md` | Prompt Claude wrote to send the implementation to the external paper reviewer (no code access). |
| 8 | `RESPONSE_TO_REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR.md` | Claude's response to the reviewer, returning five points for decision. |
| 9 | `CLAUDE_DIRECTIVE_FIX_BEFORE_RESUBMIT_PLUS_MOOR.md` | Claude directive: stop, fix F1/F2, re-plan (after the reviewer confirmed two blockers + a gate violation). |
| 10 | `PROMPT_TO_CODEX_UPDATED_PLAN_CORRECTED_PLUS_MOOR.md` | Prompt to Codex to produce the corrected plan, with server-side implementability findings. |
| 11 | `CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_PLAN.md` | Claude audit of the corrected plan (→ approve, cost the sensitivity suite). |
| 12 | `CODEX_RESPONSE_TO_CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_PLAN.md` | Codex response to #11. |
| 13 | `CODEX_IMPLEMENTATION_HANDOFF_CORRECTED_PLUS_MOOR.md` | Codex handoff for the corrected implementation. |
| 14 | `CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_IMPLEMENTATION.md` | Claude audit of the corrected implementation (→ approve; M1/M2/M3). The current known-issues list in `../STATUS_AND_REMAINING_WORK.md` is drawn from here. |
| 15 | `SERVER_HANDOFF_RUN_CORRECTED_ADAPTED_PLUS_MOOR.md` | Codex handoff for the canary run. |
