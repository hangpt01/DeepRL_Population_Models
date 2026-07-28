# `fix_implement_ecology_baseline/` — index

Building **paper-aligned mechanistic PLUS/MOOR baselines** to replace the polynomial-template
`plus_native`/`moor_native`. New IDs: `plus_adapted_mechanistic_pbvi`,
`moor_adapted_ricker_misspec_pbvi`.

## Start here

- **[STATUS_AND_REMAINING_WORK.md](STATUS_AND_REMAINING_WORK.md)** — current state, measured runtime,
  known code issues, what's left, and the pre-registration guardrails. **Read this first.**

## Authoritative / controlling documents (external + settled)

| File | What it is |
|---|---|
| [ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md](ECOLOGY_BASELINE_PAPER_FAITHFULNESS_PLUS_MOOR.md) | External controlling scientific spec — the problem statement and fidelity bar. |
| [REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md](REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md) | External paper reviewer's findings (F1–F6) against the papers. |
| [DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md](DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md) | The five settled scientific decisions. **Do not reopen.** |
| [CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md](CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md) | The approved implementation plan that was actually built. |
| [paper_faithful_fit_requirements.lock](paper_faithful_fit_requirements.lock) | Pinned CPU-PyTorch environment for the fitter. |

## `archive/`

The full Claude↔Codex working history — audits, prompts, responses, superseded plans and handoffs —
consolidated out of the way. Nothing here is needed to understand or continue the work; kept only for
provenance. `archive/INDEX.md` lists the contents. Safe to delete if you don't want the history.
