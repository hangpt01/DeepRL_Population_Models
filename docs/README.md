# Documentation

Current benchmark reference material lives in `benchmark/`. Start there for the
problem setting, real data tables, algorithm adaptations, run protocol, recent
results, and limitations.

The other subdirectories are provenance or supporting material:

- `history/` contains dated plans, audits, handoffs, implementation notes, and
  retired current-reference docs.
- `real_ecology_history/` preserves the real-ecology implementation/audit trail
  that informed the current bundle.
- `references/` contains external paper PDFs and literature notes.
- `claude_build_report/` preserves the retained report from the earlier
  discretized-state build.
- `plots_selected/` contains selected legacy figures.

Use `make docs` to compile the benchmark TeX files and `make docs-check` to run
the documentation lint.
