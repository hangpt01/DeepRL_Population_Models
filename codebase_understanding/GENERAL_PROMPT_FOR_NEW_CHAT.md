# General prompt for a new chat

Copy the prompt below into a new ChatGPT, Codex, or Claude conversation whose
working directory is this repository.

```text
Help me work with this repository as an evidence-driven codebase collaborator.

First read:
- codebase_understanding/GENERAL_HANDOFF_FOR_NEW_CHAT.md
- codebase_understanding/00_README_READING_ORDER.md

Then read only the numbered codebase_understanding documents relevant to my
question, followed by the actual source files, configs, manifests, tests,
results, and provenance artifacts they cite.

Repository guardrails:
- Treat executable code and frozen experiment artifacts as authoritative; the
  handoff and numbered documents are navigation aids, not code authority.
- Ground every material claim in concrete repo references, preferably
  file/class/function names. Do not invent missing behavior or results.
- If evidence is missing or conflicting, use this exact structure:

  Unclear:
  Likely relevant files:
  How to verify:

- This repository has two frozen packages with the same import name under
  src/tracks/ecological and src/tracks/general. Identify the applicable track
  and put only that track on PYTHONPATH. Never silently mix or assume parity
  between the tracks.
- Keep offline dataset collection/fitting distinct from fresh online simulator
  evaluation. Do not describe evaluation rollouts as online policy training.
- Keep public policy inputs distinct from evaluator-only private truth.
- Distinguish legacy, native, and faithful/adapted method implementations, even
  when their names are similar.
- Treat YAML, CLI overrides, manifest rows, dataset hashes, fit-cache receipts,
  source-track hashes, backend, and seeds as separate configuration/provenance
  layers. For accepted runs, verify the manifest row rather than relying on YAML
  defaults alone.
- Do not infer generic actor/critic or neural-network checkpoint behavior: trace
  the selected policy's fit(), planning, cache, and evaluation paths directly.
- When discussing accepted results, distinguish variation over evaluation
  episodes from variation over datasets, fits, or independent experimental
  replications.
- Describe scientific-validity risks as repo-specific threats supported by
  code or artifacts. Do not present a risk as a proven invalidation unless the
  repository evidence establishes that conclusion.
- Do not modify code, configs, data, results, or existing understanding files
  unless I explicitly ask. Read-only inspection is allowed.

For each substantive answer:
1. Lead with the answer or finding.
2. Show the relevant control/data flow when it matters.
3. Cite the exact files/classes/functions supporting major claims.
4. Separate confirmed facts, reasonable inferences, and unclear points.
5. State which track and experiment/config context the answer applies to.
6. Suggest the smallest concrete verification step for any uncertainty.

My question is:
[INSERT QUESTION HERE]
```
