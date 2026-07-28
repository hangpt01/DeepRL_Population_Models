# Response to the hidden-r/K results-report audit

Date: 2026-07-16

Reviewed inputs:

- `SERVER_PLAN_REVIEW_hide_rK.md`, especially Round 9;
- `REAL_ECOLOGY_HIDDEN_RK_EXPERIMENT_RESULTS.tex` and its generated tables;
- `manifest_full.csv`, `manifest_hidden.csv`, and the raw `summary.json` files;
- the authoritative framing in `SERVER_FIX_BRIEF_hide_rK.md` and
  `SERVER_IMPLEMENTATION_PLAN_hide_rK.md`.

## Accepted audit findings

The independently reproduced result checks are reasonable and are accepted:

- 288 matched cells per arm and zero unmatched cells;
- safe-mode best-general minus MOOR-native is `-1.060` in `full` and `+2.666`
  in `hidden`;
- the best-general envelope beats MOOR-native in 6/144 full cells and 125/144
  hidden cells;
- 12 genuine terminations occur in 576,020 hidden safe transitions, and the
  constant-risk fallback activates in 140/144 cells;
- all 11 figure references resolve in PNG and PDF, and the visualization-only
  Slurm array completed all eight tasks with exit code `0:0`;
- the known-parameter arm is retained as a valid ablation/upper-bound, the
  hidden arm is the corrected uncertainty test, and safe mode is explicitly
  reported as information-limited under the private safety objective.

No benchmark or scientific-code change is required in response to these
findings.

## Concrete report correction

Round 9 missed one factual error in the report. The TeX setting table said
there were **36 populations, 28 recoverable and 8 sinks**. The manifests contain
**9 populations, 7 recoverable and 2 sinks**. The larger numbers are
population-family combinations:

- `9 populations x 4 families = 36` population-family combinations;
- `7 recoverable x 4 families = 28` recoverable population-family combinations;
- `2 sinks x 4 families = 8` sink population-family combinations.

The report and its generator have been corrected. Population counts are now
generated from the manifest-derived recoverability map through TeX macros, so
they are not manually duplicated in the report.

## Claims not adopted

### 1. "The motivation experiment's negative result was an artifact of the r/K leak"

This is too strong and conflicts with the authoritative brief's framing. The
known-parameter result is valid for the public-demographics / structure-known
regime. It answered a different question; it was not meaningless or merely a
computational artifact.

The matched intervention supports the narrower claim that exact demographic
access materially changes the general-versus-native ordering under the current
frozen semantics. It does not justify erasing the known-parameter result or
calling every part of its outcome spurious. In addition, the historical 07-11
run used reward labels that differ from the current-semantics matched run, so
the historical result cannot be attributed solely to `expose_rk` without a
like-for-like regeneration.

### 2. "The paper's original premise is recovered"

This is not used as a report conclusion. The experiment supports the intended
hidden-demographics comparison, but two qualifications prevent the broader
claim:

- "best general" is a cellwise maximum over RefPlan, BA-MCTS, and OGSRL, not a
  single deployable general offline-MBRL policy;
- safe mode is pre-registered and empirically confirmed as information-limited:
  the public termination target is nearly inert and reward-surrogate error is
  higher in the evaluator-only penalty stratum.

The defensible statement is that the hidden-r/K repair restores a fair test of
the intended uncertainty setting and reverses the aggregate safe-mode
best-general-versus-native ordering. Broader paper-level claims require
method-level interpretation and the registered safety caveat.

### 3. "Nothing was hand-copied" and "no defects found"

These statements were accurate for the generated result tables but too broad
for the complete prose report. The incorrect population counts were manually
entered setting metadata and were not checked by the original generator. That
defect is now fixed by generating the counts from the manifests. The numerical
headline audit remains valid.

## Historical 07-11 numbers

No further report change is required now. The report already labels the 07-11
figures as historical motivation results and uses the fresh `full` and `hidden`
arms for the causal comparison. Choosing whether to regenerate the historical
presentation numbers under current semantics remains a separate bookkeeping
decision, not a blocker for this matched report.
