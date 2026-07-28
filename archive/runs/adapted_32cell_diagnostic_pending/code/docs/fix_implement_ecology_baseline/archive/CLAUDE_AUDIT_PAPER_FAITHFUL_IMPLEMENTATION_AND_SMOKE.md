# Audit — Paper-Faithful PLUS/MOOR Implementation + Smoke (v4, job 58351686)

**Reviewer:** Claude (independent; verified against the frozen v4 snapshot, Slurm, artifacts, tests)
**Date:** 2026-07-17
**Subject:** `CLAUDE_AUDIT_REQUEST_PAPER_FAITHFUL_PLUS_MOOR_IMPLEMENTATION.md`
**Status:** read-only. Nothing modified. Nothing under any run root was written.

---

## 0. Verdict

**The core defect is genuinely fixed.** The four "families" are now real mechanistic equations, not
polynomial bases; MOOR fits ordered episodes; the planner is real PBVI, not QMDP. Privacy holds
structurally. The smoke executed cleanly and is honestly self-labelled as *not* a headline experiment.

**No blocking defect found in what was built.** The findings below are (1) one genuine faithfulness
gap in *what was actually exercised*, (2) one missing test half, (3) one unverifiable provenance
artifact, and (4) scope limits — all but #3 already disclosed by codex.

---

## 1. Findings, ordered by severity

### F1 — Paper faithfulness — the executed smoke does not exercise PLUS's defining property

**Severity: high (for any faithfulness claim); not a code defect. Disclosed by codex (limitations 2–3).**

The smoke's bank is **exactly one candidate per form**:

```
candidate_bank.json → candidate_count: 4, prior_type: uniform,
candidate_ids: [candidate_00_00, candidate_01_00, candidate_02_00, candidate_03_00]
per-candidate forms → ricker:1, allee:1, theta:1, regime:1
```
(`…/plus_faithful_pbvi/faithful_internal/faithful_artifacts/candidate_bank.json`)

With one candidate per form and a uniform prior, the executed PLUS is *posterior over four structural
forms with fixed MAP parameters* — **zero within-family parameter uncertainty**. That is structurally
the same shape as the old `plus_native` (form-posterior only). The decisive improvement is real —
the forms are now genuine equations — but the parameter-uncertainty axis, which the controlling spec
names as defining ("Multiple parameter candidates or posterior samples within each family") and which
plan §4.2 requires ("Multiple fixed parameter candidates within every included family"), **is absent
from every run that exists.**

**Consequence:** nothing executed to date demonstrates faithful PLUS. The 4,000-row smoke and any
headline must run **≥2 candidates/form** (the full config's 16 = 4/form) before the word "faithful" is
attached to a result. Recommend the 4,000-row smoke be the first run at 16, not 4.

### F2 — Implementation correctness — the order-sensitivity test is only half-implemented

**Severity: medium. Not disclosed.**

Plan test #9 specified **both** directions:
> reverse time within each episode → objective/fit changes materially; **permute complete episodes →
> equality within numerical tolerance.**

Only the first exists — `tests/real/test_moor_faithful.py:10`
`test_ordered_objective_uses_within_episode_order` (reverses actions/next_observations per episode,
asserts SSE changes; passes). A grep across `test_moor_faithful.py`, `test_plus_faithful.py`, and
`test_faithful_*.py` finds **no permute/shuffle-episode invariance test**.

The missing half is the one that proves episodes are treated as *independent units* — i.e. that no
latent state, capacity, or accumulator leaks across an episode boundary. That is precisely the failure
mode the episode-reset requirement (plan §5.4, §9) exists to prevent, and it is currently unguarded by
test. Cheap to add; recommend before the 4,000-row smoke.

### F3 — Execution validity — the frozen tree hash is not independently verifiable

**Severity: medium (provenance), low (risk). Not disclosed.**

The request doc publishes `frozen snapshot tree SHA-256:
a7836e85690316681296dbc5fe095fc195d2d1ee8108ad5a2b5e8ec740af386f` but **not the recipe** used to
compute it. My recomputation over the v4 `code/` tree excluding `.pyc`/`__pycache__` —

```
find code -type f ! -name '*.pyc' ! -path '*__pycache__*' -print0 | sort -z \
  | xargs -0 sha256sum | sha256sum   →  478213bc8d773f23fdd199e12e24740e4797c4f058bb7fe8471c2926fcece64e
```

— does not match. **I am not calling this a mismatch:** tree-digest conventions differ (path
inclusion, sort order, name hashing), and codex almost certainly used a different one. The finding is
that *a provenance artifact whose only purpose is independent verification cannot currently be
independently verified.* Fix: publish the exact command, or list per-file hashes the way the 20260716
handoff did (those I could and did verify). By contrast, every **file-level** hash claim checked out.

### F4 — Execution validity — smoke scale establishes plumbing only

**Severity: low (properly disclosed; codex limitation 6).**

The run is 160 public rows / 10 episodes (8 fit + 2 reporting-only holdout), 17 s and 11 s, MaxRSS
≈220 MB. Nothing about fit quality, parameter recovery on real data, identifiability, or runtime
scaling is established. In particular the plan's runtime/CPU-hour questions remain entirely
unmeasured, so **no feasibility or budget conclusion may cite this run**. Codex states this plainly
and did not submit the canary or sweep — correct call.

### F5 — Paper faithfulness — bootstrap-MAP candidates vs the plan's proposal-and-rescore

**Severity: low. Disclosed (codex limitation 3, which asks for a ruling).**

**My ruling: acceptable as a preregistered variant; §12.1 is not mandatory verbatim.** The spec
requires within-family uncertainty via "a grid or posterior samples." Episode-bootstrap MAP fits are a
recognised posterior approximation, so they satisfy the *scientific* requirement provided (a) they
produce genuinely diverse candidates (codex says diversity is enforced — untested at 1/form, see F1)
and (b) the artifact and report name the method **bootstrap-MAP**, not curvature/low-discrepancy
proposal-and-rescore. Do not claim §12.1's construction until it exists.

---

## 2. Categories

### Implementation correctness — PASS (one test gap, F2)

The defect that motivated this whole workstream is fixed. Frozen
`code/src/real_ecology_benchmark/faithful_ecology.py`:

- imports **only** `dataclasses`, `hashlib`, `json`, `numpy` (lines 8–14) — no tables, no
  `EnvironmentConfig`, no evaluator, no actions;
- carries **real** mechanistic parameters: `depensation_thresholds`, `theta_exponent`,
  `regime_multipliers`, `regime_matrix` (lines 44–47), with validation that regime rows are
  probability simplexes and thresholds lie inside reset capacity (lines 78–83);
- `noiseless_next` (lines 100–138) implements genuine Ricker, Allee-Ricker with the `(m/C − 1)`
  depensation term, theta-logistic with a real exponent, and two-regime depensation indexing
  per-regime threshold and multiplier.

Compare the old `src/real_ecology_benchmark/native_fit.py:17-27`, where `ricker→[1,x]`,
`allee→[1,x,x²]`, `theta→[1,x,√x,x²]`, `regime→[1,x,x²,1(x<1)]`. **The polynomial-label defect is
genuinely repaired.**

- **Ordered fitting:** `faithful_fit.py` exposes `ordered_trajectory_sse`, `fit_episode_ids` /
  `holdout_episode_ids`, `_ordered_indices`; the reverse-time test passes.
- **Planner:** `planners/pbvi.py` — "finite-horizon backups over a seeded graph of reachable belief
  points"; capacity, survey history and time remain explicit context; explicit no-QMDP/no-`t=0`
  reduction. `_reachable_graph` builds real belief layers. This is PBVI, not QMDP.
- **Solver honesty:** `planner_invocations` = 6 (MOOR) and 24 (PLUS) in `acceptance.json` — the ID
  carries `_pbvi` and the artifact proves PBVI actually ran. No `_sarsop`/`_despot` registered; the
  binaries are genuinely absent.

### Privacy / fairness — PASS

- `faithful_ecology.py` cannot read a table: it imports nothing that can.
- Privacy tests patch table readers, `resolve_actions`, and `NativeSolver.build` to raise; fitting
  still succeeds. Perturbing sanitized public transitions changes fitted parameter hashes.
- `privacy_audit.json` scans every emitted JSON (`candidate_00x`, `candidate_bank`, `faithful_fit`,
  `planner_provenance`, `pomdp_model_00x`).
- `registration.json` records `frozen_20260716_run_modified: false`. I confirmed independently that
  no file under `hidden_rk_comparison_20260716/` was touched.

### Paper faithfulness — PARTIAL (F1 blocking for claims; F5 ruled acceptable)

Equations, ordered fitting, belief-state planning, fixed-online parameters, and uniform prior are all
in place. **What has not been demonstrated is within-family parameter uncertainty** (F1). Until a run
executes ≥2 candidates/form, "faithful PLUS" is implemented but unexercised.

### Experiment-execution validity — PASS for what it claims to be

- `sacct 58351686`: `58351686_0` and `_1` both **COMPLETED, exit 0:0**, 17 s / 11 s, MaxRSS 220,332K /
  222,940K, 2 CPUs — matches the doc exactly.
- **126 tests pass** — I re-ran `pytest -q` in `.venv-paper-faithful`; 126 passed, exit 0.
- **Dependency versions match the doc exactly**: Python 3.10.14, NumPy 2.2.6, PyYAML 6.0.3,
  **torch 2.13.0+cpu**, pytest 9.1.1, ruff 0.12.12 — all confirmed installed and consistent with
  `paper_faithful_fit_requirements.lock`.
- Manifest/registration/config/lock SHA-256 present and internally consistent; `manifest_sha256`
  matches the doc.
- All three superseded runs retained with `SUPERSEDED.md` at `paper_faithful_smoke_20260717{,_v2,_v3}`.

---

## 3. Practices worth preserving (called out deliberately)

These are better than the bar I would have set, and should survive into the headline run:

1. **`acceptance.json` records `return_fields_opened: false`** — the acceptance checker is *blinded to
   returns*. This is the blinded-canary principle actually implemented, not just promised.
2. **`registration.json` self-labels**: `run_kind: registered_interface_smoke_not_headline_experiment`,
   `headline_sweep_authorized: false`, `blinded_canary_authorized: false`. A run that refuses to be
   mistaken for a headline.
3. **Superseded jobs retained with honest reasons** — including 58351609's *"exposed a duplicate
   outer/inner holdout split"*, a real bug codex found in its own work and disclosed rather than
   quietly re-running.
4. **`planner_invocations` as a registration gate** — the ID cannot claim a solver the artifacts don't
   prove was called.

---

## 4. Recommended order before any headline

1. **F2** — add the permute-whole-episodes invariance test (cheap; closes the episode-independence gap).
2. **F3** — publish the tree-hash recipe or per-file hashes.
3. **F1** — run the 4,000-row one-cell smoke at **16 candidates (4/form)**, not 4. This is the first
   run that would actually exercise faithful PLUS, and the first that can inform runtime.
4. Then the blinded canary → CPU ceiling → full-vs-balanced-core rule (all still unset, correctly).

Unchanged from my plan audit: the ten §25 decisions remain the PI's, and this smoke does not settle
any of them.

## 5. Scope of this audit (stated explicitly)

I verified: the frozen v4 equations/imports, ordered-fit and PBVI structure, the order test, the
candidate bank composition, `sacct`, logs/exit codes/RSS, the test suite (re-run), dependency
versions, artifact presence, privacy-audit and registration/acceptance contents, superseded-run
retention, and that the 20260716 frozen run is untouched.

I did **not**: audit the numerical correctness of the PBVI backups or the discretization against an
exact reference; verify parameter recovery on real data (no run exists at usable scale); re-derive the
autodiff gradients; or review the APPL license question (no solver integrated). Those remain open and
should not be inferred from this PASS.

---

# Round 2 — fixes verified (2026-07-17)

Codex accepted all findings and applied fixes. **I re-verified each independently; all confirmed.**

| Finding | Status | Independent verification |
|---|---|---|
| **F2** permute-episode invariance test missing | **FIXED** | `tests/real/test_moor_faithful.py:29` `test_ordered_objective_is_invariant_to_whole_episode_permutation` — reverses episode blocks, remaps `episode_id`, asserts SSE invariant. Suite passes. |
| **F3** tree hash unverifiable | **FIXED** | `scripts/hash_paper_faithful_snapshot.py` reproduces the registered digest **exactly**: `included_file_count: 118`, `matched: true`, `snapshot_tree_sha256: a7836e85…`, exit 0. Recipe now published in the output ("lexically sorted recursive files; exclude `__pycache__`/`.pyc`/`.pyo`; append UTF-8 POSIX relative path then ASCII SHA-256 hex"). My Round-1 divergence was exactly the convention mismatch I hypothesised — correctly *not* reported as a defect. |
| **F5** construction unnamed | **FIXED** | `src/real_ecology_benchmark/faithful_fit.py:18` `CANDIDATE_CONSTRUCTION = "episode_bootstrap_map_v1"`, propagated to artifacts/manifests/method version. |
| **F1** within-family uncertainty unexercised | **OPEN BY DESIGN — correct** | Codex explicitly refuses to relabel the 4-candidate smoke; next run is a newly frozen 4,000-target one-cell smoke at 16 PLUS candidates. Exactly the recommended order. |
| **F4** smoke scale | unchanged | Still true; still correctly disclosed. |

**Additional checks I ran:**

- **The verifier is a usable gate:** with a deliberately wrong `--expected` it returns `matched:false`
  and **exit code 1**. A verification tool that exited 0 on mismatch would be worse than none; this
  one fails correctly.
- **Frozen v4 is provably untouched:** the code tree still hashes to the registered `a7836e85…`.
- **Live ≠ frozen, as it should be:** live `src/` digest now differs from `v4/code/src` — the fixes
  landed in live while the frozen snapshot stays as-run.
- **128 tests pass, Ruff clean** — re-ran both independently; matches codex's report.

**Defect codex self-caught that I missed.** The frozen v4 manifest records
`moor_faithful_ricker_misspec_pbvi | candidate_count: 4 | prior: uniform` — MOOR inheriting PLUS's
bank metadata. The *artifacts* are correct (`acceptance.json` shows MOOR `candidate_count: 1`), so it
was metadata-only and the run itself is sound; codex fixed the live generator to emit one fitted model
and `prior=not_applicable`. This is the **second** time codex has caught descriptive-metadata errors I
did not check (cf. the 36/28/8 population counts). My result-artifact checks were correct both times;
my metadata coverage was not. Standing correction to my own method: audit the *registration/manifest
fields*, not only the produced artifacts.

**Note for the record:** frozen v4's manifest retains that known MOOR metadata defect (correctly —
frozen stays as-run). Do not cite v4's manifest for MOOR's candidate count; cite its artifacts.

## Round-2 verdict

All F2/F3/F5 fixes verified. F1 correctly left open and explicitly not relabelled. **The
implementation is in good standing; the next legitimate run is the newly frozen 4,000-target one-cell
smoke at 16 PLUS candidates**, which is the first run capable of demonstrating faithful PLUS and the
first that can inform runtime. The ten §25 decisions remain the PI's.
