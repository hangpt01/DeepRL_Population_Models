# Independent read-only review

Date: 2026-07-28
Target: `/home/hphung/ce25_scratch2/DeepRL_Population_Models/` (real path
`/fs04/scratch2/ce25/DeepRL_Population_Models`), branch `main`, commit `d02ee27`.

Reviewer had no prior involvement in building this repository. Every number below
was recomputed in this session; no claim from `VERIFY_REPORT.md` was taken on
trust. All commands, logs, and scratch scripts are under `.review/logs/`.

**Read-only compliance.** No tracked file was modified. `git status --porcelain`
is empty before and after the review, including after `pip install -e .`, both
track test suites, and the accepted-cell replay. The frozen tracks, the original
scratch project, and every accepted artifact were read only. The one exception to
"no writes" is this file plus `.review/` (which `.gitignore` already swallows via
its generic `logs/` and `venv/` patterns). Replay output was directed to
`.review/logs/diagnostic_replay/` with `--output-root` so the pre-existing
`.verification/` tree from the earlier build was left untouched.

---

## 1. Verdict

The scientific core is sound and the integrity story checks out exactly. Every
hash matches, the two-track claim is genuine and verifiable, and I was able to
reconstruct an accepted cell's seven fields from first principles using only this
repository's tables and frozen code. **The accepted-cell reproduction passes
bit-exactly** — `max_abs_diff = 0.0` on all seven fields, from a fresh venv built
only from `requirements.txt`, with the scratch project supplying nothing but
provenance-pinned data.

What does not hold up is the *standalone* framing around that core. Two defects
are serious enough to block: the documented install command fails outright, and
the parity gate — the repository's headline guarantee — is computed and printed
but never enforced, so `make verify` exits 0 whether or not the accepted cell
reproduces. It happened to reproduce here; nothing in the harness would have told
us if it had not.

| Class | Count | Items |
|---|---:|---|
| BLOCKER | 2 | B1, B2 |
| SHOULD-FIX | 7 | S1–S7 |
| NICE-TO-HAVE | 9 | N1–N9 |
| FUTURE-REFACTOR | 1 | F1 |

Per the review brief, every finding is a **note for a future reviewed change**.
Nothing was edited.

---

## 2. Task 1 — Standalone reproducibility

### 2.1 Environment

Built a fresh venv from a stock Python 3.10.14
(`/apps/miniforge3/24.3.0-0/miniforge3/bin/python`, the same base interpreter the
accepted `.venv-paper-faithful` uses). Result: `numpy 2.2.6`, `torch 2.13.0+cpu`
— the exact pins.

`requirements.txt` is genuinely what it claims to be. I ran `pip freeze --all`
inside `.venv-paper-faithful` and diffed:

```
requirements.txt == pip freeze --all of .venv-paper-faithful (EXACT)
```

### 2.2 The documented install command fails → **B1**

README step 3 is `python -m pip install -r requirements.txt`. Run verbatim:

```
ERROR: Could not find a version that satisfies the requirement torch==2.13.0+cpu
       (from versions: 1.11.0, ..., 2.12.1, 2.13.0)
ERROR: No matching distribution found for torch==2.13.0+cpu
exit 1
```

`torch==2.13.0` exists on PyPI; the `+cpu` local-version build does not — it lives
only on `download.pytorch.org/whl/cpu`, and `requirements.txt` carries no
`--index-url` / `--extra-index-url` directive. Adding
`--extra-index-url https://download.pytorch.org/whl/cpu` installs all 22 pins
cleanly, `torch 2.13.0+cpu` included. Logs:
`.review/logs/pip_install_verbatim.log`,
`.review/logs/pip_install_with_torch_index.log`.

### 2.3 `make verify` results

Run with `PYTHON=$PWD/.review/venv/bin/python` — i.e. entirely from this
repository's code, with the scratch project supplying only provenance-pinned data
(fit cache, public surrogate, fit receipt, dataset).

**1. Self-tests — PASS.** 36/36 assertions, 0 failures, exit 0. The README's "36
current assertions" is accurate: `test_metrics.py` contains exactly 36 top-level
`check(...)` calls and all 36 printed PASS.
(`.review/logs/verify_self_tests.log`)

**2. Constants — PASS.** exit 0. (`.review/logs/verify_constants.log`)

```
PASS vulture_bound=42.937296032209
PASS dominated_actions[egyptian_vulture]={5: 0, 6: 0, 7: 3, 8: 3, 9: 4}
PASS dominated_actions[amur_tiger]={5: 0, 6: 0}
PASS dominated_actions[crab_eating_fox]={}
```

**3. Accepted-cell reproduction (A6, Egyptian vulture / ricker / σ=0.1, MOOR) — PASS.**

```
{"cell": "A6", "side": "moor", "parity": "PASS", "max_abs_diff": 0.0,
 "elapsed_s": 1595.4, "recomputed_fits": 0, "reuse_ok": true}
```

**Seven-field `max_abs_diff` = `0.0`** — exact, not merely within 1e-9, on every
field:

| Field | Replayed | Accepted | abs_diff |
|---|---:|---:|---:|
| return_mean | −187.26465138858708 | −187.26465138858708 | 0.0 |
| return_sd | 0.0 | 0.0 | 0.0 |
| unsafe_fraction_mean | 1.0 | 1.0 | 0.0 |
| persistence_mean | 0.0 | 0.0 | 0.0 |
| collapse_entry_mean | 0.0 | 0.0 | 0.0 |
| min_population_mean | 0.4289444166804965 | 0.4289444166804965 | 0.0 |
| economic_cost_mean | 9.375 | 9.375 | 0.0 |

`fit_cache_reuse: true`, `recomputed_fits: 0` (but see B2 on how that field is
produced), elapsed 1595.4 s. Run on `m3-login2`, Intel Xeon Gold 6548Y+ — i.e.
**not** under the `accepted-8452Y` Slurm profile, and parity was still bit-exact.
That is consistent with REPRODUCE.md's note that A6-MOOR has passed on the login
6548Y+ host; it does not generalise to other cells, and the accepted profile
should still be used where strict parity is required.

This is the strongest single result in the review: with the scientific code taken
entirely from this repository and only provenance-pinned data read from scratch, a
fresh venv built from `requirements.txt` reproduces the accepted cell bit-for-bit.
(`.review/logs/verify_cell_A6_moor.log`, `.review/logs/diagnostic_replay/parity/`)

### 2.4 Hidden dependencies on the original tree → **S1**

The brief asked me to flag any dependency on the original messy tree that is *not*
a provenance-pinned data reference. There are many, and they are dependencies on
**code**, not data.

Only `run_diagnostic_replay.py` was actually parameterized. **1 of 15**
diagnostics entry points is runnable from this repository alone. The other 14
either hard-code a `/fs04/...` absolute path, or assume
`Path(__file__).resolve().parents[3]` is the *scratch project* root containing
`real_ecology_runs/`, or assume the run-output tree is a sibling of `scripts/` —
none of which holds in this layout:

```
script                            declared anchor                        runnable here?
replay/run_diagnostic_replay.py   env DEEPRL_* (default parents[3])      YES
replay/a0_baseline.py             abs /fs04/.../Claude_DeepRL_...        no  (imports code from scratch)
replay/constant_action_sweep.py   abs /fs04/.../Claude_DeepRL_...        no  (imports code from scratch)
replay/run_tier_b_replay.py       abs /fs04/.../general_rl_phase2_iso    no  (+ needs REPO/real_ecology_runs/)
followups/run_s2.py               parents[3] == scratch project root     no
followups/run_s2_convergence.py   imports run_s2                         no  (inherits the above)
followups/run_reward_screen.py    parents[3] == scratch project root     no
followups/run_s6_fit_probe.py     parents[3] == scratch project root     no
followups/resume_s6_moor.py       parents[3] == scratch project root     no
followups/run_h12_arm.py          parents[3] == scratch project root     no
followups/run_h12_m1_anchor.py    parents[3] == scratch project root     no
followups/run_h14.py              parents[3] == scratch project root     no
followups/aggregate_h12.py        parents[1] == followups RUN dir        no  (-> scripts/diagnostics/H12_out)
followups/aggregate_h14.py        parents[1] == followups RUN dir        no  (-> scripts/diagnostics/H14_out)
followups/finalize_followups.py   parents[1].parents[1] + real_ecology_runs  no  (accepted CSV is at results/accepted/)
```

(`.review/logs/diagnostics_path_audit.log`)

The three `parents[1]` aggregators are a slightly different failure: they expect
the H12/H14 output directories to sit beside `scripts/`, as they did in the
original run tree, and so resolve to `scripts/diagnostics/H12_out` here. The
`finalize_followups.py` gate re-reads the accepted CSV from
`REPO/real_ecology_runs/...` rather than from `results/accepted/`, where this
repository actually keeps it.

The important part is *what* they import. `a0_baseline.py`,
`constant_action_sweep.py`, `run_reward_screen.py`, `run_s2.py`,
`run_s6_fit_probe.py` and friends all do:

```python
FROZEN = ROOT / "ricker_only_plus_72_20260720" / "runtime_snapshot_routing_fix"
sys.path.insert(0, str(FROZEN / "scripts")); sys.path.insert(0, str(FROZEN / "src"))
import run_real_manifest_row as rrmr
from real_ecology_benchmark.config import load_config
```

They import `real_ecology_benchmark` from the **scratch snapshot**, not from
`src/tracks/ecological/`. The bytes happen to be identical today (I verified all
107), but the repository copy is not the one being executed, so the "all CODE
comes from this repo" property does not hold for these entry points. The
Phase-1 audit lists S1 constant actions, S2 oracle solver, reward-screen scorer,
and H12/H14/S6 followups as CORE elements; they are present as files but not
runnable from this repository alone.

`CLEANUP_MANIFEST.md` states the copied harness was changed to "parameterize
external scratch/data and output roots with environment variables". That is true
of `run_diagnostic_replay.py` and of nothing else.

Cluster specifics (`--account=ce25 --partition=m3h --qos=m3h
--constraint=xenon-8452Y`) are correctly retained per the accepted-8452Y profile
and are **not** counted as a finding. The `/fs04/...` `--output=` paths, `ROOT=`
assignments, and hard-wired `srun .../.venv-paper-faithful/bin/python`
interpreters in the same `.sbatch` files are a different matter and are covered
by S1.

---

## 3. Task 2 — Integrity re-verification

Recomputed independently with `sha256sum -c`. All logs in `.review/logs/`.

| Check | Expected | Result |
|---|---:|---|
| `provenance/frozen_tracks.sha256` → repo destinations | 107 | **107/107 OK**, 0 mismatch |
| `provenance/frozen_tracks.sha256` → canonical scratch sources | 107 | **107/107 OK**, 0 mismatch |
| `provenance/copied_artifacts.sha256` → repo destinations | 18 | **18/18 OK**, 0 mismatch |
| `provenance/copied_artifacts.sha256` → canonical scratch sources | 18 | **18/18 OK**, 0 mismatch |
| `MATCHED_P10_144_METHOD_CELLS.csv` SHA-256 | `7431318803e468c1…cadc1c` | **matches** |
| Receipt's own `matched_csv_sha256` | same | **matches** |

**No mismatches anywhere — 250 hash verifications, all pass.** Additional coverage
checks I ran that the ledger does not itself assert:

- The 107 manifest entries are a *bijection* with the `.py` files on disk under
  `src/tracks/` — 52 ecological + 55 general, no unhashed extras, no missing
  entries.
- `configs/ecology/*.csv` (the authoritative action/species tables) are
  byte-identical to the corresponding `real_ecology_data/` in **both** frozen
  originals.
- All 20 distinct `dataset_sha256` values in the accepted table are present in
  `provenance/dataset_hashes.csv` (144 rows).

**The two-track claim is real**, and I verified it rather than assuming it. The
tracks differ in exactly 14 paths: 3 general-only files (`behavior_model.py`,
`general_canary_acceptance.py`, `methods/ensemble_value_disagreement.py`) and 11
files with differing content (`config.py`, `faithful_fit.py`, `manifest.py`,
`public_models.py`, `training_monitor.py`, `methods/{__init__,bamcts,delphic,
ogsrl,plus_faithful,refplan}.py`). This matches the Phase-1 audit's "14
differing/one-sided paths" exactly. The remaining 41 shared files — including
`envs.py`, `evaluator.py`, `reward.py`, `actions.py` — are byte-identical, which
is what makes "a common evaluator" true rather than merely asserted.

---

## 4. Task 3 — Standalone / usability audit

### 4.1 Does REPRODUCE.md work step by step?

Not quite. Three gaps, in the order a newcomer hits them:

1. **Python 3.10.14 is not obtainable from the instructions** (→ S7). There is no
   `python3.10` on this host's PATH; system Python is 3.9.25 and `pyproject.toml`
   requires `>=3.10,<3.11`. The interpreter actually used by the accepted venv is
   `/apps/miniforge3/24.3.0-0/miniforge3/bin/python`, discoverable only by
   inspecting the scratch venv's symlinks.
2. **The install fails** (→ B1, above).
3. **`make verify` dies at step 1 on a missing `pandas`** (→ S4). REPRODUCE.md's
   Requirements section lists only Python 3.10.14, `requirements.txt`, CPU torch,
   and the external data. But `run_analysis.py` imports `pandas` and `figures`
   (which imports `matplotlib`) at module scope, *before* the `--self-test`
   branch, and neither is in `requirements.txt`. Proved with a venv holding only
   the `pyproject` core deps:

   ```
   File ".../run_analysis.py", line 14, in <module>
       import pandas as pd
   ModuleNotFoundError: No module named 'pandas'
   make: *** [Makefile:11: verify-self-tests] Error 1
   ```

   README's install block *does* cover this (`pip install -e '.[analysis]'`), so
   the fix is to bring REPRODUCE.md's Requirements section in line with README.
   (`.review/logs/selftest_without_analysis_extra.log`)

After working around all three, the documented flow runs to completion.

### 4.2 Are both tracks selectable and documented?

Track selection by `PYTHONPATH` is documented and works; the "never both at once"
warning is correct and worth keeping. But the two tracks are not equally served:

- **Ecological**: `experiments/accepted_p10/` ships both configs and both 24-row
  manifests; `make verify-cell` replays an accepted cell end to end. Complete.
- **General**: no accepted-experiment package at all (→ S2). The 576-row general
  manifest is not in the repository, and `run_tier_b_replay.py` hard-codes both it
  and the general code root to `/fs04/.../general_rl_phase2_iso/...`. There is no
  `make` target for the general side. The 96 general cells (RefPlan, OGSRL,
  BA-MCTS, EVD × 24) cannot be replayed from this repository.

`make test-ecological` passes (123 tests, OK). **`make test-general` fails**
(→ S3): 150 tests ran, 1 collection error, exit 1.

```
File ".../tests/general/real/test_general_privacy.py", line 38, in <module>
    from tests.real.test_general_paper_mechanisms import _hidden_fixture
ModuleNotFoundError: No module named 'tests.real.test_general_paper_mechanisms'
```

Root cause is the re-layout, not an edit: the general original had a flat
`tests/real/` containing `test_general_paper_mechanisms.py`, and the repo file is
byte-identical to it. In this repository `tests/real/` is instead the stale
ecological copy, which lacks that module. Note this is test-harness code and is
**not** in the frozen 107-file manifest, so a future fix here would not touch
frozen scientific source.

### 4.3 Hard-coded absolute paths that escaped parameterization

Covered by S1 above. Within the `make verify` path itself, parameterization is
clean: `Makefile` and `run_diagnostic_replay.py` expose scratch/data/output roots
as overridable variables and the only absolute path is a *default*.

`configs/paths.example.yaml` advertises seven `DEEPRL_*` overrides but omits
`DEEPRL_P10_PACKAGE`, and most scripts it implicitly speaks for honour none of
them (→ N2).

### 4.4 CORE element coverage

Every CORE element is **present**. Two are present-but-not-runnable-standalone.

| CORE element | Where | Status |
|---|---|---|
| Environment + 4 family maps | `tracks/*/real_ecology_benchmark/envs.py` (`ricker`/`allee`/`theta`/`regime` branches) | present, verified §5 |
| 11-action table | `configs/ecology/{actions,action_effects_long}.csv` + `actions.py`/`realdata.py` | present, verified (a0–a10) |
| Species / `K_ref` / `s_safe` | `configs/ecology/species.csv` | present, verified §5 |
| True-next-state reward | `reward.py::state_reward`, byte-identical across tracks | present, verified §5 |
| `ContinuousEvaluator` | `evaluator.py`, byte-identical across tracks | present |
| PLUS faithful | `tracks/ecological/.../methods/plus_faithful.py` | present |
| MOOR faithful | `tracks/ecological/.../methods/moor_faithful.py` | present |
| RefPlan / OGSRL / BA-MCTS | `tracks/general/.../methods/{refplan,ogsrl,bamcts}.py` | present |
| EVD | `tracks/general/.../methods/ensemble_value_disagreement.py` | present |
| Public surrogate | `public_surrogate.py` | present |
| Diagnostic replay + parity gate (eco) | `scripts/diagnostics/replay/run_diagnostic_replay.py` | present; **gate not enforced** (B2) |
| Diagnostic replay (general / Tier-B) | `scripts/diagnostics/replay/run_tier_b_replay.py` | present; **not runnable standalone** (S1/S2) |
| S1 constant-action reference | `a0_baseline.py`, `constant_action_sweep.py` | present; **not runnable standalone** (S1) |
| M1–M15 analysis | `src/diagnostics/replay_analysis/metrics.py` | present; `m1`–`m15` all defined |

The accepted table is well formed: 144 rows = 6 methods × 3 populations × 4
families × 2 σ, and in all 24 cells the six methods share one `dataset_sha256` —
so "matched" is literally true, not just claimed.

---

## 5. Task 4 — Science spot-check (read-only)

Everything asked for **checks out**. Full transcript:
`.review/logs/science_spotcheck.log`, driver `.review/logs/science_spotcheck.py`.

**Vulture bound 42.937.** Recomputed from the closed form independently of the
repo's function: with `dN(a10)=4.1` and `r=−0.0912`,
`x* = dN·e^r/(1−e^r) = 42.937296032209`, agreeing with
`max_reachable_abundance("egyptian_vulture")` to `<1e-12` and satisfying
`(x*+dN)e^r = x*` exactly. It is below `s_safe = 81.25` and above `N0 = 41`, so
the vulture is a permanent-penalty sink under every policy — as the repo says.
`max_reachable_abundance` correctly returns `nan` for tiger and fox, whose
premise (`r ≤ 0` for all actions) fails.

**Dominated-action sets.** Re-derived by brute force from `configs/ecology`
without using the repo's function: vulture `{5:0, 6:0, 7:3, 8:3, 9:4}`, tiger
`{5:0, 6:0}`, fox `{}`. Identical to both the documented sets and
`constants.dominated_actions`.

**Family fixed points.** Exercised the frozen `transition_value` directly:
ricker `x*=K`; allee `x*=K` and `x*=C`; theta-logistic `x*=K` for θ∈{3,6};
regime `x*=K` and `x*=threshold` in both regimes. All residuals `<1e-9`.

**`r_pos = max(r_eff, 0)`.** Confirmed at `envs.py:240` with
`r_mort = min(r_eff, 0)` applied as an unconditional `value *= exp(r_mort)`. I
verified the consequence that matters: when `r_pos ≡ 0`, all four families
collapse to the *same* map `x' = (x + dN)·exp(r_mort)` — checked for all 11
actions × 4 probe states in each family, agreement `<1e-12`. This is what makes
the vulture bound family-independent, and it is also why the theta variant of the
bound (44.998) is still far below `s_safe`.

**True-state reward.** `R = x'/(x'+K_ref) − cost(a) − 10·1[x' ≤ s_safe]`,
evaluated on the true next state, `occupancy` indicator, `collapse_penalty = 10`.
Matches a hand computation to `<1e-12`, including the boundary (`x' = s_safe` is
penalised; `x' = s_safe + 1e-7` is not). `reward.py` is byte-identical across the
two tracks.

### 5.1 Two cross-checks worth recording

These were not asked for; they fell out of the above and they materially raise
confidence in the accepted numbers.

**(a) The accepted table's dataset count is explained by the degeneracy.** The
144-cell table has 24 cells but only 20 distinct datasets. The four "extra"
collapses are exactly the vulture's ricker/allee/regime cells, which share one
dataset per σ. I confirmed why: for the vulture every set-point is ≤ 0, so
`r_pos ≡ 0` and those three families are *the same dynamical system* — 50-step
trajectories agree to `<1e-12`. Theta differs only because it reads the LGM
set-point column (−0.0872 vs −0.0912), not because its functional form survives.
So `24 − 4 = 20`, exactly. (`.review/logs/vulture_family_degeneracy.log`.) This
is a genuine scientific property of the benchmark, and it is precisely what the
repo's own M7 "family degeneracy" metric exists to measure — not a defect.

**(b) I reconstructed the accepted A6 cell analytically.** Using only
`configs/ecology` and the frozen `transition_value`, with no replay harness:
`economic_cost_mean/50 = 0.1875` identifies the deployed policy as constant `a5`;
rolling that policy for 50 steps from `N0` reproduces all seven accepted fields.

| Field | Analytic | Accepted | abs diff |
|---|---:|---:|---:|
| return_mean | −187.26465138858714 | −187.26465138858708 | 5.7e−14 |
| return_sd | 0.0 | 0.0 | 0 |
| unsafe_fraction | 1.0 | 1.0 | 0 |
| persistence_mean | 0.0 | 0.0 | 0 |
| collapse_rate | 0.0 | 0.0 | 0 |
| min_population_mean | 0.42894441668049654 | 0.4289444166804965 | same double |
| economic_cost_mean | 9.375 | 9.375 | 0 |

Consistent with the repo's own diagnostic framing: `a5` is in the dominated set
`{5,6,7,8,9}` — dynamically identical to `a0` (same set-point, same `dN`) but
costing 0.1875/step — so the accepted MOOR policy pays `0.1875 × 18.4611 =
3.4615` in return for nothing. (`.review/logs/a6_analytic_crosscheck.log`.) This
independently corroborates the accepted number without relying on the replay
harness at all.

---

## 6. Findings

### BLOCKER

**B1 — `pip install -r requirements.txt` fails; the documented install is broken.**
`torch==2.13.0+cpu` is a PyTorch-index local build, absent from PyPI, and
`requirements.txt` has no index directive. README's step 3 exits 1 on a clean
machine. *Note for a future change:* add
`--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of
`requirements.txt` (which keeps `pip freeze` fidelity for the pins themselves), or
document the flag in both README and REPRODUCE.md. Verified: with that flag all
22 pins install and `torch.__version__ == '2.13.0+cpu'`.

**B2 — The 1e-9 parity gate and the fit-cache reuse gate are computed but never
enforced.** `run_diagnostic_replay.py::main` contains exactly one `assert` (line
546, the accepted-CSV hash), no `sys.exit`, no `raise`, and no `return` of a
status code. `parity()` (line 342) computes `PASS` / `INVESTIGATE` / `FAIL` and
`run_cell` (line 399) computes `reuse_ok`; both are printed and written into
`PARITY_*.json`, and neither is ever tested. `verify_accepted_cell.py` faithfully
propagates the child's exit code — which is 0 regardless of verdict. Consequence:
`make verify` reports success even when the accepted cell fails to reproduce,
which inverts the guarantee README and REPRODUCE.md advertise. Separately,
`recomputed_fits` is the hard-coded literal `0` at line 402 — it is written into
the receipt and described as a gate, but nothing measures it.
*Mitigating:* several strong invariants **do** raise inside `run_cell` — dataset
SHA, `s_safe`/`K_ref` agreement, `n_steps == 50`, the per-step reward identity to
1e-9, shadow-rollout non-mutation, and RNG-stream invariance. So a silent pass
requires the failure to be confined to the seven aggregate fields or the cache
gate. *Note for a future change:* exit non-zero unless
`verdict == "PASS" and reuse_ok`, and derive `recomputed_fits` from
`policy.fit_diagnostics` instead of hard-coding it.

### SHOULD-FIX

**S1 — 14 of 15 diagnostics entry points cannot run from this repository, and
several import scientific code from the original scratch tree.** Detailed in
§2.4. Only `run_diagnostic_replay.py` is parameterized; the rest hard-code
`/fs04/...`, assume `parents[3]` is the scratch project root, or expect run
outputs beside `scripts/`. This covers CORE elements S1, S2, reward-screen, H12,
H14, S6 and the general Tier-B replay. *Note:* apply the same `DEEPRL_*`
environment pattern and point `sys.path` at `src/tracks/ecological` (or
`general`);
`CLEANUP_MANIFEST.md`'s parameterization claim should be narrowed to the one file
it is true of.

**S2 — The general track has no accepted-experiment package.**
`experiments/accepted_p10/` holds only the ecological MOOR/PLUS configs and 24-row
manifests. The general 576-row manifest is absent, `run_tier_b_replay.py`
hard-codes the general code root and manifest to scratch, and there is no `make`
target for the general side. 96 of the 144 accepted cells are therefore not
reproducible from this repository. *Note:* add
`experiments/accepted_general/` and a `verify-cell-general` target mirroring the
ecological one.

**S3 — `make test-general` fails.** `tests/general/real/test_general_privacy.py:38`
imports `tests.real.test_general_paper_mechanisms`, which does not exist in this
layout (§4.2). 1 collection error, exit 1. *Note:* the cleanest fix is a
`tests/general/__init__.py`-rooted relative import or a `conftest.py` shim; the
file is byte-identical to the general frozen original but is not covered by the
frozen 107-file manifest, so this is safe to change.

**S4 — REPRODUCE.md omits the `[analysis]` extra**, so following it alone makes
`make verify` fail at step 1 on `import pandas` (§4.1). README is correct;
REPRODUCE.md is the designated reproduction document and should match it.

**S5 — Stale third copies of `scripts/` and `tests/` that differ from both
tracks.** Beyond the intentional two-track layout there is an *undifferentiated
flat copy* at the top level, and it is stale. Most damagingly,
`scripts/run_real_manifest_row.py` (387 lines) lacks the
`plus_adapted_ricker_only_pbvi` branch present in
`scripts/ecological/run_real_manifest_row.py` (414 lines) — the registration path
the accepted PLUS cells require. Anyone who puts `scripts/` on `PYTHONPATH`
instead of `scripts/ecological/` gets silently different scientific behaviour.
`scripts/run_adapted_fit_row.py` likewise differs from the ecological copy, and
`tests/real/` is the ecological set minus `test_ricker_only_plus.py`. *Note:*
delete the flat copies or make them explicit forwarding shims. (The
`configs/*.yaml` flat copies are byte-identical to both tracks and are harmless —
see N5.)

**S6 — Integrity manifest does not cover all load-bearing scientific inputs.**
`frozen_tracks.sha256` covers exactly the 107 `src/tracks/**/*.py`. It does not
cover `scripts/ecological/` and `scripts/general/` — which the verify path
imports for `apply_row_config`, `validate_adapted_fit_receipt_gate` and
`validate_adapted_plan_cache_hit` — nor `configs/ecology/*.csv` (the authoritative
11-action and species tables), nor `experiments/accepted_p10/`. These are
scientific inputs with no hash ledger. I verified them by hand against the frozen
originals and they are correct today; the point is that nothing would catch drift.

**S7 — REPRODUCE.md does not say how to obtain Python 3.10.14.** No `python3.10`
on PATH; system Python is 3.9.25. The accepted venv's base interpreter is
`/apps/miniforge3/24.3.0-0/miniforge3/bin/python` (module `miniforge3/24.3.0-0`).
One line in REPRODUCE.md would remove a real dead end.

### NICE-TO-HAVE

**N1 — The `src/tracks/real_ecology_data -> ../../configs/ecology` symlink is
load-bearing and effectively undocumented.** This is a genuinely elegant solution:
the frozen `realdata.py` computes `DATA_DIR = Path(__file__).parents[3] /
"real_ecology_data"`, and re-nesting the package one level deeper would have
broken it — the symlink restores the anchor without editing frozen code. But it is
mentioned only in `VERIFY_REPORT.md`; README and REPRODUCE.md are silent. It will
not survive a `git archive`/zip export or a checkout without symlink support, and
the failure mode is an opaque error deep inside `realdata`. *Note:* document it,
and consider a startup existence check with a clear message.

**N2 — `configs/paths.example.yaml` is incomplete and over-promises.**
`DEEPRL_P10_PACKAGE` is missing from the list, and the overrides it advertises are
honoured by only one of the scripts it implicitly describes (see S1).

**N3 — Inconsistent parity constraint across sbatch scripts.** 8 of 12 carry
`--constraint=xenon-8452Y`; `postprocess_followups`, `reward_screen`,
`s6_fit_probe` and `s6_moor_resume` do not. Two of those produce results committed
under `results/followups/{reward_screen,S6}/`. Worth stating explicitly whether
those artifacts are inside or outside the accepted-profile determinism claim.

**N4 — `.review/` is not in `.gitignore`.** It stays invisible only because
`.review/logs/` and `.review/venv/` happen to match the generic `logs/` and
`venv/` patterns. A future reviewer writing elsewhere under `.review/` would dirty
the tree.

**N5 — `configs/*.yaml` duplicate `configs/tracks/*/` byte-for-byte.** All 20
shared configs are identical in both tracks, so unlike S5 there is no divergence
risk — just clutter.

**N6 — Dead and silently-degrading analysis code.** `metrics.m10_noise_sensitivity`
has no caller anywhere and is absent from the `ALL_METRICS` registry (it needs two
logs, so this is understandable, but it means M10 never runs). More notably,
`compute_all` wraps every metric in `except Exception` and converts failures to
`{"available": False, "reason": ...}` — deliberate ("never lose the report") but it
means a broken metric cannot fail the analysis, which compounds B2.

**N7 — ruff is pinned and configured but cannot act as a gate.**
`ruff check src scripts` reports 322 findings (28 in the frozen tracks, which are
byte-preserved and should stay that way; 294 in repo-layer/copied scripts, mostly
`E402`/`E702` style). No `make` target runs it.

**N8 — `archive/` is tracked, not gitignored.** Only `archive/artifacts/` is
excluded. 3,060 of 3,769 tracked paths (81%) live under `archive/`. README states
this accurately; flagging it because the review brief described `archive/` as
gitignored, and anyone inheriting that phrasing will be surprised by the repo
size.

**N9 — `scripts/top_up_followups.py` copies without updating the ledger, and
re-copies an already-committed artifact.** The script itself is one of the
well-built repo-layer pieces (required `--scratch-project`, repo-relative
destinations, prints hashes, refuses to run until all four sources exist). Two
small gaps: it prints the new hashes but does not append them to
`provenance/copied_artifacts.sha256`, so the ledger must be updated by hand; and
its `FILES` map includes `S6_out/S6_RECEIPT.json`, which is already committed and
already hash-recorded. That is a no-op today — I verified the scratch source still
hashes to the ledgered `c33b6052…` — but if S6 were ever re-run, running the H12
top-up would silently overwrite a ledgered artifact and leave the ledger stale.

### FUTURE-REFACTOR

**F1 — `constants.SPECIES["egyptian_vulture"]["r_lgm"]` has 5 of 11 entries copied
from the Ricker column.** Declared vs. actual theta/LGM set-points from
`configs/ecology`:

| action | declared `r_lgm` | true LGM | note |
|---|---:|---:|---|
| a1 | −0.2179 | −0.1958 | Ricker value |
| a2 | −0.2558 | −0.2257 | Ricker value (`r_min_ricker`, not `r_min_lgm`) |
| a3, a7, a8 | −0.0198 | −0.0196 | Ricker value |

Tiger and fox `r_lgm` columns are correct. **Currently inert**: nothing in the
repository calls `dominated_actions` or `max_reachable_abundance` with
`r_column="r_lgm"` — every caller uses the `r_ricker` default — and both functions
return identical answers under the declared and true values anyway
(`{5:0,6:0,7:3,8:3,9:4}` and `44.998138182140984` either way), because the errors
preserve the equality pattern the derivations depend on. So no published number
changes. Flagged for a future reviewed change only.
(`.review/logs/rlgm_discrepancy.log`)

---

## 7. Verified correct — please do not re-litigate

Recorded so a later reviewer does not redo this work:

- **All provenance hashes match — 250 verifications** (107 frozen + 18 copied,
  each checked on both the repo side *and* the canonical scratch source), plus
  the accepted CSV and its receipt's self-recorded hash.
- **The two-track layout is justified and was verified, not assumed** — 14
  genuinely divergent paths, 41 byte-identical shared files including the
  evaluator, environment, reward and action modules. Merging would change
  scientific behaviour. This is correct as designed.
- **`requirements.txt` is exactly `pip freeze --all` of `.venv-paper-faithful`.**
- **`configs/ecology/` matches both frozen originals byte-for-byte.**
- **The accepted table is internally consistent**: 144 = 6×3×4×2, one shared
  dataset per cell across all six methods, all dataset hashes in provenance.
- **The A6-MOOR accepted cell reproduces bit-exactly** (`max_abs_diff = 0.0`,
  `reuse_ok: true`) from a repo-only venv, and independently again by closed-form
  reconstruction to 5.7e−14. Two mutually independent routes to the same seven
  numbers.
- **The science spot-checks all pass** (§5): vulture bound, dominated-action
  sets, four family fixed points, the `r_pos`/`r_mort` split, and the
  true-next-state reward.
- **Cluster specifics behind `accepted-8452Y` are intentional** and were not
  treated as defects.
- **The instrumentation design is sound**: `__class__`-swap logging, shadow
  rollouts asserted to consume no RNG and mutate no state, planner
  `action_values` captured rather than recomputed. These are careful choices and
  the asserts backing them do fire.

---

## 8. Logs

All under `.review/logs/` (gitignored):

| File | Contents |
|---|---|
| `pip_install_verbatim.log` | B1: documented install failing |
| `pip_install_with_torch_index.log` | B1: install succeeding with the PyTorch index |
| `pip_install_analysis.log` | `pip install -e '.[analysis]'` |
| `frozen_check.txt` / `frozen_source_check.txt` | 107 hashes, repo + scratch sides |
| `copied_check.txt` / `copied_source_check.txt` | 18 copied-artifact hashes, repo + scratch sides |
| `verify_self_tests.log` | 36/36 self-test assertions |
| `verify_constants.log` | constants check |
| `verify_cell_A6_moor.log` | accepted-cell replay (7-field parity) |
| `diagnostic_replay/` | replay outputs, redirected away from `.verification/` |
| `selftest_without_analysis_extra.log` | S4: `ModuleNotFoundError: pandas` |
| `test_ecological.log` / `test_general.log` | 123 OK / 150 with 1 error |
| `diagnostics_path_audit.log` | S1: entry-point path resolution |
| `science_spotcheck.py` / `.log` | Task 4 transcript |
| `rlgm_discrepancy.log` | F1 |
| `vulture_family_degeneracy.log` | §5.1(a) |
| `a6_analytic_crosscheck.log` | §5.1(b) |
| `venv_paper_faithful_freeze.txt` | `pip freeze --all` of the accepted venv |
