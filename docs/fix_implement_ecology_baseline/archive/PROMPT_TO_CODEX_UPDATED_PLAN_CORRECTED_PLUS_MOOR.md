# Prompt to Codex — updated plan for the corrected PLUS/MOOR (decisions are settled)

**Read first, in order:**

1. `docs/fix_implement_ecology_baseline/REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR_IMPLEMENTATION.md`
   — the external paper reviewer's findings (F1–F4).
2. `docs/fix_implement_ecology_baseline/DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`
   — **the controlling decisions. All five are settled. Do not reopen them.**
3. `docs/fix_implement_ecology_baseline/RESPONSE_TO_REVIEW_PROVISIONAL_PAPER_ALIGNED_PLUS_MOOR.md`
   — what we conceded and what we asked.
4. This file — server-side implementability findings you must build on.

---

## 0. What is authorized right now

**Produce the updated implementation plan — the 13 items in `DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`
§"Next required deliverable". Nothing else.**

**Do not:** modify source, install dependencies, freeze a snapshot, or submit jobs. The decisions
document states this explicitly, and the last time this gate was skipped it produced a void run, two
blocking defects, and ~4 wasted CPU-hours. Implementation follows immediately after the plan is
approved — you already have every scientific decision, so the plan should be quick.

The scientifically void run at `real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/` stays
preserved and excluded from all claims.

## 1. Server-side implementability findings — use these, do not re-derive

I verified the decisions against this server. All five are implementable. Three carry traps that would
silently defeat them.

### 1.1 Decision 3's caching claim is TRUE — but the obvious cache key defeats it

I compared every cell in the completed hidden run:

- **144/144 cells: `observations`, `actions`, `next_observations`, `episode_id`, `timestep`,
  `terminated`, `truncated`, `costs`, `action_costs`, `pop_ids`, `dones` are byte-identical across
  `reward_safe` and `reward_yield`.**
- **Only `rewards` differs.**

So the reviewer is right that model fitting is reward-mode independent, and the 2× reuse is real.

**The trap:** the public `dataset_sha256` **includes `rewards`**, so it differs across reward modes for
all 144 cells (verified: 0/144 match). **Keying the fit cache on `dataset_sha256` yields a 0 % hit rate
and silently destroys the entire saving.** The cache key must be a hash of the **fit-relevant transition
fields only** (`observations, actions, next_observations, episode_id, timestep, terminated, truncated`),
plus fit config, form, candidate seed, and observation protocol — never the full dataset hash.

**Second nuance the decisions do not state:** the saving applies to **fitting only**. PBVI planning
depends on the reward, so it still runs per reward mode. Your plan must therefore split the cost model
into *fit* vs *plan*, and the canary must **measure that split** — the ≥1,000 CPU-h projection assumes
no reuse, and the true figure depends on what fraction of PLUS's ~3 h is fitting.

### 1.2 Decision 4's public category exists — it is the `channel` column, and ONLY that

`real_ecology_data/actions.csv` already carries a population-independent mechanism category:

```
a0  none            a1–a4  rate            a5,a6  capacity
a7–a9  rate+capacity                       a10    state
```

This is exactly the reviewer's "qualitative intervention mechanism category" and maps directly onto the
required structural zeros: `u_a ≠ 0` only for `state` (a10); `d_a ≠ 0` only where the channel includes
`capacity` (a5–a9); rate-only actions get neither; `none`/`capacity`-only actions share the baseline
rate.

**The trap — the same file leaks the hidden magnitudes.** `actions.csv` also contains
`K_multiplier` (1.1, 1.3), `dN_fraction` (0.1), `lambda_source`, and `name_mechanistic`, whose strings
are literally `"K x1.3"`, `"0.5x adult mortality"`, `"+10% N0 to N"`. **Those are the effect magnitudes
the hidden regime exists to hide.** Requirements:

- expose **only** the `channel` value per action, as a categorical token in `MethodContext`
  (e.g. `action_channels: tuple[str, ...]`);
- never place `K_multiplier`, `dN_fraction`, `lambda_source`, `name_mechanistic`, `name_original`,
  `interpretation`, or any other column into `MethodContext`, artifacts, or the public dataset;
- add a privacy test asserting the above names/values are unreachable from any method-facing object,
  and extend the forbidden-name guard accordingly;
- the channel map is population-independent, so it introduces no per-population leak — state this.

### 1.3 Decision 5's signed `r_a` has a numerical risk worth pre-empting

`g_a = max(r_a, 0)`, `h_a = max(−r_a, 0)` makes the exponent `r_a·(1 − m/k)` for `r_a ≥ 0` and `r_a`
for `r_a < 0` — continuous at `r_a = 0`, but with a **kink** (left derivative 1, right derivative
`1 − m/k`). **L-BFGS with `strong_wolfe` assumes a smooth objective and can fail its curvature
condition at a kink.** The decisions require you to "document treatment near `r_a = 0`" — do so
concretely: state whether you use the subgradient as-is, a smooth (softplus) split, or an offset, and
add a test exercising a fit whose optimum sits near `r_a = 0`.

**Upside worth stating in the plan:** structural zeros + signed `r_a` cut the action parameter count
from **44** (4 × 11) to roughly **14** (≈8 rate incl. shared baseline, 5 capacity, 1 stocking). That is
a ~3× reduction, and it is the strongest identifiability argument you have — quantify it exactly.

### 1.4 Decision 1's Π grid multiplies the PLUS candidate count — price it

Fixed per-candidate `Π` means distinct persistence values become **distinct candidates**. With four
candidates/form today, a `Π` grid of size *G* makes the regime family `G × bootstrap` candidates unless
you hold the total fixed. PLUS already ran **>3 h/cell at 16 candidates**. State explicitly: the grid,
the resulting total bank size, and the cost — and if the bank grows, say what gives.

### 1.5 Subset confirmed

Amur tiger (recoverable) and Egyptian vulture (sink) are both present, and Egyptian vulture is indeed
one of the two registered sinks. 2 populations × 4 families × 4 σ = **32 dynamics cells** ✓.

## 2. What the plan must contain

The 13 items from the decisions document, plus explicit treatment of §1.1–§1.4 above. In particular:

1. the fixed `Π` persistence grid, its expected regime switches over 25 steps, and the **resulting bank
   size and cost**;
2. the exact corrected MOOR objective — `ŷ = x·exp(σ_o²/2)`, MC over **process paths only**, no second
   survey draw;
3. the signed-`r_a` + structural-zero parameterization keyed on the public `channel`, with the
   `r_a ≈ 0` treatment;
4. the revised parameter count (show the 44 → ~14 arithmetic) and identifiability analysis;
5. removal of the three regularizers from the primary objective;
6. the optional hierarchical-shrinkage fallback as a **separate** configuration, not silently activated;
7. the caching design — **transition-field hash, not `dataset_sha256`** — and the fit/plan cost split;
8. the 32-cell diagnostic-subset manifest keys;
9. candidate-count and regime-path MC sensitivity settings;
10. corrected canary runtime plan and full-sweep CPU projection **with and without cache reuse**;
11. exact tests for: conditional-mean survey fitting (synthetic recovery at **σ_o = 0.4**, where the old
    objective is biased −14.8 %; plus a **σ_o = 0** regression proving the fix is inert there), and
    regime fit/deployment consistency (hash-identity of the transition law used in fitting vs
    deployment);
12. confirmation the void run stays preserved and excluded;
13. confirmation that no code/dependency/snapshot/job changes occurred.

## 3. Non-negotiables

- **Do not reopen the five decisions.** If one is genuinely not implementable, report the concrete
  mathematical or server constraint and **stop** — do not substitute a cheaper method.
- Keep the adopted IDs `plus_adapted_mechanistic_pbvi` / `moor_adapted_ricker_misspec_pbvi` and the
  reviewer's display descriptions. No reproduction claim.
- `Π` is a **preregistered candidate parameter**, never described as fitted.
- Fitting, POMDP construction, filtering, and planning must share one regime law.
- No headline sweep is authorized. On the measured ≥1,000 CPU-h, it is not affordable without the
  registered subset rule and an approved CPU ceiling.

Return the plan and stop.
