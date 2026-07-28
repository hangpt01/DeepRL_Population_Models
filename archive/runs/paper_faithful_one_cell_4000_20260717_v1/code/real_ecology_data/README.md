# Revised cost + action tables (coding-ready)

Final, implementation-ready specification of the 11-action set, its real-population
parameters, and the per-step costs. Two layers:

- **Relational core** (authoritative, no duplication): `actions.csv`, `species.csv`, `species_lambda.csv`.
- **Precomputed convenience table** (load-and-go for the environment): `action_effects_long.csv`
  — every per-`(species, action)` value already joined, so the env can read one file.

## Provenance
- **Population data + action mechanics** (`N0`, `K`, per-action `lambda`, the a0–a10 vital-rate/K/N rules):
  from **Dominik** (`action_perturbed_lambda_wide.csv`).
- **Cost data** (`ciu`, `cost_step`, `cost_sources.csv`, `cost_anchors_portal.csv`): from a **separate
  cost-function colleague**, grounded in the **Conservation Costing Portal** (`CostPortal_7_26_24.xlsx`,
  90 studies) plus 2 supplementary per-unit studies. Not from Dominik.

## Chosen dynamics: set-point r + cumulative K
Per episode, fix one population `p` (a row of `species.csv`): `s = N0`, `kappa = 0`.
Each step `t`, for chosen action `a`:

```
r_eff   = clip(r_setpoint(a, p), r_min(p), r_max(p))          # SET-POINT: regime, absolute lambda
kappa  += dK_step(a, p)                                       # CUMULATIVE K
K_eff   = clip(K_base(p) + kappa, K_base(p), K_max(p))
s       = s + dN(a, p)                                        # translocation acts on the STATE (a10 only)
s_next  = f_model(s; r_eff, K_eff)                            # Ricker / Allee / theta / regime map
reward  = alpha * o/(o + K_ref) - cost_step(a) - P * collapse # cost paid each step the action is held
```

- `r_setpoint(a,p)` = `ln(lambda)` (Ricker, primary) or `lambda - 1` (LGM). a0 → r_base.
- Clip bounds are **data-derived**: `r_min = r(lambda_a2)` (heaviest exploitation),
  `r_max = r(lambda_a4)` (strongest recovery), per population.
- `dK_step` = `(K_multiplier - 1) * K_base`; `K_max = 2 * K_base` (tunable landscape ceiling).
- `dN` = `dN_fraction * N0` (10% of N0, a10 only).
- `cost_step` in `[0,1]`; harvest is revenue (negative), so the effective range is `[-0.10, 1.00]`.

## Files
| file | rows | what |
|------|------|------|
| `actions.csv` | 11 | action menu (species-independent): both names, channel, lambda_source, K_multiplier, dN_fraction, ciu, **cost_step**, interpretation, cost_range, cost_source_keys |
| `species.csv` | 9 | populations: N0, K_base, K_max, and set-point clip bounds r_base/r_min/r_max (Ricker + LGM) |
| `species_lambda.csv` | 45 | the **distinct** measured growth rates (9 species × 5 lambda_id) with r_ricker / r_lgm |
| `action_effects_long.csv` | 99 | **precomputed** species×action: lambda, r_setpoint (Ricker+LGM), dK_step, dN, cost_step, clip bounds — the env's lookup table |
| `cost_sources.csv` | 12 | source keys behind cost_step (portal + supplementary), with figure, CMP category, origin |
| `cost_anchors_portal.csv` | 12 | the portal studies carrying a numeric cost, extracted verbatim from `CostPortal_7_26_24.xlsx` |

## How the relational core joins (if you don't use the precomputed table)
```
lambda     = species_lambda[common_name, lambda_id = actions.lambda_source].lambda
r_setpoint = ln(lambda)  (Ricker)  or  lambda - 1  (LGM)
dK_step    = (actions.K_multiplier - 1) * species.K_base
dN         = actions.dN_fraction * species.N0
cost_step  = actions.cost_step           # same across species
clip bounds: species.r_min_*, species.r_max_*, species.K_max
```
`action_effects_long.csv` is exactly this join precomputed; regenerate it from the core if you edit inputs.

## Action set (a0–a10)
a0 Do Nothing · a1 Sustainable Harvest · a2 Aggressive Harvest · a3 Predator/Disease Control ·
a4 Breeding/Recruitment Support · a5 Moderate Restoration · a6 Intensive Restoration ·
a7 Integrated Conservation (light) · a8 Adaptive Conservation Trial · a9 Flagship Conservation Programme ·
a10 Translocation (new 11th action).

cost_step (a0→a9): 0.00, −0.05, −0.10, 0.25, 0.50, 0.1875, 0.50, 0.4375, 0.75, 1.00; a10 = 0.3125.

## Cost grounding
ordering grounded in the portal compilation: habitat/land management is the cheap recurring end
(Adams AU$1–2/ha/yr; Green US$2.3–8.3/ha/yr; James US$893/km²/yr); species management is the expensive
end (Laycock £500–7M/species; McCarthy up to US$24,487/ha; condor >US$35M). The portal resolves cost to
the broad CMP category, so within-Species-Management ordering uses Weise 2014 (translocation) and the condor
programme (breeding). `cost_step` is a relative [0,1] index — it encodes the **ordering/spacing**, not dollars;
the [0,1] scaling compresses the orders-of-magnitude spread so the reward keeps resolution.

## Source URLs
- Paper: https://academic.oup.com/bioscience/article/72/5/461/6549353
- Costing portal: https://jcleme19.wixsite.com/costingportal/cost-studies
- Yong 2023: https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2664.14377
- Weise 2014: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0105042
