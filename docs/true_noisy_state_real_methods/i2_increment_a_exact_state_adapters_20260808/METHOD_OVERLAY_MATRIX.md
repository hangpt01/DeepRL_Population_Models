# I2A method-overlay matrix

This matrix is a design/contract artifact only. No method is integrated and no learned
artifact is built.

| Method | State interface | Context preserved | Point mass / scale | Preprocessing | Possible changed axes | Eligibility |
|---|---|---|---|---|---|---|
| adapted PLUS | Eight candidate latent-grid beliefs | Action history and timing | Direct point mass; raw abundance divided by each fitted `survey_scale`, then nearest registered latent bin | Public zero-SD branch is vacuous | None allowed for frozen-fit; realized candidate sharpness still reported later | Frozen-fit only after every required identity is valid and exactly equal, plus byte-identical eight-kernel/policy proof and direct replacement demonstration |
| adapted MOOR | One latent-grid belief | Action history and timing | Direct point mass; raw abundance divided by fitted `survey_scale`, then nearest registered latent bin | Public zero-SD branch is vacuous | None allowed for frozen-fit | Frozen-fit only after every required identity is valid and exactly equal, plus byte-identical kernel/policy proof and direct replacement demonstration |
| RefPlan | Exact current-abundance public particles/features | Previous/current observation, timestep, action and observation history | Public interface; no `survey_scale` | Primary refit later; optional secondary reuses Arm O bytes/hash | Dynamics, `residual_sigma`, posterior sharpness, behavior prior | Primary end-to-end; secondary only when explicitly enabled after every required identity is valid and exactly equal |
| OGSRL | Exact current-abundance public features | Previous/current observation, timestep, action and observation history | Public interface | Later end-to-end actor/guardian preprocessing | Dynamics, ensemble, actor, guardian, dataset-derived safety budget/calibration | End-to-end only |
| BA-MCTS | Exact public search-root features/particles | Previous/current observation, timestep, action and observation history | Public interface | Later end-to-end | Dynamics, ensemble, `residual_sigma`, model-posterior sharpness | End-to-end only |
| EVD pessimism | Exact current-abundance public features | Previous/current observation, timestep, action and observation history | Public interface | Later behavior/Q rebuild | Behavior reference and 20-Q ensemble; unmatched raw-reward objective | End-to-end only; separately reported |

For every method, sigma/observation-scale collapse, wholesale oracle-filter reuse,
history loss, direct access to `truth.npz`, runtime `next_states`, private parameters,
and edits to `src/tracks/**` are forbidden. Any changed learned artifact forces
`MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY` and bars a causal state-representation
label.

For the three methods with a possible frozen-fit route, each arm must provide all eight
registered learned-artifact SHA-256 values as exactly 64 lowercase hexadecimal
characters and a nonempty tuple of canonical finite float64 `residual_sigma` hex
values. Missing, placeholder, sentinel, prefixed, uppercase, or malformed identities
fail closed. Equality is evaluated only after both arms validate; equality alone never
satisfies the method-specific condition.
