# Revision 3.1 changelog — method-level state-information pilot

Date: 2026-08-08 (Australia/Melbourne)  
Revision 3 source: `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md`  
Revision 3 SHA-256 before and after Revision 3.1: `164a536bb9a0afc4d78b42189ac119a089677bf6b83129d739ecb7b7ba103290`  
Revision 3 changelog SHA-256 before and after Revision 3.1: `5e1d4da1a97b5b8e4b8607b0888fb7d915e4c176520e82ce9ed45933ef266f7d`  
Controlling audit: `docs/true_noisy_state_real_methods/CLAUDE_REV3_READONLY_AUDIT.md`  
Controlling audit SHA-256: `3139e575859badd11c4b4ef8cc3d2d058fe312598e622831aa11d6795f6167cb`  
Revision 3.1 target: `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3_1.md`  
Revision 3.1 SHA-256: `701b4509f1dc04b7885891be72e526420ac5558be56bdde2b9628789398fe959`

Revision 3 was copied to the new Revision 3.1 document and amended there. Revision 3,
its changelog, all audits, I0 artifacts, source code, tests, configurations, registrations,
accepted outputs and frozen tracks were not edited.

## Four-amendment compliance table

| Amendment | Source audit section | Revision 3.1 location | Status | Verification |
|---|---|---|---|---|
| 1. Correct activity-record explanation | §3.1 and §9 item 2 | G4, lines 545–551 | **APPLIED** | Records the same files/hashes, 12-of-61 display truncation, column 25, exact eight-value reproduction, and verified I0/G4 labels; gate unchanged. |
| 2. Name and bind exact-state source | §5.2 and §5.4 | §5 item 1, lines 225–244 | **APPLIED** | Names both absolute `truth.npz` paths and full hashes, 4,000-row alignment, corresponding public artifacts and full bindings, accepted cell identities, and audit noise checks; states `public.npz` has no true abundance. |
| 3. Add strict truth-field allowlist | §5.3 and §5.4 | §5 item 1, lines 246–281; I2 lines 756–767 | **APPLIED** | Only `states` and conditionally scoped `next_states` are scientific truth inputs; every private category is excluded; schema, finiteness, units, order and binding fail closed; runtime future access prohibited. |
| 4. Register and isolate private-truth override | §5.3 and §5.4 | §5 item 1, lines 283–303; I2 lines 756–769; authorization lines 867–881 | **APPLIED** | Registers the `pipeline.py:432` override only for two exact-state cells, external extraction, a new authorized namespace, Arm O isolation, hash/alignment receipts, provenance retention policy and leakage tests. |

No amendment is partially applied or blocked.

## Amendment 1 — correct the activity-record explanation

### Claude's controlling requirement

> “The ‘different files’ explanation is inaccurate. Revision 3 line 471 and changelog line
> 181 attribute my prior error to inspecting ‘different’/‘shorter’ files. The hashes are
> identical to the eight files Revision 3 lists; all twelve accepted episode files
> (ecological and general) carry 61 columns and `action_entropy`. Replace with: ‘Claude's
> prior audit truncated the column listing and reported the column absent; the files are
> the same and the column is present at index 25.’”

### Old Revision 3 wording

> “The shorter source-repository episode summaries inspected during Claude's audit are
> different files and do not contain that column.”

### New Revision 3.1 wording

> “Claude's prior audit inspected these same accepted files and hashes but displayed only
> the first 12 of their 61 columns, then incorrectly reported the action column absent.
> `action_entropy` is present at column 25, all eight I0 activity-entropy values reproduce
> exactly, and the I0 activity table and G4 activity labels are verified.”

- Final location: lines 545–551.
- Status: **APPLIED**.
- Verification: the correction expressly leaves the decision-activity gate unchanged.

## Amendment 2 — name and bind the exact-state source artifact

### Claude's controlling requirement

> “The exact-abundance source is
> `<GEN>/quarantine/private/regime_hidden/reward_safe/<species>/allee/sigma_0p2/truth.npz`,
> SHA-256 `1658f587cc2b1144362bd90182efce1bd33b65d35c47c1e9dae4320d705b8668`
> (tiger) and
> `4149e293d70a59b000a3b947ce663fb3de282fb404cea857f02be00718a467e5`
> (fox). Both carry `public_dataset_sha256` matching the accepted datasets; I1 must verify
> that binding before use.”

### Old Revision 3 wording

> “In a new output namespace, general methods receive a derived exact-state view whose
> current/next state fields come from the pinned true-abundance fields.”

Revision 3 did not state that accepted `public.npz` has no true-abundance field and did
not name, path, hash or bind the actual truth source.

### New Revision 3.1 wording

> “The accepted `public.npz` contains no true-abundance field: its 13 keys are public-only.
> For the four general methods, the offline Arm T dataset **view changes** while row
> identities, row ordering, splits and episode structure do not.”

Lines 237–244 then register both full absolute `truth.npz` paths, full source hashes,
4,000-row alignment, corresponding accepted public paths and file hashes, full embedded
`public_dataset_sha256` values, accepted cell identities, and the verified
`sd[log(observation/state)]` checks of 0.19942 for tiger and 0.19655 for fox.

- Final location: lines 225–244.
- Status: **APPLIED**.
- Verification: both absolute paths existed and their read-only SHA-256 values matched
  Claude's full truth hashes. I1 must reverify and freeze all identities before use.

## Amendment 3 — strict truth-field allowlist

### Claude's controlling requirement

> “The Arm T adapter may read **only** `states` and `next_states`. Reading `r_base`,
> `r_eff_true`, `C`, `theta`, `regime`, `next_regime`, `entry`, `reward_true`,
> `initially_unsafe`, `safety_penalty_applied` or any part of
> `metadata_json.environment` is prohibited. `metadata_json` may be opened solely to
> verify `public_dataset_sha256`. I2 must add a test that fails closed if any other field
> is dereferenced.”
>
> “`next_states` may be used only to build the offline transition target. It must never
> reach an online adapter, deployment feature or planning root.”

### Old Revision 3 wording

Revision 3 prohibited hidden family, threshold, evaluator constants, future state,
randomness, hidden `r` and hidden `K` in general terms, but supplied no source-field
allowlist, no direct-access boundary, no schema check and no explicit offline-only
`next_states` rule.

### New Revision 3.1 wording

> “Methods and training pipelines must never open or read the original `truth.npz`
> directly. The dedicated external extraction stage is the sole permitted reader.”

Only current `states` and, where a primary end-to-end method refit requires it,
same-row `next_states` as one-step supervision are allowed. Lines 255–269 prohibit
deployment/future access and enumerate every forbidden private category. Lines 271–281
require exact I1 source/derived schemas, row/order/binding/unit/finiteness checks, and
fail-closed aborts. I2 leakage assertions appear at lines 756–767.

- Final location: lines 246–281 and 756–767.
- Status: **APPLIED**.
- Verification: `next_states` is unavailable to frozen-fit diagnostics, online adapters,
  deployment features, action selection and planning roots; all non-allowlisted truth is
  excluded.

## Amendment 4 — register and isolate the private-truth override

### Claude's controlling requirement

> “`pipeline.py:432` states that truth must never be cached for training. Arm T
> deliberately overrides this for the derived exact-state view only, outside
> `src/tracks/**`, under the field allowlist above. Record the override explicitly in the
> I1 registration.”

### Old Revision 3 wording

Revision 3 required a derived exact-state training view but did not acknowledge that this
deliberately overrides the no-private-cache invariant near `pipeline.py:432`, nor did it
define custody, isolation or retention requirements for the derived artifact.

### New Revision 3.1 wording

> “Arm T deliberately overrides that invariant, narrowly and only for this preregistered
> diagnostic's two registered pilot cells and exact-state arms.”

Lines 283–303 restrict the override to an external stage outside frozen tracks, immutable
accepted artifacts, a future separately authorized namespace, source/derived hashes and
row alignment, method-level denial of original truth access, Arm O isolation, removal of
all evaluator-private fields, and a preregistered destruction/retention policy. Lines
756–769 require I2 leakage tests and prove the original invariant remains absolute
outside the isolated path.

- Final location: lines 283–303, 756–769 and 867–881.
- Status: **APPLIED**.
- Verification: Revision 3.1 creates no derived artifact and performs no override; it
  documents the future authorization and fail-closed test boundary only.

## Preserved Revision 3 safeguards

Revision 3.1 retains the end-to-end method-bundle estimand, separate frozen-fit diagnostic,
context-preserving adapters, PLUS/MOOR direct point-mass assignment, zero-noise and
all-zero-likelihood prohibitions, tiger/fox activity roles, primary raw paired loss,
no cross-species pooling, EVD's separate objective, sigma-0.2 screening status, conditional
separately authorized Stage C, source-only hashes, bytecode protections, immutable accepted
artifacts, all stop gates and independent audits.

## Remaining registration requirements and authorization

The audit did not enumerate the exact complete `truth.npz` archive key set or the
current/next-state unit declaration. Revision 3.1 therefore makes both mandatory I1
registration items and prohibits extraction until they are frozen and verified. Claude's
non-blocking recommendation to preregister a `residual_sigma` interpretation threshold
or an explicit report-only rule also remains an I1 registration requirement.

I1, I2, truth extraction, derived-artifact construction, registration/output namespaces,
regression, rebaseline, Stage B, Stage C and scientific execution remain unauthorized.
No existing file was edited and no scientific data was extracted or copied.
