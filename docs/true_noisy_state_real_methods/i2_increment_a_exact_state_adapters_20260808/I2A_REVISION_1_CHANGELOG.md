# I2A Revision 1 corrective changelog

Date: 2026-08-08 (Australia/Melbourne)  
Namespace: `docs/true_noisy_state_real_methods/i2_increment_a_exact_state_adapters_20260808/`  
Controlling Claude audit SHA-256: `745e7a72b38fabb8b529ab0cd6f59134c2a26eb2f7b58eb934b8412f75913ca3`

## Corrective scope

This bounded patch addresses the controlling `REVISE` finding and the three specified
provenance weaknesses without integrating an adapter or accessing scientific data.

- Learned-artifact identities now validate before equality. For each frozen-fit-capable
  method, both arms require eight SHA-256 values in exactly 64 lowercase hexadecimal
  characters and a nonempty tuple of canonical finite float64 `residual_sigma` hex
  strings. Missing, null, sentinel, whitespace, malformed, prefixed, uppercase,
  unresolved or unexpected fields fail closed with an explicit reason.
- Exact equality is necessary but insufficient. RefPlan also requires explicit optional
  secondary enablement; PLUS/MOOR also require demonstrated direct point-mass-only
  replacement. Any valid change on any of the nine registered axes forces exactly
  `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`.
- Context parity uses independently captured caller-input and emitted-output snapshots,
  receipts both hashes, and compares protected features, action and observation
  histories, feature order, dtype and shape.
- Fit/runtime preprocessing parity serializes the registered artifact using strict
  canonical JSON, reloads a distinct object, and checks artifact hash, output bits,
  schema, dtype, mask, offset bits, scale bits, aliases and rank receipt.
- RefPlan disclosure fields are mandatory finite numerics, strict-JSON round-trippable,
  species-labelled against tiger `-2.7018` and fox `-2.7348`, and internally consistent
  with the mismatch-removal flag.

## Changed-file hashes

| File | Pre-revision SHA-256 | Post-revision SHA-256 |
|---|---|---|
| `README.md` | `b681a060b95eff7f0c5d1df76630169a5891a33bf116b8e5272e542105a7d6e0` | `47a01f6e4cf7e8769c89d37ed8121743a2b52684f8a44d7a764d21867e665684` |
| `adapter_interfaces.py` | `5d0f2f7c14a6a21fc3ab3eade0bbdca2233e046a37aa3fd6b4486cb5f01afeed` | `f145b8bd7694c737b553b7ef729c26e636f9f275862ce1aa9397044e19d65a84` |
| `test_i2a_contracts.py` | `379fac530c505521bf6b7a636e6217847aff34d78686649c825b1af7d4726966` | `41d8cf9dec5c48a55e66ee15a42879e4b3e594128b1cb86e7232536bc8f522e2` |
| `METHOD_CONTRACTS.json` | `5df3a6f409b38a8342bf16d83af94b0068517eb43db5e33aefb6dffafd5352b2` | `7bde9f512c21a4c21000a7d29d6c36e09fadc285313c8ed5f3eb698e7014a130` |
| `METHOD_OVERLAY_MATRIX.md` | `e3d1090230eab9d1111965d7ce3006cce15030d56c8721a51a946e181ac91a0b` | `96f47289d55f98e80c07a9c2efd5b0c5726a3e0ae130e4227000c4c3264cec79` |
| `PREPROCESSING_RECEIPT_SCHEMA.json` | `64f5d07f673c798689c5dec1f59a4aa0640f366f1ee1943e6908d60455e07165` | `c4473d6506190ae147807dca1707ca61db9609397451996b5e8701098ba766be` |
| `SYNTHETIC_PREPROCESSING_RECEIPT.json` | `13316dec8249bb591c5fc24479cff4be92bd02c50b12034141a12153baa86c8b` | `73858c796bfe291e0be8f7c6492c3297b361f632fb04177a4d991b235685b92e` |
| `TEST_REPORT.md` | `774a472230916b8b530b5ce5c10e4d5c50ed53e9b3fb79e8aceb2d37a453fc2f` | `fcfcfea42443c909706c8f8a9c4eb03ce5ad54cf5bb5349554d85b593c8b0b7e` |
| `I2A_COMPLETION_REPORT.md` | `b7b3f8724c3a3dadee3f60800dcfad4ddf5436f668be44b1a80a5af2a473e3c1` | `6a939cab6a0f9e1a5f654e1d29a3619b76e9845d0c12d78fbcdb8e98075e9168` |

`I2A_REVISION_1_CHANGELOG.md` is new (pre-revision state: absent), and its post-revision
SHA-256 is recorded by the regenerated manifest. `I2A_HASHES.sha256` changes from literal
SHA-256 `69705b5208fe9c6d378a7dbeb2b0fd4cf004b0d587e98d3bc99f378195147fd5`;
its post-revision identity is the independently verified normalized self-hash recorded
inside that file. Those two post identities cannot also be embedded here without making
the changelog and manifest recursively self-referential.

## Tests added and retained

Six named test methods were added, bringing the suite from 32 to 38 while retaining all
original tests:

- independent context mutation detection;
- incomplete/changed serialization rejection;
- exhaustive malformed/omitted identity rejection;
- valid-equal identity plus method-condition enforcement;
- non-finite and inconsistent RefPlan rejection;
- fox-reference and retained-mismatch RefPlan validation.

The exhaustive change-detection matrix now covers all nine registered axes across
RefPlan, PLUS and MOOR. The malformed matrix covers every required field for all three
methods, including both arms jointly, each arm separately, omissions, sentinels and all
specified digest-length/format failures.

Final results from fresh `/tmp/i2a-rev1-final.dMQfUT`:

- `unittest`: 38/38 passed in 0.049 seconds;
- pytest 9.1.1 with `-p no:cacheprovider`: 38/38 passed;
- bytecode disabled; temp contents verified empty; exact temp directory removed.

## Unchanged I2A files

| File | Unchanged SHA-256 |
|---|---|
| `CLAUDE_I2A_READONLY_CODE_AUDIT.md` | `745e7a72b38fabb8b529ab0cd6f59134c2a26eb2f7b58eb934b8412f75913ca3` |
| `INFORMATION_ACCESS_RECEIPT.json` | `46e83a14a49c9f10aeb2f920d69111236287fde51a9aee1d479d4b1fd6dfd5aa` |
| `__init__.py` | `20584740ebd1c120131c1ca71d8b9d6812790f28ca010bc46de209181dff9345` |
| `information_boundary.py` | `6aeee51006a4722d0e02183386fa997b3628977039ed25a82443d110895f4c7a` |
| `synthetic_fixtures.py` | `9b2b94063d176cbd1ef789627a1f3c438bba0d767c9203de76caaa1856d1343f` |

## Authorization boundary

Only implementation, synthetic tests, contracts, receipts, reports and checksum
artifacts inside the I2A namespace changed. The Claude audit was read but not modified.
No real truth/public array or dataset was accessed; no adapter was integrated; no model
or policy was fitted or evaluated; no execution namespace, regression, rebaseline,
Stage B/C run or Slurm job occurred. I1, Revision 3.1, sealed predictions, accepted
outputs, frozen tracks, repository source/config/tests and Git objects remain protected.
I2B remains unauthorized and was not begun.
