# M3 deployment handoff for corrected Stage B sigma=0.2

The local package is development infrastructure only. Every M3 result remains
`PROVISIONAL — NOT YET INDEPENDENTLY AUDITED` until a separate read-only audit passes.

## 1. Clean-clone and commit gate

1. Review the complete local diff, then create and push an authorized commit. No commit was
   created by the local implementation session.
2. Start from a fresh approved clone/worktree on a healthy M3 shared filesystem. Do not
   deploy from either failed corrected namespace or the damaged server worktree.
3. Record and verify the exact 40-hex commit SHA. Confirm the target tree is clean using
   narrow operations compatible with the registered Git-object/SIGBUS restriction.
   The synthetic test fixture must independently derive this identity with
   `git rev-parse HEAD`, verify that it names a commit, and reject malformed or additional
   output. It must never contain a static pre-commit SHA. Production registration validation
   still independently requires the frozen SHA to equal checked-out `HEAD`.
4. Recompute the registered source-only recipes. Required values are:
   ecological `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e`;
   general `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd`.
5. Verify Revision 3.1, I1/zero-SD, I2A, fast-track, the new source/test manifest and all
   registration templates by SHA-256. Reproduce the manifest's `SELF-NORMALIZED-SHA256`
   after replacing its one recorded self-digest with 64 zeroes. Do not run broad `git fsck`, `git gc`, `git repack`,
   full history traversal or any server repair workflow.
   Freeze that normalized self-digest as `source_manifest_sha256`; do not substitute the
   manifest's ordinary whole-file digest.

## 2. Environment recreation

- Architecture: Intel Xeon Platinum 8452Y; Slurm constraint `xenon-8452Y`; partition/QOS
  `m3h`; account `ce25`; one CPU per task.
- Ecological tasks: accepted paper-faithful interpreter, independently revalidated.
- General tasks: registered `/usr/bin/python`, independently revalidated.
- Set `LC_ALL=C`, `PYTHONDONTWRITEBYTECODE=1`, and all registered BLAS/thread variables to
  one.
- Recreate only the pinned registered environments. Do not substitute the ARM64 virtual
  environment, broaden versions, or treat a Mac wheel as parity evidence.
- Run 38 I2A + 15 fast-track + all corrected tests in the pinned M3 environments. Seal a
  test receipt before any return path is opened.

## 3. Accepted input/configuration gates

Required configuration SHA-256 values:

- general: `4e878a9ec7d528af0bfb4948e78036dc864f0d2b158a2fdf5559b24798b1222b`;
- PLUS: `10a064344e07cac16a7b2d5bd717bdce109c8970148de4295d6ff3532297fa68`;
- MOOR: `a3655a846d951439177da350ddf40ddc6d4962b42b317582dd29e72ca05323bd`.

Required public dataset identities:

- tiger public file `9e9c3a6d4bcee3c27c785f46ce17b66e1c9099c446213f46ebcfa3c3851f771c`,
  embedded dataset `7e71172af4bc95b2c31291414af53371c1ea94393e01675d269b2b7c360324c9`;
- fox public file `7d1b2fc8a1949dda6f53f39eff1dbfe220719d0cbad9f8956ab7ca2d08e87e19`,
  embedded dataset `688d580f1e47a61dcad9b8a6cb96f1d62835bfbd0b835554cf31c8ab8bdd0c13`.

Required accepted derived offline identities, bound from accepted server artifacts only:

- tiger `37eace7473df7d78e3c0468d902fd971826d7b42c5a0732ca7f3df04f8ed90dc`;
- fox `2d0d2d11e90bae1539576cbcbd4b20156281bc861bd0fdbb1695529ebd06c920`.

Never copy ignored Mac/server output into Git. Methods must never open original
`truth.npz`; runtime `next_states` remains impossible.

## 4. Prospective freeze and fitted artifacts

Before creating any corrected return path:

1. Fill all six registration sections, including explicit authorization, exact commit/code/
   configuration hashes, all 24 task rows and the known-results disclosure.
2. Freeze the bundle with status `FROZEN_BEFORE_CORRECTED_RETURNS` and attest that no
   corrected return exists. The earlier sigma=0.2 results are known; this is not blinded.
   The local `FrozenRegistration` is an issued non-serializable process capability; direct
   construction or reuse of registration-looking bytes must remain rejected.
3. For each cell, bind or prospectively reconstruct exactly one Arm-O-derived surrogate
   before either arm. RefPlan, OGSRL and BA-MCTS must load identical serialized bytes in O
   and T; Arm T fitting is impossible. If no historical serialized artifact exists, label it
   exactly `PROSPECTIVELY RECONSTRUCTED MATCHED ARM-O SURROGATE`.
4. Accepted historical surrogate candidates, if independently validated, have recorded
   hashes tiger `be0f835b7f0a9e8c3c63ca681998acc900678b08776bc6277b50bdfdd4d5da7c`
   and fox `4d9305a94509e2bb4eacd4b25b8ac4ba138a29e269dd240eba81684b1ecc7a4b`.
   Failed-attempt copies are forbidden even when their hashes match.
5. Serialize every applicable component named in `REQUIRED_COMPONENTS`, then serialize the
   complete actual frozen policy/dynamics instance graph through `real_artifacts.py`; reload
   fresh objects and run the registered real prediction/action operation. Training histories
   never qualify and generic canonical probes cannot substitute for real-object parity.
6. PLUS/MOOR must bind one Ricker cache, ecological surrogate and complete PBVI policy per
   method/cell across both arms. Only the registered abundance/belief interface may differ.
7. If any fitted state is inaccessible without changing `src/tracks/**`, stop. Do not patch
   the frozen tracks or fabricate an identity.

## 5. Arm O parity/artifact gate

The twelve Arm O tasks must pass before Arm T is structurally available. The gate verifies:

- all task/input/configuration/registration/source/surrogate/policy hashes;
- complete serialized/reloaded artifacts, dereferenced from each immutable published task tree,
  with an exact registration/cell/public-view/logical-component binding;
- independent gate-side replay of each published real frozen-object operation input and parity
  receipt—CanonicalArtifact-only evidence and caller-authored parity claims are insufficient;
- strict, recognized Arm O transition/activity/posterior/dispersion and information-boundary
  receipts; arbitrary nonempty diagnostic maps are forbidden;
- registered 8452Y CPU/profile and interpreter roles;
- 4,000 rows = 160 complete episodes × 25 transitions;
- all twenty ordered identities, horizon 50, discount 0.95 and eleven actions;
- strongest accepted per-episode return, action/event and scientific-field parity at the
  registered tolerance—never aggregate-only agreement;
- no truth/archive access, forbidden field, runtime future state, context/RNG mutation or
  sigma collapse.

Any mismatch exits nonzero. The gate may not rebaseline, repair, retry, or submit a
replacement. Arm T must depend on successful gate completion. `export_arm_o_gate_receipt`
signs the immutable completed-gate and registration bytes with Ed25519;
`verify_arm_o_gate_receipt` revalidates both and issues a new process-local `ArmTGateToken`.
Provision an owner-only 32-byte private seed through `STAGEB_GATE_SIGNING_SEED_FILE` only to
the gate job, and its derived 32-byte public key through `STAGEB_GATE_PUBLIC_KEY_FILE` only
to Arm T. Never place the seed or a receipt in Git, and independently audit the M3 driver wiring.
Arm T must fail before method construction if the key, signature, embedded bytes, commit,
manifest, templates, source tracks or registration bindings differ.

## 6. Dependency chain and publication

Configure the M3 environment variables documented under `slurm/`; no path is hard-coded.
The gate driver must call `load_gate_signing_key`; Arm T must call
`load_gate_verification_key` and never receive the signing seed.
Use the frozen chain only:

1. Arm O 12-task array;
2. parity/artifact gate with `afterok:<Arm-O-array>`;
3. Arm T 12-task array with `afterok:<gate>`;
4. inspection-only finalizer with `afterany:<Arm-T-array>`.

Each scientific task writes exclusively to a unique task-local directory, validates and
checksums the complete tree, fsyncs it, and publishes atomically once with the directory
rename as its final filesystem operation. No operation may fail after a success receipt is
visible under the final target. No automatic retry is permitted. The finalizer inspects
identity-bound terminal states only; it never runs or repairs methods.

## 7. Final independent audit boundary

A new independent read-only audit must verify registration timing, code/config/data hashes,
surrogate and policy byte identity, full fitted-object serialization/reload, residual floors
and O/T ratios, realized BA-MCTS posterior/RefPlan dispersion, exact evaluator-only return
reconstruction, complete action-derived activity flags, pairing/RNG receipts, access
boundaries, dependency satisfaction and output sealing.

Do not start sigma=0.1, tune hyperparameters, add seeds, modify slides or describe corrected
returns as accepted before that audit.
