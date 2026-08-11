# CPU guard and durable-log correction

## Preserved failed attempt

Scientific chain `59006293`–`59006296` is preserved as:

`TECHNICAL FIRST-PAYLOAD STARTUP FAILURE — CPU GUARD REJECTED REQUIRED CPU BEFORE PUBLIC DATA OPEN`

All twelve Arm O array elements stopped at the first scientific-environment guard with exit 2.
No public dataset was opened and no policy, evaluator, episode, return, role publication or role
receipt was created. The original output root retained only its unchanged canonical
`DRIVER_INPUTS.json`. The impossible gate, Arm T and finalizer jobs were subsequently cancelled;
Arm O was not altered. Historical submission evidence and independent failure-audit namespaces
remain unchanged.

The failed scripts inherited Slurm's default output destination from a `/tmp` working directory.
Those node-local logs are unrecoverable. This disclosure is permanent and is not replaced by the
new logging contract.

## Canonical CPU identity

The registered label remains `Intel Xeon Platinum 8452Y`. The common runtime helper removes only
registered trademark markers `(R)` and `(TM)`, normalizes whitespace, then compares the complete
normalized identity. It accepts the node's real `Intel(R) Xeon(R) Platinum 8452Y` value and the
normalized label. It rejects other Gold or Platinum models, suffixes, prefixes, and unrelated
strings that merely contain `8452Y`.

The same helper is imported by scientific runtime, login/compute preflight and the pinned-CPU
validation job. Its canonical receipt records raw and normalized expected and observed identities.
Arm O and Arm T publication receipts embed and validate that receipt.

## Registered durable logs

Every replacement registration includes a `scientific_log_plan` derived from two disjoint absolute
roots: the scientific output root and a shared `/fs04` evidence root. Exact stdout and stderr
templates are deterministic:

- Arm O and Arm T arrays use `%A_%a`;
- the gate and finalizer use `%j`;
- every role and stream has a distinct path;
- all paths are strictly below the shared evidence root and outside scientific results.

Preparation creates the directories as owner-only real directories. Validation rejects relative or
node-local roots, path traversal, symlink rebinding, template changes and collisions. The four-call
template resolves the frozen plan before submission and passes explicit `--chdir`, `--output` and
`--error` values on every call. Direct node SSH is neither required nor used.

## Scientific scope

This correction does not change a method, cell, dataset, scientific setting, evaluation identity,
seed, fitted object, artifact plan or replay closure. The previously audited 36 fit/replay receipts
may be carried forward only after byte-for-byte closure-equivalence verification. A tracked-source
change requires a new commit, registration, canonical DRIVER_INPUTS and owner-only keypair before
any later scientific authorization. The corrected test count increases from 249 to 269 solely for
the new CPU-identity, durable-log and registration-binding regression coverage.
