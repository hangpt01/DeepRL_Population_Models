# M3 Slurm templates

Templates only; they were not submitted or tested against Slurm. Repository, driver,
registration, gate and output paths are supplied through explicit environment variables.
Interpreter paths are never operator-selectable: each script freezes the registration,
resolves the task or command role, and invokes the registered absolute interpreter. The
driver independently checks the configured and resolved executable paths, short and full
Python versions, NumPy version and task/command context before reading inputs. No Mac path
is embedded and `STAGEB_PYTHON`, `STAGEB_GATE_PYTHON` and `STAGEB_FINALIZER_PYTHON` are not
accepted.

The submission template freezes the required dependency chain:

1. Arm O 12-task array;
2. Arm O parity/artifact gate with `afterok:<Arm-O-array>`;
3. Arm T 12-task array with `afterok:<gate>`;
4. inspection-only finalizer with `afterany:<Arm-T-array>`.

The finalizer must never execute or repair a method. Scientific tasks must never be
retried automatically.

Before submission, the prepared package must write `SCIENTIFIC_CHAIN_ENVIRONMENT.json`
with the complete frozen binding for both interpreter roles, both command-role mappings and
the resolved task/command launch map. That environment receipt is evidence only; it cannot
override the registration.

The Python registration and Arm-T tokens remain process-local capabilities. The implemented
cross-process seam exports an Ed25519-signed gate receipt and revalidates its embedded frozen
registration and completed gate before issuing a new Arm-T token. The gate alone receives
`STAGEB_GATE_SIGNING_SEED_FILE`, an owner-only 32-byte private seed; Arm T receives only the
distinct `STAGEB_GATE_PUBLIC_KEY_FILE`. The driver must load/verify these through the provided
functions before constructing any Arm T method. The seed and signed receipt must never enter Git.
