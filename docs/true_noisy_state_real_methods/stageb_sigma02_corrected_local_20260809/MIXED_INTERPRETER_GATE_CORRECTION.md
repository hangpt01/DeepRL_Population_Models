# Mixed-interpreter Arm O gate correction

The original gate validated every published frozen fitted object inside the gate's
general Python 3.9 / NumPy 1.23 process. That was correct for general tasks but not for
the four ecological tasks that were prospectively registered and produced under the
paper-faithful Python 3.10 / NumPy 2.2 interpreter.

The corrected gate keeps its general interpreter for orchestration and signing, but
dispatches each fitted-object replay to a separate process launched with the exact
absolute interpreter bound to that task. The child independently freezes the source
registration and verifies its resolved executable, Python version, NumPy version, CPU,
commit, source manifest, task identity, operation input, frozen payload, exact output,
and exact post-call state. Both the serialized payload and the complete parity receipt
must remain byte-exact. Signing-seed environment entries are removed before every child
launch, and the worker rejects their presence.

The old Arm O publications remain bound to their original registration and commit. A
corrected-commit gate may consume them only through the explicit
`corrected_stageb_gate_carry_forward_bridge_v1` contract. That bridge binds the old
registration, DRIVER_INPUTS, immutable output tree and twelve task receipts to the new
gate verifier, while prohibiting Arm O rerun, modification, and return recalculation.
It requires independent audit before execution. A replacement scientific registration
must not claim ownership of the existing Arm O outputs.

Because the prior gate signing seed has already been exercised, a later audited package
must rotate the gate key pair. Arm T remains authorized only by a newly signed gate
receipt after the bridge and all twelve registered-interpreter replays pass.
