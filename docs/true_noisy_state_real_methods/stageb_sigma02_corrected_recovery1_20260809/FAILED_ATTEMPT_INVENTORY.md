# Failed corrected Stage B attempt inventory

Status: **quarantined provenance only; no file in this namespace is trusted, reused,
imported, executed, copied, or used as a scientific source.**

The failed attempt stopped on 2026-08-09 (Australia/Melbourne) while creating
`corrected_arm_o_gate.py`. The observed error was exactly:

`Cannot send after transport endpoint shutdown`

No prospective registration or pre-execution manifest was frozen, no corrected return
was generated, and no Slurm job was submitted. Successful `sacct` accounting for the
date contained only the earlier exploratory `i2b_*` chain; `squeue -u hphung` was empty,
and no dependency receipt referred to the failed corrected namespace.

## Preserved documentation namespace

`docs/true_noisy_state_real_methods/stageb_sigma02_corrected_20260809/`

| Path | Bytes | SHA-256 | Disposition |
|---|---:|---|---|
| `corrected_common.py` | 33,996 | `7d821663669551af6961e589039cfbc694a62bdf2ba69c18d1d13f1af8603e24` | readable; quarantined |
| `run_corrected_task.py` | 29,401 | `be2dadbaf3032b10558f0dcb9febd391e4788351323b2197ace5e92196719dff` | readable; quarantined |
| `corrected_arm_o_gate.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | the formerly unreadable/partial entry; ordinary `stat` later succeeded and showed a regular empty file; never opened or reused |

## Preserved output namespace

`outputs/stageb_sigma02_corrected_20260809/`

| Path | Bytes | SHA-256 | Disposition |
|---|---:|---|---|
| `matched_surrogates/tiger_public_arm_o_surrogate.npz` | 2,725 | `be0f835b7f0a9e8c3c63ca681998acc900678b08776bc6277b50bdfdd4d5da7c` | readable; quarantined |
| `matched_surrogates/fox_public_arm_o_surrogate.npz` | 2,736 | `4d9305a94509e2bb4eacd4b25b8ac4ba138a29e269dd240eba81684b1ecc7a4b` | readable; quarantined |

No other partial file was present within the bounded inventory depth. In particular,
there was no task receipt, evaluation file, scientific return, registration, checksum
freeze, Slurm log, dependency receipt, gate receipt, Arm T result, or finalizer output.

The recovery implementation is reconstructed independently from the controlling plan,
I1/I2A registrations and audits, current frozen source, and the independent failed-results
audit. It does not import or read implementation code from this failed namespace.
