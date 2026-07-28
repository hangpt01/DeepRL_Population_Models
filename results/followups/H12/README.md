# H12 completed top-up

At Phase-2 construction time, Slurm array `58588371` (`h12_plus_repair`) was still
running. No partial aggregate was copied and the repository build did not wait.
The completed aggregate and receipt were added on 2026-07-29 using
`scripts/top_up_followups.py`.

Completed files:

- `H12_dose_response.csv`
- `H12_degradation_slopes.csv`
- `H12_RECEIPT.json`

Their canonical scratch paths and SHA-256 hashes are recorded in
`provenance/copied_artifacts.sha256`.
