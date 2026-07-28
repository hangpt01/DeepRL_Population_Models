# H12 pending top-up

At Phase-2 construction time, Slurm array `58588371` (`h12_plus_repair`) was still
running. No partial aggregate was copied and the repository build did not wait.

Expected completed files:

- `H12_dose_response.csv`
- `H12_degradation_slopes.csv`
- `H12_RECEIPT.json`

Run `scripts/top_up_followups.py --scratch-project PATH` after all three exist.

