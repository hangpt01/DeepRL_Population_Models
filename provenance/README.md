# Provenance

- `copied_artifacts.sha256`: committed result destination, SHA-256, and canonical
  scratch source.
- `frozen_tracks.sha256`: every canonical track file and its source counterpart.
- `scientific_inputs.sha256`: complete coverage of `scripts/ecological/`,
  `scripts/general/`, `configs/ecology/*.csv`, and `experiments/`.
- `dataset_hashes.csv`: dataset identities extracted from the accepted 144-cell
  table.
- `audit/`: Phase-1 audit, exact-source duplicates, hygiene findings, and complete
  file manifest.

Canonical frozen roots:

- Ecological HEAD `a69f3b0cbf61a9c8802f506979bd0e454125ec02`.
- General enclosing HEAD `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536`.
- General 576-row manifest SHA-256
  `1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4`,
  copied from
  `/home/hphung/ce25_scratch2/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/manifests/full_general_sigma01_02_576_rows.csv`.
- Accepted result SHA-256
  `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`.
