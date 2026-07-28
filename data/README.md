# External data and caches

Raw datasets, private trajectories, demographic fit caches, learned surrogate
blobs, and active run outputs are not committed.

Canonical scratch roots used by the accepted work:

- Ecological project:
  `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`
- Accepted ecological P=10 data/caches:
  `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1`
- Accepted general run:
  `/home/hphung/ce25_scratch2/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1`
- General public datasets/surrogates:
  the accepted general run’s `quarantine/datasets/regime_hidden/reward_safe/`
  subtree.

Dataset identities from the accepted table are in
`provenance/dataset_hashes.csv`. Result and frozen-source identities are under
`provenance/`. Override machine paths through the CLI or the variables documented
in `configs/paths.example.yaml`; do not edit canonical scientific source.

