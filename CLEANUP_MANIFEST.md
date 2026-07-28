# Cleanup manifest

This repository was built copy-only from the Phase-1 approved manifest. Nothing
was moved, renamed, deleted, or edited in the source repositories or run trees.

## Canonical scientific tracks

| Track | Canonical source | Destination | Policy |
|---|---|---|---|
| Ecological | `real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix` | `src/tracks/ecological/` | byte-preserved |
| General | `general_rl_phase2_iso/.../general_phase2e_full_sigma01_02_20260720_v1/code` | `src/tracks/general/` | byte-preserved |

The root working source history is retained under the archive; it is not silently
merged with either accepted track.

## Copy totals from the approved manifest

- KEEP: 566 entries, 6,416,237 bytes.
- ARCHIVE: 107,272 entries, 2,785,772,046 bytes.
- EXCLUDE: 72,661 entries, limited to approved VCS/venv/cache/build/raw
  dataset/model/fit-cache categories.

Artifact-only ARCHIVE entries are retained under `archive/artifacts/` and ignored
by Git so the initial publishable commit has no large run data. Authored archived
code, configuration, tests, and documentation are tracked.

## Approved Phase-2 amendments

- `results/accepted/`: accepted 144-cell CSV and receipt.
- `results/followups/S2/`: CSV outputs and receipts.
- `results/followups/H14/`: M13 table and receipt.
- `results/followups/reward_screen/`: small CSV output and receipt.
- `results/followups/S6/`: S6 receipt, which existed by the Phase-2 copy.
- `results/diagnostic_replay/`: M1–M15 comparison and schema report.
- `results/followups/H12/`: placeholder pending the still-running job.

Every copied result has its canonical source path and SHA-256 in
`provenance/copied_artifacts.sha256`.

## Mechanical changes

No canonical track file changed. Repository-layer diagnostic entry points now:

- select only `scripts/ecological` + `src/tracks/ecological`, or
  `scripts/general` + `src/tracks/general`, as appropriate;
- read accepted config/manifests from `experiments/` and the controlling CSV from
  `results/accepted/`;
- resolve external dataset/cache and output locations through the shared
  `DEEPRL_*` environment pattern in `scripts/diagnostics/repo_paths.py`;
- write generated local replay/follow-up files below `.verification/` by default.

The accepted Slurm launchers retain their historical cluster defaults for
provenance, but Python diagnostic entry points no longer import code from scratch.
Repository-layer files—README, build metadata, profiles, wrappers, provenance,
and verification targets—do not alter frozen scientific behavior.

The exhaustive Phase-1 manifest is retained in compressed form under
`provenance/audit/` and uncompressed in the ignored local artifact archive.
