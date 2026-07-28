# Configuration layout

`configs/tracks/ecological/` and `configs/tracks/general/` are the explicit
track-specific configuration sets. The YAML files directly under `configs/` are
retained byte-identical compatibility copies because archived launch scripts
resolve that historical location; they are not a third scientific track.

`configs/ecology/` is the single authoritative scientific table set. The
load-bearing link `src/tracks/real_ecology_data -> ../../configs/ecology` restores
the repository-root location expected by both byte-preserved packages.
