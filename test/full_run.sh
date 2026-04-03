#!/usr/bin/env bash
set -euo pipefail
trash-put out/minimal_testrun/
for a in embed all; do
    python3 scripts/attention_embed.py "test/minimal.toml" \
        --in_boundary_mode "$a"
    python3 scripts/lahis.py "test/minimal.toml" \
        --in_boundary_mode "$a"
    for b in attention_mass lahis; do
        python3 scripts/patchscopes.py "test/minimal.toml" \
            --in_boundary_mode "$a" \
            --ablation_map_type "$b"
    done
done
