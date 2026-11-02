#!/usr/bin/env bash
# Reuse caches; preserve every previously prepared model/audio bundle.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$#" -ne 0 ]; then
    printf 'Usage: bash scripts/prepare_coreai_bass_conditioning.sh\n' >&2
    exit 2
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
environment_python="$repo_root/.runtime-cache/coreai-decoder-venv/bin/python"
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/bass-conditioning"
if [ -e "$bundle" ]; then
    "$environment_python" scripts/export_coreai_bass_conditioning.py --out "$bundle" --verify-only
else
    "$environment_python" -u scripts/export_coreai_bass_conditioning.py --out "$bundle"
fi
printf '\nReady: %s\nRebuild on iPhone, then tap Uji conditioning bass. No recording picker in this step.\n' "$bundle"
