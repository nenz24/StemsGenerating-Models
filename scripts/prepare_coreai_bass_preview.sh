#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$#" -ne 0 ]; then
    printf 'Usage: bash scripts/prepare_coreai_bass_preview.sh\n' >&2
    exit 2
fi
if [ ! -f prototypes/ios27-coreai/ModelCandidates/source-context/reference.json ] || \
   [ ! -f prototypes/ios27-coreai/ModelCandidates/bass-conditioning/reference.json ]; then
    printf 'Prepare both existing components first:\nbash scripts/prepare_coreai_source_context.sh\nbash scripts/prepare_coreai_bass_conditioning.sh\n' >&2
    exit 2
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
environment_python="$repo_root/.runtime-cache/coreai-decoder-venv/bin/python"
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/bass-preview-512"
if [ -e "$bundle" ]; then
    "$environment_python" scripts/export_coreai_bass_preview.py --out "$bundle" --verify-only
else
    "$environment_python" -u scripts/export_coreai_bass_preview.py --out "$bundle"
fi
printf '\nReady: %s\nBuild on iPhone, then tap Generate bass 5,12 detik. Existing bundles retained; no new recording picker.\n' "$bundle"
