#!/usr/bin/env bash
# Compile existing models only: no PyTorch export or reference regeneration.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$#" -ne 0 ]; then
    printf 'Usage: bash scripts/prepare_coreai_bass_preview_aot.sh\n' >&2
    exit 2
fi
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run AOT compilation on your Apple Silicon Mac.\n' >&2
    exit 2
fi
if ! xcrun --find coreai-build >/dev/null 2>&1; then
    printf 'Install Metal Toolchain first: xcodebuild -downloadComponent MetalToolchain\n' >&2
    exit 2
fi
if [ ! -f prototypes/ios27-coreai/ModelCandidates/bass-preview-512/reference.json ]; then
    printf 'Prepare the source preview first: bash scripts/prepare_coreai_bass_preview.sh\n' >&2
    exit 2
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
"$repo_root/.runtime-cache/coreai-decoder-venv/bin/python" -u scripts/prepare_coreai_bass_preview_aot.py
