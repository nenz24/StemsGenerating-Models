#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run on an Apple Silicon Mac; Linux may invoke the Python driver directly.\n' >&2; exit 2
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
"$repo_root/.runtime-cache/coreai-decoder-venv/bin/python" -u scripts/prepare_coreai_instrument_preview.py "$@"
