#!/usr/bin/env bash
# Prepare one real generator block; preserve all existing codec resources.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
profile=short
if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 2 ] || [ "$1" != --profile ] || { [ "$2" != short ] && [ "$2" != context ] && [ "$2" != storage16 ]; }; then
        printf 'Usage: bash scripts/prepare_coreai_dit_block.sh [--profile short|context|storage16]\n' >&2
        exit 2
    fi
    profile="$2"
fi
export UV_PYTHON_INSTALL_DIR="$repo_root/.runtime-cache/uv/python"
export UV_CACHE_DIR="$repo_root/.runtime-cache/uv/cache"
export UV_PYTHON_BIN_DIR="$repo_root/.runtime-cache/uv/bin"
bash scripts/prepare_coreai_decoder.sh --environment-only
environment_dir="$repo_root/.runtime-cache/coreai-decoder-venv"
if command -v uv >/dev/null 2>&1; then
    uv pip install \
        --python "$environment_dir/bin/python" -r requirements-coreai-generator.txt
else
    "$environment_dir/bin/python" -m pip install -r requirements-coreai-generator.txt
fi
bundle_name=dit-block-pretrained
if [ "$profile" = context ]; then bundle_name=dit-context-pretrained; fi
if [ "$profile" = storage16 ]; then bundle_name=dit-storage16-pretrained; fi
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/$bundle_name"
if [ -e "$bundle" ]; then
    "$environment_dir/bin/python" scripts/export_coreai_dit_block.py --profile "$profile" --out "$bundle" --verify-only
else
    "$environment_dir/bin/python" -u scripts/export_coreai_dit_block.py --profile "$profile" --out "$bundle"
fi
if [ "$profile" = storage16 ]; then
    printf '\nReady: rebuild on iPhone, then tap Uji bobot FP16. Computation stays FP32; previous baselines are retained.\n'
elif [ "$profile" = context ]; then
    printf '\nReady: rebuild on iPhone, then tap Uji DiT konteks panjang. The single-block baseline is retained.\n'
else
    printf '\nReady: rebuild on iPhone, then tap Uji blok generator. This is not full instrument generation.\n'
fi
