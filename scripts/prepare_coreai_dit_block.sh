#!/usr/bin/env bash
# Prepare one real generator block; preserve all existing codec resources.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
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
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/dit-block-pretrained"
if [ -e "$bundle" ]; then
    "$environment_dir/bin/python" scripts/export_coreai_dit_block.py --out "$bundle" --verify-only
else
    "$environment_dir/bin/python" -u scripts/export_coreai_dit_block.py --out "$bundle"
fi
printf '\nReady: rebuild on iPhone, then tap Uji blok generator. This is not full instrument generation.\n'
