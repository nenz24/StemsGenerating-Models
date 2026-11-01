#!/usr/bin/env bash
# Complete denoiser, one forward; leave all previous model bundles intact.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export UV_PYTHON_INSTALL_DIR="$repo_root/.runtime-cache/uv/python"
export UV_CACHE_DIR="$repo_root/.runtime-cache/uv/cache"
export UV_PYTHON_BIN_DIR="$repo_root/.runtime-cache/uv/bin"
bash scripts/prepare_coreai_decoder.sh --environment-only
environment_dir="$repo_root/.runtime-cache/coreai-decoder-venv"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$environment_dir/bin/python" -r requirements-coreai-generator.txt
else
    "$environment_dir/bin/python" -m pip install -r requirements-coreai-generator.txt
fi
"$environment_dir/bin/python" scripts/export_coreai_denoiser_sharded.py --retire-monolith
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/denoiser-sharded"
if [ -e "$bundle" ]; then
    "$environment_dir/bin/python" scripts/export_coreai_denoiser_sharded.py --out "$bundle" --verify-only
else
    "$environment_dir/bin/python" -u scripts/export_coreai_denoiser_sharded.py --out "$bundle"
fi
printf '\nReady: rebuild on iPhone, then tap Uji denoiser bertahap. Monolithic loading is disabled.\n'
