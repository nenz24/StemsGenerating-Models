#!/usr/bin/env bash
# Prepare local app resources without putting large model assets into Git.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
component=decoder
frames=""
environment_only=false
usage() {
    printf 'Usage: bash scripts/prepare_coreai_decoder.sh [--component decoder|encoder] [--frames 5|25] [--environment-only]\n' >&2
    exit 2
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --component) [ "$#" -ge 2 ] || usage; component="$2"; shift 2 ;;
        --frames) [ "$#" -ge 2 ] || usage; frames="$2"; shift 2 ;;
        --environment-only) environment_only=true; shift ;;
        *) usage ;;
    esac
done
case "$component" in decoder|encoder) ;; *) usage ;; esac
if [ -z "$frames" ]; then
    if [ "$component" = encoder ]; then frames=25; else frames=5; fi
fi
case "$frames" in 5|25) ;; *) usage ;; esac
if [ "$component" = encoder ] && [ "$frames" != 25 ]; then usage; fi
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run this Mac preparation script in a native Apple Silicon terminal.\n' >&2
    printf 'Linux development can use export_coreai_decoder.py with CPU dependencies.\n' >&2
    exit 2
fi
environment_dir="$repo_root/.runtime-cache/coreai-decoder-venv"
# uv's managed Python and cache defaults may point at non-writable ~/.local or
# ~/.cache directories. Scope all uv writes to this ignored checkout-local tree.
export UV_PYTHON_INSTALL_DIR="$repo_root/.runtime-cache/uv/python"
export UV_CACHE_DIR="$repo_root/.runtime-cache/uv/cache"
export UV_PYTHON_BIN_DIR="$repo_root/.runtime-cache/uv/bin"
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_BIN_DIR"
if command -v uv >/dev/null 2>&1; then
    if [ ! -x "$environment_dir/bin/python" ]; then
        uv venv --python 3.12 "$environment_dir"
    fi
    uv pip install --python "$environment_dir/bin/python" -r requirements-coreai-audit.txt
else
    if [ ! -x "$environment_dir/bin/python" ]; then
        selected_python=""
        for candidate in "$repo_root/.venv/bin/python" python3.12 python3.13 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && \
                "$candidate" -c 'import sys; raise SystemExit(not ((3,11) <= sys.version_info[:2] < (3,14)))' 2>/dev/null; then
                selected_python="$candidate"
                break
            fi
        done
        if [ -z "$selected_python" ]; then
            printf 'Python 3.11–3.13 is required. Install Python 3.12 or uv, then rerun.\n' >&2
            exit 2
        fi
        "$selected_python" -m venv "$environment_dir"
    fi
    "$environment_dir/bin/python" -m pip install -r requirements-coreai-audit.txt
fi
if [ "$environment_only" = true ]; then
    printf 'CoreAI Python environment ready. Existing model bundles are retained.\n'
    exit 0
fi
bundle_name=decoder-pretrained
if [ "$frames" = 25 ]; then bundle_name=decoder-pretrained-1s; fi
if [ "$component" = encoder ]; then bundle_name=encoder-pretrained-1s; fi
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/$bundle_name"
if [ -e "$bundle" ]; then
    "$environment_dir/bin/python" scripts/export_coreai_decoder.py --component "$component" --out "$bundle" --frames "$frames" --verify-only
else
    "$environment_dir/bin/python" -u scripts/export_coreai_decoder.py --component "$component" --out "$bundle" --frames "$frames"
fi
if [ "$component" = encoder ]; then
    printf '\nReady: rebuild on iPhone, then tap Uji encoder 1 detik. Decoder bundles are retained.\n'
elif [ "$frames" = 25 ]; then
    printf '\nReady: rebuild on iPhone, then tap Uji decoder 1 detik. The 0.2 s bundle is retained.\n'
else
    printf '\nReady: rebuild on iPhone, then tap Uji decoder pretrained.\n'
fi
