#!/usr/bin/env bash
# Prepare local app resources without putting large model assets into Git.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run this Mac preparation script in a native Apple Silicon terminal.\n' >&2
    printf 'Linux development can use export_coreai_decoder.py with CPU dependencies.\n' >&2
    exit 2
fi
environment_dir="$repo_root/.runtime-cache/coreai-decoder-venv"
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
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/decoder-pretrained"
if [ -e "$bundle" ]; then
    "$environment_dir/bin/python" scripts/export_coreai_decoder.py --verify-only
else
    "$environment_dir/bin/python" -u scripts/export_coreai_decoder.py
fi
printf '\nReady: reopen CoreAIProbe, rebuild on iPhone, then tap Uji decoder pretrained.\n'
