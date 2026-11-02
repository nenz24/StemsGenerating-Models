#!/usr/bin/env bash
# Prepare the local real-recording audio-context gate, retaining existing assets.
set -euo pipefail
recording_workdir="$PWD"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$#" -gt 1 ]; then
    printf 'Usage: bash scripts/prepare_coreai_source_context.sh [path/to/piano.mp3]\n' >&2
    exit 2
fi
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run this preparation script on an Apple Silicon Mac. Linux can run export_coreai_source_context.py.\n' >&2
    exit 2
fi
recording_path="${1:-}"
case "$recording_path" in
    ""|/*) ;;
    *) recording_path="$recording_workdir/$recording_path" ;;
esac
bundle="$repo_root/prototypes/ios27-coreai/ModelCandidates/source-context"
if [ ! -e "$bundle" ]; then
    if [ -z "$recording_path" ]; then
        recording_path="$(osascript -e 'POSIX path of (choose file with prompt "Pilih rekaman piano untuk uji audio-context 5,12 detik")')"
    fi
    if [ ! -f "$recording_path" ]; then printf 'Recording not found: %s\n' "$recording_path" >&2; exit 2; fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        printf 'ffmpeg is required for canonical PCM. Install on Mac with: brew install ffmpeg\n' >&2
        exit 2
    fi
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
environment_python="$repo_root/.runtime-cache/coreai-decoder-venv/bin/python"
if [ -e "$bundle" ]; then
    if [ -n "$recording_path" ]; then
        "$environment_python" scripts/export_coreai_source_context.py --out "$bundle" --verify-only --source "$recording_path"
    else
        "$environment_python" scripts/export_coreai_source_context.py --out "$bundle" --verify-only
    fi
else
    "$environment_python" -u scripts/export_coreai_source_context.py --source "$recording_path" --out "$bundle"
fi
printf '\nReady: rebuild on iPhone, then tap Uji audio-context piano. Existing DiT/codec gates are retained.\n'
