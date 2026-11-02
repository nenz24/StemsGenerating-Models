#!/usr/bin/env bash
set -euo pipefail
recording_workdir="$PWD"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [ "$#" -gt 1 ]; then printf 'Usage: bash scripts/prepare_coreai_bass_preview_1024.sh [path/to/piano.mp3]\n' >&2; exit 2; fi
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf 'Run on an Apple Silicon Mac; Linux may run prepare_coreai_preview_window.py directly.\n' >&2; exit 2
fi
recording_path="${1:-}"
case "$recording_path" in ""|/*) ;; *) recording_path="$recording_workdir/$recording_path" ;; esac
if [ ! -e prototypes/ios27-coreai/ModelCandidates/source-context-1024 ]; then
    if [ -z "$recording_path" ]; then
        recording_path="$(osascript -e 'POSIX path of (choose file of type {"public.audio"} with prompt "Pilih rekaman piano minimal 10,24 detik untuk referensi host")')"
    fi
    if [ ! -f "$recording_path" ]; then printf 'Recording not found: %s\n' "$recording_path" >&2; exit 2; fi
    if ! command -v ffmpeg >/dev/null 2>&1; then printf 'Install ffmpeg first: brew install ffmpeg\n' >&2; exit 2; fi
fi
bash scripts/prepare_coreai_decoder.sh --environment-only
args=(--seconds 10.24)
if [ -n "$recording_path" ]; then args+=(--source "$recording_path"); fi
"$repo_root/.runtime-cache/coreai-decoder-venv/bin/python" -u scripts/prepare_coreai_preview_window.py "${args[@]}"
