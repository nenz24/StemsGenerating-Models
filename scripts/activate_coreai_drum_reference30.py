#!/usr/bin/env python3
"""Verify/reuse the reviewed native export and select its component-only build.

No model export/generation/download; preserve source, baseline and old selector.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from export_coreai_decoder import ROOT
from coreai_drum_reference30_contract import PROFILE, DECODER_PROFILE, BUNDLE, validate_reference
from prepare_coreai_drum_reference30 import verify_bundle


def activate(source, candidates=None, backup_root=None, decoder_only=False):
    source = Path(source).resolve()
    candidates = Path(candidates) if candidates is not None else ROOT / 'prototypes/ios27-coreai/ModelCandidates'
    backup_root = Path(backup_root) if backup_root is not None else ROOT / '.runtime-cache/drums-reference30-selector-backups'
    # Validate the complete original bundle before copying or changing selection.
    validate_reference((source / 'reference.json').read_bytes())
    verify_bundle(source)
    candidates.mkdir(parents=True, exist_ok=True)
    target = candidates / BUNDLE
    if target.is_symlink():
        raise ValueError('Diagnostic bundle must not be a symlink; preserved')
    if target.exists():
        validate_reference((target / 'reference.json').read_bytes())
        verify_bundle(target)
        print('Existing reviewed native bundle verified; no export/copy repeated.', flush=True)
    else:
        if source.is_relative_to(target.resolve()):
            raise ValueError('Source cannot be nested in the new diagnostic bundle')
        # Keep interrupted copies in a separate cache directory, never active.
        staging_root = ROOT / '.runtime-cache/drums-reference30-build-staging'
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='build-', dir=staging_root)) / BUNDLE
        print('Copy verified native assets to an isolated build bundle; source retained.', flush=True)
        shutil.copytree(source, staging)
        verify_bundle(staging)
        if target.exists():
            raise FileExistsError('Diagnostic target appeared during copy; both results preserved')
        staging.rename(target)
    profile = DECODER_PROFILE if decoder_only else PROFILE
    select_profile(candidates, profile, backup_root)
    return target


def select_profile(candidates, profile, backup_root=None):
    """Back up only the old selector; callers first verify their full bundle."""
    candidates = Path(candidates)
    backup_root = Path(backup_root) if backup_root is not None else ROOT / '.runtime-cache/drums-reference30-selector-backups'
    selector = candidates / 'active-preview.json'
    if selector.is_symlink():
        raise ValueError('Selector must not be a symlink; preserved')
    payload = {'schema_version': 1, 'resource_profile': profile, 'audio_seconds': 30.}
    encoded = (json.dumps(payload, indent=2) + '\n').encode()
    previous = selector.read_bytes() if selector.exists() else None
    if previous == encoded:
        print('Diagnostic build already selected:', profile, flush=True)
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='selector-', dir=backup_root))
    if previous is not None:
        (backup / 'active-preview.json').write_bytes(previous)
    else:
        (backup / 'selector-was-absent.txt').write_text('No selector before native30 component selection.\n')
    descriptor, temporary = tempfile.mkstemp(prefix='.active-preview-', dir=candidates)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
        os.replace(temporary, selector)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print('Prior build selection backup:', backup, flush=True)
    print('Selected diagnostic:', profile, '; baseline assets/source/cache retained.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--decoder-only', action='store_true', help='Isolate unchanged full30s decoder in a fresh process; no DiT/APG')
    args = parser.parse_args()
    activate(args.source, decoder_only=args.decoder_only)


if __name__ == '__main__':
    main()
