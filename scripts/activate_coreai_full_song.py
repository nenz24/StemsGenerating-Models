#!/usr/bin/env python3
"""Stage verified full-song inputs with existing BF16 sampler/FP32 decoder; no export."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from export_coreai_decoder import ROOT,sha256
from coreai_full_song_contract import PROFILE,CONFIG,SAMPLER_BUNDLE,DECODER_BUNDLE,configuration,validate_inputs


def write_atomic(path,raw):
    fd,temporary=tempfile.mkstemp(prefix='.full-song-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
        os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def activate(source,candidates=None,backup_root=None):
    from prepare_coreai_full_song_inputs import verify_job
    from prepare_coreai_drum_reference30 import verify_bundle as verify_sampler
    from prepare_coreai_drum_reference30_window_decoder import verify_bundle as verify_decoder
    from coreai_drum_reference30_generation_contract import CONFIG as WINDOW_CONFIG,configuration as window_configuration
    from activate_coreai_drum_reference30 import select_profile
    source=Path(source).resolve()
    candidates=Path(candidates) if candidates is not None else ROOT/'prototypes/ios27-coreai/ModelCandidates'
    backup_root=Path(backup_root) if backup_root is not None else ROOT/'.runtime-cache/full-song-activation-backups'
    validate_inputs(verify_job(source))
    verify_sampler(candidates/SAMPLER_BUNDLE);verify_decoder(candidates/DECODER_BUNDLE)
    job_sha=sha256(source/'reference.json');decoder_sha=sha256(candidates/DECODER_BUNDLE/'reference.json')
    config=configuration(job_sha,decoder_sha)
    target=candidates/config['job_bundle']
    if target.is_symlink():raise ValueError('Linked full-song destination preserved')
    if target.exists():
        if sha256(target/'reference.json')!=job_sha:raise FileExistsError('Different existing full-song job preserved')
        verify_job(target)
    else:
        staging_root=ROOT/'.runtime-cache/full-song-build-staging';staging_root.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='job-',dir=staging_root))/config['job_bundle']
        staging.mkdir()
        ref=json.loads((source/'reference.json').read_text())
        for name in ['reference.json',*ref['artifact_sha256']]:
            destination=staging/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,destination)
        verify_job(staging)
        if sha256(staging/'reference.json')!=job_sha:raise ValueError('Input job changed during copy; staging preserved')
        if target.exists():raise FileExistsError('Destination appeared; both results preserved')
        staging.rename(target)
    payload=(json.dumps(window_configuration(decoder_sha),indent=2)+'\n').encode()
    window_config=candidates/WINDOW_CONFIG
    if window_config.is_symlink() or (window_config.exists() and window_config.read_bytes()!=payload):
        raise FileExistsError('Existing30s configuration differs; preserved')
    if not window_config.exists():
        with window_config.open('xb') as stream:stream.write(payload)
    path=candidates/CONFIG;selector=candidates/'active-preview.json'
    if path.is_symlink() or selector.is_symlink():raise ValueError('Linked selector/configuration rejected')
    previous=path.read_bytes() if path.exists() else None
    previous_selector=selector.read_bytes() if selector.exists() else None
    encoded=(json.dumps(config,indent=2)+'\n').encode()
    backup_root.mkdir(parents=True,exist_ok=True)
    backup=Path(tempfile.mkdtemp(prefix='select-',dir=backup_root))
    if previous is not None:(backup/CONFIG).write_bytes(previous)
    if previous_selector is not None:(backup/'active-preview.json').write_bytes(previous_selector)
    try:
        if previous!=encoded:write_atomic(path,encoded)
        # Selector duration remains30s: this is the fixed model-window geometry.
        # Exact song duration is taken only from the immutable job reference.
        select_profile(candidates,PROFILE,backup_root)
    except BaseException:
        if previous is None:path.unlink(missing_ok=True)
        else:write_atomic(path,previous)
        if previous_selector is None:selector.unlink(missing_ok=True)
        else:write_atomic(selector,previous_selector)
        raise
    print('Full-song diagnostic selected. Inputs/baselines/models retained. Backup:',backup)
    print('Native batch, final music, memory and thermal acceptance: NOT EVALUATED; host source context reused.')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--inputs',type=Path,default=ROOT/'.runtime-cache/full-song-inputs-v1')
    args=parser.parse_args()
    if not args.execute:print('PLAN ONLY: verify/copy existing global inputs and reuse unchanged models; no inference/export/selector change.');return
    activate(args.inputs)


if __name__=='__main__':main()
