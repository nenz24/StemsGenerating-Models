#!/usr/bin/env python3
"""Verify existing sampler/window decoder and select one frozen30s full run.

No inference, model export, checkpoint download or baseline replacement.
"""
import argparse
import json
from pathlib import Path
import os
import tempfile

from export_coreai_decoder import ROOT,sha256
from coreai_drum_reference30_contract import validate_reference
from prepare_coreai_drum_reference30 import verify_bundle as verify_sampler
from prepare_coreai_drum_reference30_window_decoder import verify_bundle as verify_decoder
from activate_coreai_drum_reference30 import select_profile
from coreai_drum_reference30_generation_contract import PROFILE,CONFIG,SAMPLER_BUNDLE,DECODER_BUNDLE,SOURCE_SHA,configuration


def activate(candidates=None):
    candidates=Path(candidates) if candidates is not None else ROOT/'prototypes/ios27-coreai/ModelCandidates'
    sampler=candidates/SAMPLER_BUNDLE;decoder=candidates/DECODER_BUNDLE
    validate_reference((sampler/'reference.json').read_bytes())
    verify_sampler(sampler);verify_decoder(decoder)
    if sha256(sampler/'capture/source48.f32le')!=SOURCE_SHA:raise ValueError('Accepted prepared piano changed; preserved')
    payload=configuration(sha256(decoder/'reference.json'))
    config=candidates/CONFIG
    if config.is_symlink():raise ValueError('Generation configuration symlink rejected; preserved')
    encoded=(json.dumps(payload,indent=2)+'\n').encode()
    if config.exists() and config.read_bytes()!=encoded:raise FileExistsError('Different generation configuration preserved: '+str(config))
    if not config.exists():
        descriptor,temporary=tempfile.mkstemp(prefix='.drums30-generation-',dir=candidates)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(encoded)
            os.link(temporary,config)
        finally:Path(temporary).unlink(missing_ok=True)
    select_profile(candidates,PROFILE)
    print('Ready: frozen30s reference generation. Existing BF16 sampler/FP32 decoder, inputs and original bounds retained.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    activate()
