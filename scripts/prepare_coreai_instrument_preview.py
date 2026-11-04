#!/usr/bin/env python3
"""Prepare one pinned long instrument preview; preserve all existing bundles.

Reuses source encoding and exact-contract native assets. New conditioning,
original-FP32 trajectories and strict component checks are always computed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from coreai_preview_window import PreviewWindow, LONG
from export_coreai_decoder import ROOT
from prepare_coreai_preview_window import activate

INSTRUMENTS = ('drums','guitar-strum','guitar-pick')


def reuse_candidate(candidates, window, kind):
    # Prefer matching geometry; each exporter verifies the entire chosen bundle
    # before any native payload is copied. Corrupt bundles stop, not silently fall back.
    options=[LONG]+[PreviewWindow(256,i) for i in INSTRUMENTS if i!=window.instrument]
    options.sort(key=lambda w:w.condition_tokens!=window.condition_tokens)
    return next((candidates/getattr(w,kind) for w in options
                 if (candidates/getattr(w,kind)/'reference.json').is_file()),None)


def prepare(instrument, diagnostic=False, activate_only=False, candidates=None, runner=subprocess.run):
    if instrument not in INSTRUMENTS: raise ValueError('Unknown instrument')
    candidates=Path(candidates or ROOT/'prototypes/ios27-coreai/ModelCandidates')
    window=PreviewWindow(256,instrument)
    from export_coreai_source_context import verify_bundle as verify_source
    # This workflow deliberately requires the existing accepted source window.
    verify_source(candidates/window.source_bundle,LONG)
    if not activate_only:
        for script,kind in [('export_coreai_bass_conditioning.py','condition_bundle'),('export_coreai_bass_preview.py','preview_bundle')]:
            output=candidates/getattr(window,kind)
            command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--seconds','10.24','--instrument',instrument,'--out',str(output)]
            if output.exists(): command+=['--verify-only']
            else:
                reuse=reuse_candidate(candidates,window,kind)
                if reuse: command+=['--reuse-native-from',str(reuse)]
            if kind=='preview_bundle':
                command+=['--source-bundle',str(candidates/window.source_bundle),'--condition-bundle',str(candidates/window.condition_bundle)]
                if diagnostic: command+=['--diagnostic-trajectory']
            runner(command,check=True,cwd=ROOT)
    result=activate(candidates,window,diagnostic)
    print(f'Ready: {instrument}, 10.24 s, {window.text_tokens} text / {window.condition_tokens} condition tokens. '
          'One active instrument per build; source/checkpoint/bass and other bundles retained.',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument',choices=INSTRUMENTS,required=True)
    parser.add_argument('--diagnostic-trajectory',action='store_true')
    parser.add_argument('--activate-only',action='store_true')
    args=parser.parse_args()
    prepare(args.instrument,args.diagnostic_trajectory,args.activate_only)


if __name__=='__main__':main()
