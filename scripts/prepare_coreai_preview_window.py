#!/usr/bin/env python3
"""Prepare/verify one fixed preview window, then select it for the next build.

No prior bundle is replaced. A selection is written atomically only after all
three matching bundles pass integrity and the explicitly selected admission
mode. Failed whole-trajectory parity is never relabeled as passed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from coreai_preview_window import PreviewWindow
from export_coreai_decoder import ROOT


def activate(candidates, window, diagnostic_trajectory=False):
    if diagnostic_trajectory and window.frames!=256:raise ValueError('Diagnostic trajectory is restricted to 10.24 seconds')
    from export_coreai_source_context import verify_bundle as source
    from export_coreai_bass_conditioning import verify_bundle as conditioning
    from export_coreai_bass_preview import verify_bundle as preview
    source(candidates/window.source_bundle,window)
    conditioning(candidates/window.condition_bundle,window)
    verified=preview(candidates/window.preview_bundle,candidates/window.source_bundle,candidates/window.condition_bundle,window,diagnostic_trajectory)
    if diagnostic_trajectory and verified.get('host_trajectory_diagnostic_requested') is not True:
        raise ValueError('Existing bundle uses the normal gate; activate it without --diagnostic-trajectory. No files changed.')
    payload={'schema_version':1,'resource_profile':window.resource_profile,'audio_seconds':window.seconds}
    if diagnostic_trajectory:payload['diagnostic_trajectory']=True
    descriptor,name=tempfile.mkstemp(prefix='.active-preview-',dir=candidates)
    try:
        with os.fdopen(descriptor,'w') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
        os.replace(name,candidates/'active-preview.json')
    finally:
        Path(name).unlink(missing_ok=True)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,choices=[5.12,10.24],required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--activate-only',action='store_true')
    parser.add_argument('--diagnostic-trajectory',action='store_true')
    args=parser.parse_args();window=PreviewWindow.for_seconds(args.seconds)
    if args.diagnostic_trajectory and window.frames!=256:parser.error('Diagnostic trajectory is restricted to 10.24 seconds')
    candidates=ROOT/'prototypes/ios27-coreai/ModelCandidates'
    source=candidates/window.source_bundle;condition=candidates/window.condition_bundle;preview=candidates/window.preview_bundle
    if not args.activate_only:
        if not (source/'reference.json').is_file() and (args.source is None or not args.source.is_file()):
            parser.error('--source must name a piano recording when preparing a new source window')
        jobs=[('export_coreai_source_context.py',source),('export_coreai_bass_conditioning.py',condition),('export_coreai_bass_preview.py',preview)]
        for script,bundle in jobs:
            command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--seconds',str(window.seconds),'--out',str(bundle)]
            if script=='export_coreai_bass_preview.py' and args.diagnostic_trajectory:command+=['--diagnostic-trajectory']
            if bundle.exists():command+=['--verify-only']
            elif script=='export_coreai_source_context.py':command+=['--source',str(args.source.resolve())]
            elif script=='export_coreai_bass_conditioning.py' and window.weight_storage_dtype=='float16' and (candidates/'bass-conditioning/reference.json').is_file():
                command+=['--reuse-native-from',str(candidates/'bass-conditioning')]
            subprocess.run(command,check=True,cwd=ROOT)
    payload=activate(candidates,window,args.diagnostic_trajectory)
    label="Ready (explicit trajectory diagnostic)" if args.diagnostic_trajectory else "Ready"
    print(f"{label}: active {payload['audio_seconds']} s / {payload['resource_profile']}. Old bundles retained on Mac; next app build stages only this profile.")

if __name__=='__main__':main()
