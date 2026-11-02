#!/usr/bin/env python3
"""Prepare/verify one fixed preview window, then select it for the next build.

No prior bundle is replaced. A selection is written atomically only after all
three matching bundles have passed their full integrity/host gates.
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


def activate(candidates, window):
    from export_coreai_source_context import verify_bundle as source
    from export_coreai_bass_conditioning import verify_bundle as conditioning
    from export_coreai_bass_preview import verify_bundle as preview
    source(candidates/window.source_bundle,window)
    conditioning(candidates/window.condition_bundle,window)
    preview(candidates/window.preview_bundle,candidates/window.source_bundle,candidates/window.condition_bundle,window)
    payload={'schema_version':1,'resource_profile':window.resource_profile,'audio_seconds':window.seconds}
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
    args=parser.parse_args();window=PreviewWindow.for_seconds(args.seconds)
    candidates=ROOT/'prototypes/ios27-coreai/ModelCandidates'
    source=candidates/window.source_bundle;condition=candidates/window.condition_bundle;preview=candidates/window.preview_bundle
    if not args.activate_only:
        if not (source/'reference.json').is_file() and (args.source is None or not args.source.is_file()):
            parser.error('--source must name a piano recording when preparing a new source window')
        jobs=[('export_coreai_source_context.py',source),('export_coreai_bass_conditioning.py',condition),('export_coreai_bass_preview.py',preview)]
        for script,bundle in jobs:
            command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--seconds',str(window.seconds),'--out',str(bundle)]
            if bundle.exists():command+=['--verify-only']
            elif script=='export_coreai_source_context.py':command+=['--source',str(args.source.resolve())]
            elif script=='export_coreai_bass_conditioning.py' and window.weight_storage_dtype=='float16' and (candidates/'bass-conditioning/reference.json').is_file():
                command+=['--reuse-native-from',str(candidates/'bass-conditioning')]
            subprocess.run(command,check=True,cwd=ROOT)
    payload=activate(candidates,window)
    print(f"Ready: active {payload['audio_seconds']} s / {payload['resource_profile']}. Old bundles retained on Mac; next app build stages only this profile.")

if __name__=='__main__':main()
