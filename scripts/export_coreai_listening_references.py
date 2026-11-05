#!/usr/bin/env python3
"""Export already-prepared original-FP32/partition WAVs; no inference or re-export.

These belong to the prepared host PCM, not arbitrary imported iPhone recordings.
Raw references stay unchanged. Listening copies use one common attenuation only.
"""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
from coreai_preview_window import PreviewWindow
from coreai_listening_audio import read_planar,write_float_wav,headroom_gain
from export_coreai_decoder import ROOT,sha256


def export_references(candidates, instrument, out, diagnostic=False):
    candidates,out=Path(candidates),Path(out)
    if out.exists():raise FileExistsError(f'Existing listening export preserved: {out}')
    if out.resolve().is_relative_to(candidates.resolve()):
        raise ValueError('Listening copies must be outside ModelCandidates; preserve bundle integrity')
    window=PreviewWindow(256,instrument)
    from export_coreai_source_context import verify_bundle as source
    from export_coreai_bass_conditioning import verify_bundle as condition
    from export_coreai_bass_preview import verify_bundle as preview
    source_ref=source(candidates/window.source_bundle,window)
    condition(candidates/window.condition_bundle,window)
    ref=preview(candidates/window.preview_bundle,candidates/window.source_bundle,candidates/window.condition_bundle,window,diagnostic)
    folder=candidates/window.preview_bundle
    piano=read_planar(candidates/window.source_bundle/'audio.f32le',window.samples)
    variants={'original-fp32':read_planar(folder/'audio-fp32-reference.f32le',window.samples),
              'partition-host':read_planar(folder/'audio-reference.f32le',window.samples)}
    common=headroom_gain(max(abs(v) for values in variants.values() for v in values))
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.listening-',dir=out.parent))
    write_float_wav(stage/'piano-prepared.wav',piano,window.samples)
    for name,values in variants.items():
        write_float_wav(stage/f'{instrument}-{name}-raw.wav',values,window.samples)
        write_float_wav(stage/f'{instrument}-{name}-headroom.wav',values,window.samples,common)
    report={'schema_version':1,'instrument':instrument,'audio_seconds':window.seconds,
            'scope':'Already-prepared HOST input/reference only; not exact parity against a different iPhone imported PCM',
            'source_pcm_sha256':source_ref['artifact_sha256']['audio.f32le'],
            'preview_reference_sha256':sha256(folder/'reference.json'),
            'host_export_status':ref.get('host_export_status','strict_pass'),
            'numerical_gates_unchanged':True,'inference_calls':0,'native_exports':0,
            'headroom_common_linear_gain':common,'headroom_policy':'attenuate-only, common gain, target max sample peak -3 dBFS; no limiter/EQ',
            'reference_peaks':{name:max(abs(v) for v in values) for name,values in variants.items()},
            'source_reference_files_modified':False,
            'artifact_sha256':{p.name:sha256(p) for p in stage.iterdir()}}
    (stage/'listening-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if out.exists():raise FileExistsError(f'Destination appeared; staged export preserved at {stage}')
    shutil.move(str(stage),str(out))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument',choices=['bass','drums','guitar-strum','guitar-pick'],default='drums')
    parser.add_argument('--diagnostic-trajectory',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=export_references(ROOT/'prototypes/ios27-coreai/ModelCandidates',args.instrument,args.out,args.diagnostic_trajectory)
    print('Ready:',args.out,'; common listening gain',report['headroom_common_linear_gain'],'; no model invocation or reference mutation')


if __name__=='__main__':main()
