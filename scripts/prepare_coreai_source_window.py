#!/usr/bin/env python3
"""Prepare source-only native test by reusing the existing FP32 encoder256 asset.

No native export/download. Strict original CPU gates must pass before copying
or activating the new13-file diagnostic bundle; original models are retained.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from export_coreai_decoder import ROOT,sha256
from coreai_source_window_contract import (PROFILE,BUNDLE,VAE_SHA,INPUT_SHAPES,OUTPUT_SHAPES,
    DATA_SHAPES,FILES,NATIVE_FILES,tile_plan,dependency_bounds,validate_reference)

DEFAULT_OUT=ROOT/'prototypes/ios27-coreai/ModelCandidates'/BUNDLE


def verify_reusable(folder):
    folder=Path(folder)
    raw=folder/'reference.json'
    if raw.is_symlink() or not raw.is_file():raise ValueError('Existing source256 reference required')
    ref=json.loads(raw.read_text())
    expected={'profile':'source-context-1024','asset_file':'source-context.aimodel','function_name':'main',
              'input_shapes':INPUT_SHAPES,'output_shapes':OUTPUT_SHAPES,'scalar_type':'float32','byte_order':'little',
              'checkpoint_digest':VAE_SHA,'checkpoint_full_sha256_verified':True,
              'host_distribution_sample_parity_exact':True,'host_weight_norm_folding_parity_exact':True}
    if any(ref.get(k)!=v for k,v in expected.items()):raise ValueError('Existing source encoder does not match FP32/256 provenance')
    for name in NATIVE_FILES:
        key='source-context.aimodel/'+name;path=folder/key
        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or sha256(path)!=ref['artifact_sha256'][key]:
            raise ValueError('Existing source encoder payload differs; preserved')
    return ref


def verify_bundle(folder):
    folder=Path(folder);ref=validate_reference(json.loads((folder/'reference.json').read_text()))
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if actual!=FILES:raise ValueError('Incomplete/unexpected source window bundle')
    for name,digest in ref['artifact_sha256'].items():
        p=folder/name
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha256(p)!=digest:
            raise ValueError('Source window checksum mismatch: '+name)
    for name,shape in DATA_SHAPES.items():
        if (folder/name).stat().st_size!=math.prod(shape)*4:raise ValueError('Source fixture size mismatch')
    original=json.loads((folder/'original-source-reference.json').read_text())
    for name in NATIVE_FILES:
        key='source-context.aimodel/'+name
        if ref['artifact_sha256'][key]!=original['artifact_sha256'][key]:raise ValueError('Reused source payload changed')
    return ref


def prepare(capture,out,native_source,checkpoint=None,activate=False):
    from audit_coreai_source_window import execute
    from activate_coreai_drum_reference30 import select_profile
    capture,out,native_source=Path(capture),Path(out),Path(native_source)
    if activate and out.resolve()!=DEFAULT_OUT.resolve():raise ValueError('Activation requires the default new source bundle')
    if any(out.resolve().is_relative_to(p) for p in (capture.resolve(),native_source.resolve(),ROOT/'baselines')):
        raise ValueError('Source output must be outside original capture/source/baselines')
    if out.is_symlink():raise ValueError('Linked output preserved')
    if out.exists():
        verify_bundle(out)
        if activate:select_profile(out.parent,PROFILE)
        print('Existing complete source window bundle verified; no inference/export repeated.');return
    verify_reusable(native_source) # reject wrong/missing old asset before CPU work
    cache=ROOT/'.runtime-cache/source-window-preparation';cache.mkdir(parents=True,exist_ok=True)
    audit_root=Path(tempfile.mkdtemp(prefix='host-',dir=cache))/'audit'
    execute(SimpleNamespace(capture=capture,out=audit_root,vae_checkpoint=checkpoint,window_frames=256,convolution_backend='aten'))
    audit=json.loads((audit_root/'report.json').read_text())
    if audit.get('strict_verdict')!='PASS' or audit.get('comparisons_passed')!=32:
        raise ValueError('Original32 source gates failed; preserved '+str(audit_root)+'; no activation/copy')
    staging=Path(tempfile.mkdtemp(prefix='bundle-',dir=cache))/BUNDLE;staging.mkdir()
    for name in DATA_SHAPES:shutil.copyfile(capture/name,staging/name)
    shutil.copyfile(capture/'report.json',staging/'capture-report.json')
    shutil.copyfile(native_source/'reference.json',staging/'original-source-reference.json')
    (staging/'source-context.aimodel').mkdir()
    for name in NATIVE_FILES:shutil.copyfile(native_source/'source-context.aimodel'/name,staging/'source-context.aimodel'/name)
    ref={k:audit[k] for k in ['vae_checkpoint_sha256','capture_report_sha256','window_frames','core_frames','halo_frames',
        'weight_storage_dtype','compute_dtype','epsilon_regenerated','absolute_tolerance','relative_tolerance','relative_rmse_tolerance',
        'window_plan','dependency_sample_bounds','tile_input_sha256','reference_convolution_backend']}
    ref.update(schema_version=1,profile=PROFILE,status='prepared_source_window_encoder',asset_file='source-context.aimodel',
        function_name='main',latent_frames=750,input_shapes=INPUT_SHAPES,output_shapes=OUTPUT_SHAPES,
        scalar_type='float32',byte_order='little',host_checks=audit['checks'],host_numerical_gate_passed=True,
        coverage_exactly_once=True,architecture_guard_passed=True,native_exports=0,baseline_modified=False,
        native_parity='NOT EVALUATED',native_memory_gate='NOT EVALUATED',full_song_native_readiness='NOT EVALUATED',
        original_source_reference_sha256=sha256(native_source/'reference.json'),
        artifact_sha256={name:sha256(staging/name) for name in FILES})
    (staging/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(staging)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Destination appeared; verified staging retained: '+str(staging))
    staging.rename(out)
    if activate:select_profile(out.parent,PROFILE)
    print('Prepared one unchanged source encoder, exact capture and original gates:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--activate',action='store_true')
    parser.add_argument('--capture',type=Path,default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--native-source',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/source-context-1024')
    parser.add_argument('--vae-checkpoint',type=Path);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if not args.execute:
        if args.activate:parser.error('--activate requires --execute or verification of an existing bundle via --execute')
        print('PLAN ONLY: source encoder256 reuse; four calls,32 original gates,13 app files. No native export or baseline change.');return
    prepare(args.capture,args.out,args.native_source,args.vae_checkpoint,args.activate)


if __name__=='__main__':main()
