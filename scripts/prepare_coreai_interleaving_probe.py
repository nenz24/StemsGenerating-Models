#!/usr/bin/env python3
"""Copy existing distinct BF16 pairs and compute fixed-input host references.

No native export, waveform generation or baseline replacement. Start with two
pairs; --expand-existing prepares and verifies a larger bundle before archiving
the current diagnostic bundle. Generation/checkpoint/cache payloads are retained.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

from coreai_interleaving_contract import PROFILE, SIZES, INPUTS, OUTPUT, NATIVE, assets, verify_bundle
from export_coreai_decoder import ROOT, checkpoint_path, sha256
from prepare_coreai_precision_probe import verify_bundle as verify_precision, exact_comparison
from export_coreai_dit_block import records, verified_source, load_upstream, convert_storage16, precision_metrics
from export_coreai_bass_preview import stage_plan, block_pair
from coreai_preview_window import LONG


def prepare(source, fixture, out, count, checkpoint=None):
    if out.exists(): raise FileExistsError(f'Existing bundle preserved: {out}')
    verify_precision(fixture)
    source_ref = json.loads((source/'reference.json').read_text())
    record = records()['model.safetensors']
    if (source_ref.get('profile') != 'bass-preview-1024'
            or source_ref.get('weight_storage_dtype') != 'bfloat16'
            or source_ref.get('checkpoint_digest') != record['digest']
            or source_ref.get('all_checkpoints_verified') is not True
            or source_ref.get('stages') != stage_plan(LONG)):
        raise ValueError('Invalid BF16 source contract')
    source_hashes = {a+'/'+n:source_ref['artifact_sha256'][a+'/'+n] for a in assets(count) for n in NATIVE}
    for name, digest in source_hashes.items():
        if not (source/name).resolve().is_relative_to(source.resolve()) or sha256(source/name) != digest:
            raise ValueError('Source integrity mismatch: '+name)
    checkpoint = checkpoint_path(record, checkpoint)
    publisher = records(); cache = ROOT/'.runtime-cache/coreai-generator-source-publisher'; cache.mkdir(parents=True, exist_ok=True)
    paths = {n:verified_source(publisher[n],cache) for n in
             ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg, layer_type, _, _ = load_upstream(paths['modeling_acestep_v15_base.py'],paths['config.json'],paths['configuration_acestep_v15.py'])
    import torch
    import numpy as np
    from safetensors import safe_open
    torch.set_num_threads(2)
    inputs = tuple(torch.from_numpy(np.fromfile(fixture/(n+'.f32le'),dtype='<f4').copy()).reshape(shape)
                   for n,shape in INPUTS.items())
    parent = ROOT/'.runtime-cache/coreai-interleaving-staging'; parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='pairs-',dir=parent))
    for n in INPUTS: shutil.copy2(fixture/(n+'.f32le'),stage/(n+'.f32le'))
    exact = []
    with safe_open(str(checkpoint),framework='pt',device='cpu') as weights:
        for pair, asset in enumerate(assets(count)):
            layers = []
            for index in (2*pair,2*pair+1):
                prefix = f'decoder.layers.{index}.'
                with torch.device('meta'): layer = layer_type(cfg,layer_idx=index,use_cross_attention=True).eval()
                layer.load_state_dict({k[len(prefix):]:weights.get_tensor(k) for k in weights.keys() if k.startswith(prefix)},strict=True,assign=True)
                layer.to(device='cpu',dtype=torch.float32).requires_grad_(False); layers.append(layer)
            original = block_pair(layers).eval()
            candidate = convert_storage16(copy.deepcopy(original),'bfloat16').eval()
            if any(not torch.equal(v,dict(candidate.named_parameters())[n].float()) for n,v in original.named_parameters()):
                raise ValueError('BF16 learned values changed')
            with torch.no_grad(): gold = original(*inputs); actual = candidate(*inputs)
            metric = precision_metrics(actual,gold)
            if not exact_comparison(metric): raise ValueError(f'Host exact equivalence failed for {asset}: {metric}')
            if not torch.isfinite(gold).all(): raise ValueError('Nonfinite host reference')
            gold.numpy().astype('<f4',copy=False).tofile(stage/f'reference-{pair:02d}.f32le')
            shutil.copytree(source/asset,stage/asset,ignore=shutil.ignore_patterns('.DS_Store'))
            exact.append(True)
            print(f'Pair {pair+1}/{count}: original FP32 reference and exact BF16 host check complete',flush=True)
            del original,candidate,layers,gold,actual,layer
    ref = {'schema_version':1,'profile':PROFILE,'pair_count':count,'assets':assets(count),
           'input_shapes':INPUTS,'output_shape':OUTPUT,'calls_per_asset':20,
           'scalar_type':'float32','byte_order':'little','weight_storage_dtype':'bfloat16','compute_dtype':'float32',
           'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
           'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,'host_exact':exact,
           'source_manifest_sha256':sha256(source/'reference.json'),'source_asset_sha256':source_hashes,
           'scope':'Fixed input BF16 component diagnostic only; no audio or sampler activation',
           'artifact_sha256':{p.relative_to(stage).as_posix():sha256(p) for p in stage.rglob('*') if p.is_file()}}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n')
    verify_bundle(stage)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(f'Destination appeared; staged bundle retained at {stage}')
    shutil.move(str(stage),str(out))


def activate(candidates):
    verify_bundle(candidates/PROFILE)
    descriptor, temporary = tempfile.mkstemp(prefix='.active-preview-',dir=candidates)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump({'schema_version':1,'resource_profile':PROFILE,'audio_seconds':10.24},stream)
        os.replace(temporary,candidates/'active-preview.json')
    finally: Path(temporary).unlink(missing_ok=True)


def install_expansion(out, prepared, archive_root, expected_manifest_sha256):
    """Archive the old diagnostic; restore it if installing the new one fails."""
    out, prepared, archive_root = Path(out), Path(prepared), Path(archive_root)
    if out.is_symlink() or prepared.is_symlink(): raise ValueError('Expansion requires ordinary bundle directories')
    if out.resolve().is_relative_to(prepared.resolve()) or prepared.resolve().is_relative_to(out.resolve()):
        raise ValueError('Expansion needs separate non-nested bundles')
    if archive_root.resolve().is_relative_to(out.resolve()) or archive_root.resolve().is_relative_to(prepared.resolve()):
        raise ValueError('Archive must be outside both bundles')
    old, new = verify_bundle(out), verify_bundle(prepared)
    if sha256(out/'reference.json') != expected_manifest_sha256:
        raise ValueError('Existing manifest changed during preparation; both bundles preserved')
    if new['pair_count'] <= old['pair_count']: raise ValueError('Expansion must increase pair count')
    if new['source_manifest_sha256'] != old['source_manifest_sha256']:
        raise ValueError('Source manifest changed; both bundles preserved')
    if any(new['artifact_sha256'].get(n) != h for n,h in old['artifact_sha256'].items()):
        raise ValueError('Existing assets/inputs/references changed; both bundles preserved')
    # Archive outside ModelCandidates: it must never be staged into an app.
    archive_root.mkdir(parents=True,exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix=f'k{old["pair_count"]}-to-k{new["pair_count"]}-',dir=archive_root))/PROFILE
    os.rename(out,archive)
    try:
        if out.exists(): raise FileExistsError(f'Destination appeared during expansion: {out}')
        os.rename(prepared,out)
    except Exception:
        # Do not overwrite a concurrently-created destination during rollback.
        if out.exists():
            raise RuntimeError(f'Expansion stopped: destination appeared; old bundle retained at {archive}, new at {prepared}')
        os.rename(archive,out)
        raise
    print(f'Previous K={old["pair_count"]} diagnostic archived unchanged: {archive}',flush=True)
    return archive


def expand_bundle(source, fixture, out, count, checkpoint=None):
    old = verify_bundle(out)
    if count not in SIZES or count <= old['pair_count']:
        raise ValueError('Expansion requires a larger supported pair count')
    expected = sha256(out/'reference.json')
    parent = ROOT/'.runtime-cache/coreai-interleaving-staging'; parent.mkdir(parents=True,exist_ok=True)
    prepared = Path(tempfile.mkdtemp(prefix='expansion-',dir=parent))/PROFILE
    print(f'Preparing K={count} in {prepared}; current K={old["pair_count"]} and selector remain untouched until verified.',flush=True)
    # Existing bundle/selector are untouched until preparation and validation finish.
    prepare(source,fixture,prepared,count,checkpoint)
    return install_expansion(out,prepared,ROOT/'.runtime-cache/coreai-interleaving-archives',expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'prototypes/ios27-coreai/ModelCandidates'
    parser.add_argument('--out',type=Path,default=base/PROFILE)
    parser.add_argument('--source-bundle',type=Path,default=base/'bass-preview-1024')
    parser.add_argument('--fixture-bundle',type=Path,default=base/'dit-precision-1024')
    parser.add_argument('--pairs',type=int,choices=SIZES,default=2)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--expand-existing',action='store_true',help='Prepare larger bundle first, archive existing diagnostic, then install; no deletion')
    parser.add_argument('--activate',action='store_true')
    args = parser.parse_args()
    if args.verify_only and args.expand_existing: parser.error('--verify-only cannot expand a bundle')
    if args.activate and args.out.name != PROFILE: parser.error('Activation requires standard folder name')
    if args.out.exists() or args.verify_only:
        ref = verify_bundle(args.out)
        if ref['pair_count'] != args.pairs:
            if not args.expand_existing: raise ValueError('Existing pair count differs; bundle preserved. Use --expand-existing to archive and expand.')
            expand_bundle(args.source_bundle,args.fixture_bundle,args.out,args.pairs,args.checkpoint)
    else: prepare(args.source_bundle,args.fixture_bundle,args.out,args.pairs,args.checkpoint)
    if args.activate:
        if args.out.name != PROFILE: raise ValueError('Activation requires standard folder name')
        activate(args.out.parent)
    print('Ready: BF16 interleaving diagnostic; generation bundles retained. Build Release and run on iPhone.')


if __name__ == '__main__': main()
