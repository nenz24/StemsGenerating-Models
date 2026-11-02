#!/usr/bin/env python3
"""Prepare one actual long-profile pair: existing BF16 asset versus FP32 constants.

Same learned values, FP32 calculations, fixed synthetic inputs at actual CFG
batch/long-window shapes. No source recording, full loop or waveform generation.
"""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, load_upstream, convert_storage16, precision_metrics
from export_coreai_bass_preview import stage_plan, block_pair
from coreai_preview_window import LONG

PROFILE = 'dit-precision-1024'
PAIR = stage_plan(LONG)[1]
INPUTS = PAIR['input_shapes']
OUTPUT = [2,128,2048]
ASSETS = {'bfloat16':'blocks-bf16.aimodel','float32':'blocks-fp32.aimodel'}
ORDER = ['bfloat16','float32','bfloat16']
FILES = {**{f'{name}.f32le':shape for name,shape in INPUTS.items()},'hidden-reference.f32le':OUTPUT}
NATIVE_FILES = {'metadata.json','main.hash','main.mlirb'}


def exact_comparison(value):
    return (value.get('finite') is True and value.get('parity_passed') is True
            and all(type(value.get(k)) in (int,float) and value[k] == 0
                    for k in ['max_absolute_error','rmse','normalized_rmse','elementwise_fraction_over_bound']))


def verify_bundle(folder):
    folder=Path(folder)
    ref=json.loads((folder/'reference.json').read_text())
    expected={'schema_version':1,'profile':PROFILE,'function_name':'main','output_name':'hidden_out',
              'layer_indices':[0,1],'input_shapes':INPUTS,'output_shape':OUTPUT,'assets':ASSETS,
              'pass_order':ORDER,'runs_per_pass':4,'scalar_type':'float32','byte_order':'little',
              'compute_dtype':'float32','checkpoint_digest':records()['model.safetensors']['digest'],
              'checkpoint_full_sha256_verified':True,'learned_parameter_values_identical':True,
              'reference_file':'hidden-reference.f32le','absolute_tolerance':1e-4,
              'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
              'source_pair_manifest_profile':'bass-preview-1024','source_weight_storage_dtype':'bfloat16'}
    for key,value in expected.items():
        if type(ref.get(key)) is not type(value) or ref[key]!=value:raise ValueError(f'Precision probe contract mismatch: {key}')
    for key in ['host_bf16_vs_original_fp32','host_expanded_fp32_vs_bf16']:
        if not exact_comparison(ref.get(key,{})):raise ValueError(f'Precision probe must preserve exact host values: {key}')
    if not re.fullmatch('[0-9a-f]{64}',ref.get('source_pair_manifest_sha256','')):
        raise ValueError('Missing source pair manifest identity')
    required=set(FILES)|{f'{asset}/{name}' for asset in ASSETS.values() for name in NATIVE_FILES}
    hashes=ref.get('artifact_sha256',{})
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if set(hashes)!=required or actual!=required:raise ValueError('Incomplete precision probe payload')
    source_hashes=ref.get('source_pair_artifact_sha256',{})
    if set(source_hashes)!=NATIVE_FILES:raise ValueError('Incomplete original BF16 asset identity')
    for name in NATIVE_FILES:
        if source_hashes[name]!=hashes[f'{ASSETS["bfloat16"]}/{name}']:raise ValueError('Existing BF16 source asset identity changed')
    for name in required:
        path=folder/name
        if (not path.resolve().is_relative_to(folder.resolve()) or not re.fullmatch('[0-9a-f]{64}',hashes[name])
                or sha256(path)!=hashes[name]):raise ValueError(f'Precision probe integrity mismatch: {name}')
    for name,shape in FILES.items():
        if (folder/name).stat().st_size!=math.prod(shape)*4:raise ValueError(f'Precision probe binary size mismatch: {name}')
    return ref


def activate(candidates):
    verify_bundle(candidates/PROFILE)
    payload={'schema_version':1,'resource_profile':PROFILE,'audio_seconds':10.24}
    descriptor,temporary=tempfile.mkstemp(prefix='.active-preview-',dir=candidates)
    try:
        with os.fdopen(descriptor,'w') as stream:stream.write(json.dumps(payload,indent=2)+'\n')
        os.replace(temporary,candidates/'active-preview.json')
    finally:Path(temporary).unlink(missing_ok=True)
    print('Active build profile: precision probe only. Baseline bundles retained on Mac.',flush=True)


def prepare(source_folder,out,checkpoint=None):
    if out.exists():raise FileExistsError(f'Output already exists: {out}; preserved')
    source_ref=json.loads((source_folder/'reference.json').read_text())
    record=records()['model.safetensors']
    if (source_ref.get('profile')!='bass-preview-1024' or source_ref.get('checkpoint_digest')!=record['digest']
            or source_ref.get('weight_storage_dtype')!='bfloat16' or source_ref.get('all_checkpoints_verified') is not True
            or source_ref.get('stages')!=stage_plan(LONG)):raise ValueError('Existing long-profile first pair contract changed')
    source_asset=source_folder/PAIR['asset_file']
    source_hashes={name:source_ref['artifact_sha256'][f'{PAIR["asset_file"]}/{name}'] for name in NATIVE_FILES}
    if any(sha256(source_asset/name)!=digest for name,digest in source_hashes.items()):raise ValueError('Original BF16 asset integrity mismatch')
    checkpoint=checkpoint_path(record,checkpoint)
    publisher=records();sources=ROOT/'.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True,exist_ok=True)
    paths={name:verified_source(publisher[name],sources) for name in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg,layer_type,rotary_type,make_mask=load_upstream(paths['modeling_acestep_v15_base.py'],paths['config.json'],paths['configuration_acestep_v15.py'])
    import torch
    from safetensors import safe_open
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2)
    layers=[]
    with safe_open(str(checkpoint),framework='pt',device='cpu') as weights:
        for index in [0,1]:
            prefix=f'decoder.layers.{index}.'
            with torch.device('meta'):layer=layer_type(cfg,layer_idx=index,use_cross_attention=True).eval()
            state={key[len(prefix):]:weights.get_tensor(key) for key in weights.keys() if key.startswith(prefix)}
            layer.load_state_dict(state,strict=True,assign=True)
            layer.to(device='cpu',dtype=torch.float32).requires_grad_(False);layers.append(layer)
    original=block_pair(layers).eval()
    bf16=convert_storage16(copy.deepcopy(original),'bfloat16').eval()
    expanded=copy.deepcopy(bf16).float().eval()
    original_parameters=dict(original.named_parameters())
    bf16_parameters=dict(bf16.named_parameters());expanded_parameters=dict(expanded.named_parameters())
    if (original_parameters.keys()!=bf16_parameters.keys() or original_parameters.keys()!=expanded_parameters.keys()
            or any(not torch.equal(a,bf16_parameters[name].float()) or not torch.equal(a,expanded_parameters[name])
                   for name,a in original_parameters.items())):
        raise ValueError('Learned parameter values changed; no export or activation')
    generator=torch.Generator(device='cpu').manual_seed(47)
    values={name:torch.randn(INPUTS[name],generator=generator)*0.1 for name in ['hidden','condition','temb']}
    with torch.no_grad():cos,sin=rotary_type(cfg)(values['hidden'],torch.arange(128).unsqueeze(0))
    values.update(rope_cos=cos,rope_sin=sin,
                  self_mask=make_mask(128,torch.float32,torch.device('cpu'),is_causal=False),
                  local_mask=make_mask(128,torch.float32,torch.device('cpu'),sliding_window=128,is_sliding_window=True,is_causal=False),
                  cross_mask=torch.zeros(INPUTS['cross_mask']))
    inputs=tuple(values[name].contiguous() for name in INPUTS)
    with torch.no_grad():gold=original(*inputs);a=bf16(*inputs);b=expanded(*inputs)
    host_a=precision_metrics(a,gold);host_b=precision_metrics(b,a)
    print('Existing BF16 values versus original FP32:',host_a,flush=True)
    print('Expanded FP32 values versus BF16:',host_b,flush=True)
    if not exact_comparison(host_a) or not exact_comparison(host_b):raise ValueError('Exact host equivalence failed')
    parent=ROOT/'.runtime-cache/coreai-precision-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='pair-',dir=parent))
    shutil.copytree(source_asset,stage/ASSETS['bfloat16'])
    for name,value in zip(INPUTS,inputs):value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}.f32le')
    gold.numpy().astype('<f4',copy=False).tofile(stage/'hidden-reference.f32le')
    print('Exporting ONE pair with FP32 constants; baseline BF16 asset is copied unchanged.',flush=True)
    exported=torch.export.export(expanded,inputs).run_decompositions(get_decomp_table())
    program=TorchConverter().add_exported_program(exported,input_names=list(INPUTS),output_names=['hidden_out']).to_coreai()
    program.save_asset(stage/ASSETS['float32'])
    ref={'schema_version':1,'profile':PROFILE,'function_name':'main','output_name':'hidden_out','layer_indices':[0,1],
         'input_shapes':INPUTS,'output_shape':OUTPUT,'assets':ASSETS,'pass_order':ORDER,'runs_per_pass':4,
         'scalar_type':'float32','byte_order':'little','compute_dtype':'float32',
         'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,'learned_parameter_values_identical':True,
         'reference_file':'hidden-reference.f32le','absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'source_pair_manifest_profile':'bass-preview-1024','source_weight_storage_dtype':'bfloat16',
         'source_pair_manifest_sha256':sha256(source_folder/'reference.json'),'source_pair_artifact_sha256':source_hashes,
         'host_bf16_vs_original_fp32':host_a,'host_expanded_fp32_vs_bf16':host_b,
         'input_provenance':'Fixed synthetic seed 47; actual long CFG batch=2, 128 hidden tokens / 118 condition tokens; local radius 128, all tokens valid',
         'scope':'Component A/B/A storage precision experiment, no full denoiser, audio duration or waveform-generation claim',
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','coreai-torch','coreai-core']},
         'device_execution_verified':False,
         'artifact_sha256':{p.relative_to(stage).as_posix():sha256(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='.DS_Store'}}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(stage)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(f'Destination appeared; staging preserved: {stage}')
    shutil.move(str(stage),str(out))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates'/PROFILE)
    parser.add_argument('--source-bundle',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-preview-1024')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    if args.out.exists() or args.verify_only:verify_bundle(args.out);print('Existing precision probe VERIFIED; no re-export.',flush=True)
    else:prepare(args.source_bundle,args.out,args.checkpoint)
    if args.activate:
        if args.out.name!=PROFILE:raise ValueError('Activation requires the standard probe folder name')
        activate(args.out.parent)
    print('Ready: build Release, then run Uji biaya bobot BF16 vs FP32.',flush=True)


if __name__=='__main__':main()
