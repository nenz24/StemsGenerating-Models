#!/usr/bin/env python3
"""Prepare fixed pair-0 inputs and dual references for a retained-function probe.

No native export. Use the existing block-pair asset and ship ~1 MB activations.
"""
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
from export_coreai_denoiser import load_full_upstream
from export_coreai_denoiser_sharded import SHARED, make_partitions, stage_plan
from prepare_coreai_denoiser_loop import verify_fixture as verify_loop

OUTPUT_SHAPE = [1,13,2048]
ASSET = 'blocks-00-01.aimodel'
EXPECTED_FILES = {f'{name}.f32le':shape for name,shape in SHARED.items()}
EXPECTED_FILES.update({'hidden-reference.f32le':OUTPUT_SHAPE,'hidden-fp32-reference.f32le':OUTPUT_SHAPE})


def verify_fixture(folder):
    ref=json.loads((folder/'reference.json').read_text())
    if (ref.get('schema_version')!=1 or ref.get('profile')!='denoiser-pair-reuse'
            or ref.get('asset_file')!=ASSET or ref.get('layer_indices')!=[0,1]
            or ref.get('input_shapes')!=SHARED or ref.get('output_shape')!=OUTPUT_SHAPE
            or ref.get('output_name')!='hidden_out' or ref.get('function_name')!='main'
            or ref.get('run_count')!=4 or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'
            or ref.get('weight_storage_dtype')!='float16' or ref.get('compute_dtype')!='float32'
            or ref.get('checkpoint_digest')!=records()['model.safetensors']['digest']
            or ref.get('checkpoint_full_sha256_verified') is not True
            or ref.get('expected_file')!='hidden-reference.f32le'
            or ref.get('original_fp32_reference_file')!='hidden-fp32-reference.f32le'
            or ref.get('absolute_tolerance')!=1e-4 or ref.get('relative_tolerance')!=1e-3
            or ref.get('relative_rmse_tolerance')!=1e-3
            or ref.get('host_storage16_vs_original_fp32',{}).get('parity_passed') is not True):
        raise ValueError('Unexpected retained-pair fixture contract')
    actual={p.name for p in folder.iterdir() if p.is_file() and p.name!='reference.json'}
    if actual!=set(EXPECTED_FILES) or set(ref['artifact_sha256'])!=set(EXPECTED_FILES):
        raise ValueError('Incomplete retained-pair fixture')
    for name,shape in EXPECTED_FILES.items():
        path=(folder/name).resolve()
        if (not path.is_relative_to(folder.resolve()) or path.stat().st_size!=math.prod(shape)*4
                or sha256(path)!=ref['artifact_sha256'][name]):
            raise ValueError(f'Retained-pair integrity mismatch: {name}')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-pair-reuse')
    parser.add_argument('--loop-fixture',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-loop-4')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:verify_fixture(args.out);print('Retained-pair references VERIFIED');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    loop=verify_loop(args.loop_fixture)
    publisher=records();record=publisher['model.safetensors']
    checkpoint=checkpoint_path(record,args.checkpoint)
    sources=ROOT/'.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True,exist_ok=True)
    paths={n:verified_source(publisher[n],sources) for n in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg,kind,rotary=load_full_upstream(paths)
    make_mask=sys.modules[kind.__module__].create_4d_mask
    import torch
    import numpy as np
    from safetensors import safe_open
    torch.set_num_threads(2)
    with torch.device('meta'):root=kind(cfg).eval()
    with safe_open(str(checkpoint),framework='pt',device='cpu') as source:
        state={k[len('decoder.'):]:source.get_tensor(k) for k in source.keys() if k.startswith('decoder.')}
    root.load_state_dict(state,strict=True,assign=True);del state
    root.rotary_emb=rotary(cfg,device='cpu');root.to(dtype=torch.float32,device='cpu').requires_grad_(False)
    # Snapshot only the original two layers before changing parameter storage.
    original_pair=copy.deepcopy(make_partitions(root,make_mask)[1]).eval()
    convert_storage16(root);parts=make_partitions(root,make_mask)
    values={n:torch.from_numpy(np.fromfile(args.loop_fixture/f,dtype='<f4').copy()).reshape(loop['input_shapes'][n])
            for n,f in loop['input_files'].items()}
    values.update(timestep=torch.tensor([1.]),timestep_r=torch.tensor([1.]))
    print('Full checkpoint verified; deriving fixed pair inputs from storage16 front at t=1',flush=True)
    with torch.no_grad():
        front=parts[0](*(values[n] for n in stage_plan()[0]['input_shapes']))
        values.update((stage_plan()[0]['output_bindings'][n],v) for n,v in zip(stage_plan()[0]['output_shapes'],front))
        inputs=tuple(values[n].detach().contiguous() for n in SHARED)
        original=original_pair(*inputs).detach().clone()
        expected=parts[1](*inputs).detach().clone()
    if any(not torch.isfinite(v).all() for v in inputs):raise ValueError('Nonfinite pair input')
    comparison=precision_metrics(expected,original)
    print('Same-input storage16 pair vs original FP32 pair:',comparison,flush=True)
    if not comparison['parity_passed']:raise ValueError('Pair baseline parity failed')
    parent=ROOT/'.runtime-cache/coreai-denoiser-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='pair-reuse-',dir=parent))
    for name,value in zip(SHARED,inputs):value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}.f32le')
    expected.numpy().astype('<f4',copy=False).tofile(stage/'hidden-reference.f32le')
    original.numpy().astype('<f4',copy=False).tofile(stage/'hidden-fp32-reference.f32le')
    ref={'schema_version':1,'profile':'denoiser-pair-reuse','asset_file':ASSET,'function_name':'main','output_name':'hidden_out',
         'layer_indices':[0,1],'input_shapes':SHARED,'output_shape':OUTPUT_SHAPE,'run_count':4,
         'scalar_type':'float32','byte_order':'little','weight_storage_dtype':'float16','compute_dtype':'float32',
         'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,
         'expected_file':'hidden-reference.f32le','original_fp32_reference_file':'hidden-fp32-reference.f32le',
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'input_provenance':'Storage16 front output at first loop step t=1, timestep_r=1 using exact committed v2 loop inputs; identical fixed pair inputs for both references',
         'loop_manifest_sha256':sha256(args.loop_fixture/'reference.json'),
         'host_storage16_vs_original_fp32':comparison,
         'reference_versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','numpy']},
         'artifact_sha256':{p.name:sha256(p) for p in sorted(stage.iterdir())},
         'scope':'First vs three repeated calls to one retained pair/model/function, fixed synthetic activations; not full generation',
         'device_execution_verified':False}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_fixture(stage)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; preserved {stage}')
    shutil.move(str(stage),str(args.out));print('Retained-pair fixture ready:',args.out,flush=True)


if __name__=='__main__':main()
