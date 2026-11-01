#!/usr/bin/env python3
"""Export all 24 denoiser layers as front + 12 pairs + tail, never one huge asset."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
from export_coreai_denoiser import INPUT_SHAPES, OUTPUT_SHAPE, load_full_upstream

SHARED = {'hidden':[1,13,2048],'condition':[1,64,2048],'temb':[1,6,2048],
          'rope_cos':[1,13,128],'rope_sin':[1,13,128],'self_mask':[1,1,13,13],
          'local_mask':[1,1,13,13],'cross_mask':[1,1,13,64]}
FRONT_OUTPUTS = {('condition_projected' if n=='condition' else n):s for n,s in SHARED.items()}
FRONT_OUTPUTS['final_temb']=[1,2048]


def stage_plan():
    plan=[{'asset_file':'front.aimodel','layer_indices':[], 'input_shapes':INPUT_SHAPES,'output_shapes':FRONT_OUTPUTS,
           'output_bindings':{n:('condition' if n=='condition_projected' else n) for n in FRONT_OUTPUTS}}]
    for index in range(0,24,2):
        plan.append({'asset_file':f'blocks-{index:02d}-{index+1:02d}.aimodel','layer_indices':[index,index+1],
                     'input_shapes':SHARED,'output_shapes':{'hidden_out':[1,13,2048]},'output_bindings':{'hidden_out':'hidden'}})
    plan.append({'asset_file':'tail.aimodel','layer_indices':[],
                 'input_shapes':{'hidden':[1,13,2048],'final_temb':[1,2048]},'output_shapes':{'velocity':OUTPUT_SHAPE},
                 'output_bindings':{'velocity':'velocity'}})
    return plan


def retire_monolith():
    source=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-pretrained'
    if not source.exists():return None
    parent=ROOT/'.runtime-cache/coreai-retired-bundles';parent.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='denoiser-monolithic-',dir=parent))/source.name
    shutil.move(str(source),str(destination))
    print('Monolithic resource preserved outside app bundle:',destination,flush=True)
    return destination


def verify_bundle(folder,record):
    ref=json.loads((folder/'reference.json').read_text())
    if (ref.get('profile')!='denoiser-sharded' or ref.get('schema_version')!=1 or ref.get('asset_file')!='sharded-pipeline'
            or ref.get('input_shapes')!=INPUT_SHAPES or ref.get('output_shape')!=OUTPUT_SHAPE
            or ref.get('function_name')!='main' or ref.get('output_name')!='velocity'
            or ref.get('layer_count')!=24 or ref.get('stages')!=stage_plan()
            or ref.get('parameter_count')!=1575458880 or ref.get('parameter_coverage_exactly_once') is not True
            or ref.get('checkpoint',{}).get('digest')!=record['digest'] or ref.get('checkpoint_full_sha256_verified') is not True
            or ref.get('partition_vs_monolithic',{}).get('parity_passed') is not True
            or ref.get('storage16_vs_original_fp32',{}).get('parity_passed') is not True
            or ref.get('weight_storage_dtype')!='float16' or ref.get('compute_dtype')!='float32'
            or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'
            or ref.get('original_fp32_reference_file')!='velocity-fp32-reference.f32le'):
        raise ValueError('Unexpected partitioned denoiser contract/order/coverage')
    required={f'{n}.f32le' for n in INPUT_SHAPES}|{'velocity-reference.f32le','velocity-fp32-reference.f32le'}
    for stage in stage_plan():
        required.update(f'{stage["asset_file"]}/{name}' for name in ['main.hash','main.mlirb','metadata.json'])
    hashes=ref['artifact_sha256']
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if not required.issubset(hashes) or actual!=set(hashes):raise ValueError('Incomplete sharded bundle')
    for name,digest in hashes.items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=digest:raise ValueError(f'Integrity mismatch: {name}')
    for name,shape in [(f'{n}.f32le',s) for n,s in INPUT_SHAPES.items()]+[(n,OUTPUT_SHAPE) for n in ['velocity-reference.f32le','velocity-fp32-reference.f32le']]:
        if (folder/name).stat().st_size!=math.prod(shape)*4:raise ValueError('Reference size mismatch')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-sharded')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--retire-monolith',action='store_true',help='Move the failed resource to ignored cache; preserve every file')
    args=parser.parse_args()
    if args.retire_monolith:retire_monolith();return
    publisher=records();record=publisher['model.safetensors']
    if args.verify_only:verify_bundle(args.out,record);print('Sharded denoiser: VERIFIED');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    checkpoint=checkpoint_path(record,args.checkpoint)
    sources=ROOT/'.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True,exist_ok=True)
    paths={n:verified_source(publisher[n],sources) for n in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg,kind,rotary_type=load_full_upstream(paths)
    make_mask=sys.modules[kind.__module__].create_4d_mask
    import torch
    import torch.nn.functional as F
    import numpy as np
    from safetensors import safe_open
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2)
    with torch.device('meta'):root=kind(cfg).eval()
    with safe_open(str(checkpoint),framework='pt',device='cpu') as source:
        state={k[len('decoder.'):]:source.get_tensor(k) for k in source.keys() if k.startswith('decoder.')}
    root.load_state_dict(state,strict=True,assign=True);del state
    root.rotary_emb=rotary_type(cfg,device='cpu');root.to(dtype=torch.float32,device='cpu').requires_grad_(False)
    generator=torch.Generator(device='cpu').manual_seed(0)
    values={'noise':torch.randn(INPUT_SHAPES['noise'],generator=generator),
            'condition':torch.randn(INPUT_SHAPES['condition'],generator=generator)*0.1,
            'context':torch.randn(INPUT_SHAPES['context'],generator=generator)*0.1,
            'timestep':torch.tensor([0.5]),'timestep_r':torch.tensor([0.25])}
    def full():return root(hidden_states=values['noise'],encoder_hidden_states=values['condition'],context_latents=values['context'],
                           timestep=values['timestep'],timestep_r=values['timestep_r'],attention_mask=None,encoder_attention_mask=None,use_cache=False)[0]
    with torch.no_grad():original=full().detach().clone()
    convert_storage16(root)
    with torch.no_grad():monolithic=full().detach().clone()

    class Front(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj=root.proj_in;self.condition=root.condition_embedder
            self.time=root.time_embed;self.time_r=root.time_embed_r;self.rotary=root.rotary_emb
        def forward(self,noise,condition,context,timestep,timestep_r):
            a,p=self.time(timestep);b,q=self.time_r(timestep-timestep_r)
            hidden=self.proj(F.pad(torch.cat([context,noise],dim=-1),(0,0,0,1)))
            cond=self.condition(condition);cos,sin=self.rotary(hidden,torch.arange(13).unsqueeze(0))
            mask=make_mask(13,torch.float32,hidden.device,is_causal=False)
            local=make_mask(13,torch.float32,hidden.device,sliding_window=128,is_sliding_window=True,is_causal=False)
            cross=make_mask(64,torch.float32,hidden.device,is_causal=False)[:,:,:13,:64]
            return tuple(x.contiguous() for x in [hidden,cond,p+q,cos,sin,mask,local,cross,a+b])

    class Pair(torch.nn.Module):
        def __init__(self,index):super().__init__();self.layers=torch.nn.ModuleList(root.layers[index:index+2])
        def forward(self,hidden,condition,temb,rope_cos,rope_sin,self_mask,local_mask,cross_mask):
            for layer in self.layers:
                hidden=layer(hidden_states=hidden,position_embeddings=(rope_cos,rope_sin),temb=temb,
                             attention_mask=local_mask if layer.attention_type=='sliding_attention' else self_mask,
                             encoder_hidden_states=condition,encoder_attention_mask=cross_mask,use_cache=False,past_key_value=None)[0]
            return hidden.contiguous()

    class Tail(torch.nn.Module):
        def __init__(self):super().__init__();self.table=root.scale_shift_table;self.norm=root.norm_out;self.proj=root.proj_out
        def forward(self,hidden,final_temb):
            shift,scale=(self.table+final_temb.unsqueeze(1)).chunk(2,dim=1)
            return self.proj(self.norm(hidden)*(1+scale)+shift)[:,:25,:].contiguous()

    parts=[Front()]+[Pair(i) for i in range(0,24,2)]+[Tail()]
    root_ids={id(p) for p in root.parameters()};part_ids=[id(p) for part in parts for p in part.parameters()]
    if set(part_ids)!=root_ids or len(part_ids)!=len(root_ids):raise ValueError('Partition lost or duplicated learned parameters')
    plan=stage_plan();stage_inputs=[];state=dict(values)
    with torch.no_grad():
        for part,spec in zip(parts,plan):
            part.eval();inputs=tuple(state[n] for n in spec['input_shapes']);stage_inputs.append(inputs)
            result=part(*inputs);result=result if isinstance(result,tuple) else (result,)
            state.update((spec['output_bindings'][n],value) for n,value in zip(spec['output_shapes'],result))
    expected=state['velocity'].detach()
    partition=precision_metrics(expected,monolithic);comparison=precision_metrics(expected,original)
    print('Partition vs complete model:',partition,flush=True)
    print('Partition vs original FP32:',comparison,flush=True)
    if not partition['parity_passed'] or not comparison['parity_passed']:raise ValueError('Partition numerical gate failed')
    parent=ROOT/'.runtime-cache/coreai-denoiser-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='sharded-',dir=parent))
    print('Staging, retained on failure:',stage,flush=True)
    for name,value in values.items():value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}.f32le')
    expected.numpy().astype('<f4',copy=False).tofile(stage/'velocity-reference.f32le')
    original.numpy().astype('<f4',copy=False).tofile(stage/'velocity-fp32-reference.f32le')
    for index,(part,spec,inputs) in enumerate(zip(parts,plan,stage_inputs)):
        print(f'Export {index+1}/14: {spec["asset_file"]}',flush=True)
        exported=torch.export.export(part,inputs).run_decompositions(get_decomp_table())
        program=TorchConverter().add_exported_program(exported,input_names=list(spec['input_shapes']),output_names=list(spec['output_shapes'])).to_coreai()
        program.save_asset(stage/spec['asset_file']);del program,exported;gc.collect()
    ref={'schema_version':1,'profile':'denoiser-sharded','asset_file':'sharded-pipeline','function_name':'main','output_name':'velocity',
         'input_shapes':INPUT_SHAPES,'output_shape':OUTPUT_SHAPE,'stages':plan,'layer_count':24,'latent_frames':25,
         'checkpoint_prefix':'decoder.','weight_storage_dtype':'float16','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
         'original_fp32_reference_file':'velocity-fp32-reference.f32le','partition_vs_monolithic':partition,'storage16_vs_original_fp32':comparison,
         'checkpoint':{k:record[k] for k in ['repo_id','revision','filename','size','digest']},'checkpoint_full_sha256_verified':True,
         'parameter_count':sum(p.numel() for p in root.parameters()),'parameter_coverage_exactly_once':True,
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'scope':'One complete 24-layer velocity forward partitioned into 14 assets; not instrument generation',
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','coreai-torch','coreai-core']},
         'artifact_sha256':{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()},
         'device_parity_verified':False,'runtime_cache_eviction_guaranteed':False}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(stage,record)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; preserve {stage}')
    shutil.move(str(stage),str(args.out));print('Sharded denoiser prepared:',args.out,flush=True)


if __name__=='__main__':main()
