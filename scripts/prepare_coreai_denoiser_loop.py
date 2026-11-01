#!/usr/bin/env python3
"""Generate small four-step Euler references; reuse existing native denoiser assets.

Real pinned checkpoint, independent FP32 and storage16-partition trajectories.
Synthetic conditioning, guidance=1, shift=1; no CFG, waveform or music claim.
"""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
from export_coreai_denoiser import INPUT_SHAPES, OUTPUT_SHAPE, load_full_upstream
from export_coreai_denoiser_sharded import make_partitions, stage_plan, verify_bundle as verify_shards

TIMESTEPS = [1.0, 0.75, 0.5, 0.25, 0.0]
SHARED_INPUTS = ['noise', 'condition', 'context']


def trajectory(predict, inputs):
    """Feedback is local to each trajectory; never reset to reference latents."""
    import torch
    latent = inputs['noise'].clone()
    result = []
    for current, following in zip(TIMESTEPS, TIMESTEPS[1:]):
        t = torch.tensor([current], dtype=torch.float32)
        velocity = predict(latent, t, t).detach().clone()
        # Separate multiply and subtract, matching publisher Float32 Euler ODE.
        latent = latent - velocity * (current - following)
        if not torch.isfinite(latent).all() or not torch.isfinite(velocity).all():
            raise ValueError('Nonfinite loop trajectory')
        result.append((velocity, latent.detach().clone()))
    return result


def verify_sampling_source(path):
    """Verify pinned publisher schedule, timestep_r and executable ODE update."""
    tree = ast.parse(path.read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'generate_audio']
    # Publisher revisions may use a different enclosing method name: locate the
    # unique method containing both infer_steps and the ODE branch instead.
    if not methods:
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and any(a.arg == 'infer_steps' for a in n.args.args)]
    candidates = []
    for method in methods:
        for node in ast.walk(method):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "infer_method == 'ode'":
                candidates.append((method, node))
    if len(candidates) != 1:
        raise ValueError('Unique publisher ODE branch missing')
    method, ode = candidates[0]
    statements = [ast.unparse(n) for n in ode.body]
    expected = ['dt = t_curr - t_prev',
                'dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)',
                'xt = xt - vt * dt_tensor']
    if statements != expected:
        raise ValueError('Publisher ODE update changed')
    source = ast.unparse(method)
    if ('torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)' not in source
            or 'timestep_r=t_curr_tensor' not in source):
        raise ValueError('Publisher schedule/timestep_r changed')
    return ode, {'schedule': 'torch.linspace(1.0, 0.0, infer_steps + 1)',
                 'timestep_r': 't_curr_tensor', 'ode_statements': statements,
                 'scope': 'ODE update and timestep semantics only; no conditioning/CFG/cache parity claim'}


def verify_fixture(folder):
    ref = json.loads((folder/'reference.json').read_text())
    expected_steps = [{'index': i+1, 'timestep': t, 'next_timestep': TIMESTEPS[i+1], 'timestep_r': t,
                       **{key: f'step-{i+1}-{suffix}.f32le' for key, suffix in
                          [('velocity_file','velocity'), ('latent_file','latent'),
                           ('original_velocity_file','velocity-fp32'), ('original_latent_file','latent-fp32')]}}
                      for i,t in enumerate(TIMESTEPS[:-1])]
    expected_files = {v for step in expected_steps for k,v in step.items() if k.endswith('_file')}
    if (ref.get('schema_version') != 1 or ref.get('profile') != 'denoiser-loop-4'
            or ref.get('steps') != expected_steps or ref.get('timesteps') != TIMESTEPS
            or ref.get('input_shapes') != INPUT_SHAPES or ref.get('output_shape') != OUTPUT_SHAPE
            or ref.get('scalar_type') != 'float32' or ref.get('byte_order') != 'little'
            or ref.get('checkpoint_digest') != records()['model.safetensors']['digest']
            or ref.get('publisher_source_digest') != records()['modeling_acestep_v15_base.py']['digest']
            or ref.get('checkpoint_full_sha256_verified') is not True
            or ref.get('shift') != 1 or ref.get('guidance_scale') != 1
            or ref.get('method') != 'euler_ode' or ref.get('kv_cache_enabled') is not False
            or ref.get('weight_storage_dtype') != 'float16' or ref.get('compute_dtype') != 'float32'
            or ref.get('absolute_tolerance') != 1e-4 or ref.get('relative_tolerance') != 1e-3
            or ref.get('relative_rmse_tolerance') != 1e-3
            or set(ref.get('shared_input_sha256',{})) != set(SHARED_INPUTS)
            or len(ref.get('host_comparisons',[])) != 4
            or not all(row.get('velocity',{}).get('parity_passed') and row.get('latent',{}).get('parity_passed')
                       for row in ref['host_comparisons'])):
        raise ValueError('Unexpected loop contract')
    actual = {p.name for p in folder.iterdir() if p.is_file() and p.name != 'reference.json'}
    if actual != expected_files or set(ref['artifact_sha256']) != expected_files:
        raise ValueError('Incomplete or unexpected loop references')
    for name,digest in ref['artifact_sha256'].items():
        path=(folder/name).resolve()
        if (not path.is_relative_to(folder.resolve()) or path.stat().st_size != 6400
                or sha256(path) != digest):
            raise ValueError(f'Loop reference integrity mismatch: {name}')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-loop-4')
    parser.add_argument('--sharded-bundle',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-sharded')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        verify_fixture(args.out);print('Four-step loop references VERIFIED');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    publisher=records();record=publisher['model.safetensors']
    shards=verify_shards(args.sharded_bundle,record)
    checkpoint=checkpoint_path(record,args.checkpoint)
    sources=ROOT/'.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True,exist_ok=True)
    paths={n:verified_source(publisher[n],sources) for n in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    _,sampling=verify_sampling_source(paths['modeling_acestep_v15_base.py'])
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
    inputs={n:torch.from_numpy(np.fromfile(args.sharded_bundle/f'{n}.f32le',dtype='<f4').copy()).reshape(INPUT_SHAPES[n])
            for n in SHARED_INPUTS}
    def full(noise,t,r):
        return root(hidden_states=noise,encoder_hidden_states=inputs['condition'],context_latents=inputs['context'],
                    timestep=t,timestep_r=r,attention_mask=None,encoder_attention_mask=None,use_cache=False)[0]
    print('Full checkpoint SHA-256 verified; computing independent original FP32 loop',flush=True)
    with torch.no_grad():original=trajectory(full,inputs)
    convert_storage16(root);parts=make_partitions(root,make_mask)
    def partitioned(noise,t,r):
        state={**inputs,'noise':noise,'timestep':t,'timestep_r':r}
        for part,spec in zip(parts,stage_plan()):
            values=part(*(state[n] for n in spec['input_shapes']))
            if not isinstance(values,tuple):values=(values,)
            state.update((spec['output_bindings'][n],v) for n,v in zip(spec['output_shapes'],values))
        return state['velocity']
    print('Computing independent storage16 partition loop',flush=True)
    with torch.no_grad():expected=trajectory(partitioned,inputs)
    comparisons=[{'step':i+1,'velocity':precision_metrics(a[0],b[0]),'latent':precision_metrics(a[1],b[1])}
                 for i,(a,b) in enumerate(zip(expected,original))]
    if not all(r['velocity']['parity_passed'] and r['latent']['parity_passed'] for r in comparisons):
        raise ValueError(f'Loop FP32 baseline gate failed; no fixture published: {comparisons}')
    staging=ROOT/'.runtime-cache/coreai-denoiser-staging';staging.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='loop-',dir=staging));steps=[]
    for i,(a,b) in enumerate(zip(expected,original)):
        step={'index':i+1,'timestep':TIMESTEPS[i],'next_timestep':TIMESTEPS[i+1],'timestep_r':TIMESTEPS[i]}
        for key,suffix,value in [('velocity_file','velocity',a[0]),('latent_file','latent',a[1]),
                                  ('original_velocity_file','velocity-fp32',b[0]),('original_latent_file','latent-fp32',b[1])]:
            name=f'step-{i+1}-{suffix}.f32le';step[key]=name
            value.numpy().astype('<f4',copy=False).tofile(stage/name)
        steps.append(step)
    ref={'schema_version':1,'profile':'denoiser-loop-4','method':'euler_ode','timesteps':TIMESTEPS,'steps':steps,
         'shift':1,'guidance_scale':1,'kv_cache_enabled':False,'input_shapes':INPUT_SHAPES,'output_shape':OUTPUT_SHAPE,
         'weight_storage_dtype':'float16','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
         'checkpoint_digest':record['digest'],'checkpoint_full_sha256_verified':True,
         'publisher_source_digest':publisher['modeling_acestep_v15_base.py']['digest'],'publisher_sampling_semantics':sampling,
         'shared_input_sha256':{n:shards['artifact_sha256'][f'{n}.f32le'] for n in SHARED_INPUTS},
         'artifact_sha256':{p.name:sha256(p) for p in sorted(stage.iterdir())},'host_comparisons':comparisons,
         'reference_versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','numpy']},
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'scope':'Four feedback Euler steps, synthetic context/condition, no CFG/recording/waveform generation',
         'device_parity_verified':False}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_fixture(stage)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; retained {stage}')
    shutil.move(str(stage),str(args.out))
    print(json.dumps(comparisons,indent=2),flush=True)
    print('Loop references ready; existing 14 native assets unchanged:',args.out,flush=True)


if __name__=='__main__':main()
