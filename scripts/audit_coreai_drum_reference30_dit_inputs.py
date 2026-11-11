#!/usr/bin/env python3
"""Compare CPU DiT controls to native velocities on exact saved native inputs.

Five forward evaluations per CPU variant, not another50-step trajectory.
Default is plan-only. Requires existing pinned base weights on the user's Mac.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASE_STEPS = (1, 2, 6, 11, 26)


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().numpy().astype('<f4').tobytes()).hexdigest()


def native_input_cases(native, noise, times):
    import torch
    if (list(noise.shape) != [1,750,64] or list(times.shape) != [51]
            or noise.dtype != torch.float32 or times.dtype != torch.float32
            or not torch.isfinite(noise).all() or not torch.isfinite(times).all()
            or times[0] != 1 or times[-1] != 0 or not (times[:-1] > times[1:]).all()):
        raise ValueError('Invalid original noise/timesteps')
    cases = []
    for step in CASE_STEPS:
        latent = noise if step == 1 else native.get(f'native-step-{step-1:02d}-latent.f32le')
        velocity = native.get(f'native-step-{step:02d}-velocity.f32le')
        if (latent is None or velocity is None or list(latent.shape) != [1,750,64]
                or list(velocity.shape) != [2,750,64] or latent.dtype != torch.float32
                or velocity.dtype != torch.float32 or not torch.isfinite(latent).all()
                or not torch.isfinite(velocity).all()):
            raise ValueError(f'Missing or invalid exact native input/output pair for step{step}')
        cases.append({'step':step,'input_post_step':step-1,'latent':latent,
                      'timestep':times[step-1:step], 'native_velocity':velocity})
    return cases


def measure_cases(predict, cases):
    import torch
    outputs = {}
    with torch.no_grad():
        for case in cases:
            before = (tensor_sha(case['latent']),tensor_sha(case['timestep']))
            value = predict(case['latent'], case['timestep'])
            if before != (tensor_sha(case['latent']),tensor_sha(case['timestep'])):
                raise ValueError('Predictor mutated the fixed native input')
            if list(value.shape) != [2,750,64] or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Invalid same-input CPU velocity')
            outputs[case['step']] = value.detach().contiguous().clone()
    return outputs


def execute(args):
    import torch
    from safetensors import safe_open
    from audit_coreai_drum_reference30 import verify_capture, read_tensor
    from compare_coreai_drum_lyrics import local_checkpoint
    from export_coreai_decoder import sha256
    from export_coreai_denoiser import load_full_upstream
    from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
    from generate_coreai_drum_reference import cached_predictor
    from prepare_coreai_drum_reference30 import WINDOW, plan, partitions, run_partitions
    from replay_coreai_drum_reference30_sampler import native_capture
    torch.set_num_threads(2)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Existing DiT comparison preserved: '+str(out))
    if any(out.is_relative_to(p) for p in (args.capture.resolve(),args.publisher_dir.resolve(),
                                         ROOT/'baselines',ROOT/'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Output must be outside captures/publisher/models/baselines')
    captured = verify_capture(args.capture)
    native, manifest = native_capture(args.native_archive)
    if manifest.get('velocity_trace_steps') != 50:
        raise ValueError('A complete report-bound native velocity trace is required')
    cases = native_input_cases(native, read_tensor(args.capture,'initial-noise.f32le'),
                                read_tensor(args.capture,'timesteps.f32le'))
    publisher = records()
    base_record = publisher['model.safetensors']
    if base_record['digest'] != captured['checkpoint_sha256']['base']:
        raise ValueError('Accepted/base checkpoint differs')
    paths = {n:args.publisher_dir/n for n in
             ('modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json')}
    for name,path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('Missing existing pinned publisher source; no download: '+name)
        verified_source(publisher[name],args.publisher_dir)
        if sha256(path) != captured['publisher_source_sha256'][name]:
            raise ValueError('Accepted publisher identity differs: '+name)
    base = local_checkpoint(base_record,args.base_checkpoint)
    cfg,kind,rotary = load_full_upstream(paths)
    if cfg.patch_size != 2 or cfg.num_hidden_layers != 24:
        raise ValueError('Original DiT architecture differs')
    out.mkdir(parents=True)
    report = {'schema_version':1,'profile':'frozen30-dit-exact-native-inputs','status':'loading',
              'native_archive_sha256':sha256(args.native_archive),'native_run_id':manifest['run_id'],
              'velocity_trace_sha256':manifest['velocity_trace_sha256'],'native_report_sha256':manifest['report_sha256'],
              'capture_report_sha256':sha256(args.capture/'report.json'),'base_checkpoint_sha256':base_record['digest'],
              'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
              'scope':'Five exact native input/output pairs: steps1/2/6/11/26. No reconstruction of missing step49 state.',
              'partition_weight_storage_dtype':'bfloat16','compute_dtype':'float32',
              'publisher_control_dtype':'float32 with original learned values',
              'full_publisher_calls':0,'partition_calls':0,'APG_calls':0,'VAE_calls':0,'native_calls':0,'native_exports':0,
              'baseline_modified':False,'cases':{},'torch':torch.__version__}
    def save():
        temp=out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'report.json')
    save()
    try:
        print('Load verified local base checkpoint; no source/Qwen/VAE/export',flush=True)
        with torch.device('meta'):
            model=kind(cfg).eval()
        with safe_open(str(base),framework='pt',device='cpu') as stream:
            state={k[len('decoder.'):]:stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
            null=stream.get_tensor('null_condition_emb').float()
        model.load_state_dict(state,strict=True,assign=True);del state
        model.rotary_emb=rotary(cfg,device='cpu')
        model.to(device='cpu',dtype=torch.float32).requires_grad_(False)
        condition,context=read_tensor(args.capture,'condition.f32le'),read_tensor(args.capture,'source-context.f32le')
        publisher_predict=cached_predictor(model,condition,null,context)
        def full_predict(x,t):
            value=publisher_predict(x,t)
            report['full_publisher_calls']+=1
            print(f'Publisher CPU same-input case{report["full_publisher_calls"]}/5',flush=True)
            return value
        full=measure_cases(full_predict,cases)
        # Release the publisher KV cache before replacing learned storage wrappers.
        del publisher_predict,full_predict
        convert_storage16(model,'bfloat16')
        parts=partitions(model,null,sys.modules[kind.__module__].create_4d_mask,WINDOW)
        stages=plan()['stages']
        def partition_predict(x,t):
            value=run_partitions(parts,stages,{'noise':x,'condition':condition,'context':context,'timestep':t})
            report['partition_calls']+=len(parts)
            print(f'Partitioned CPU same-input case{report["partition_calls"]//14}/5',flush=True)
            return value
        partitioned=measure_cases(partition_predict,cases)
        for case in cases:
            step=case['step'];data={'input_post_step':case['input_post_step'],'latent_sha256':tensor_sha(case['latent']),
                                   'timestep':float(case['timestep'][0]),'native_velocity_sha256':tensor_sha(case['native_velocity']),
                                   'comparisons':{}}
            for label,actual,reference in [('publisher_vs_native',full[step],case['native_velocity']),
                                           ('partitioned_vs_native',partitioned[step],case['native_velocity']),
                                           ('partitioned_vs_publisher',partitioned[step],full[step])]:
                data['comparisons'][label]={name:precision_metrics(a,b) for name,a,b in
                    [('combined',actual,reference),('conditional',actual[:1],reference[:1]),('unconditional',actual[1:],reference[1:])]}
            report['cases'][str(step)]=data
            for label,value in [('publisher',full[step]),('partitioned',partitioned[step])]:
                value.numpy().astype('<f4').tofile(out/f'{label}-step-{step:02d}-velocity.f32le')
        if report['full_publisher_calls']!=5 or report['partition_calls']!=70:
            raise ValueError('Incomplete5-case CPU comparison')
        report['comparison_verdicts']={label:'PASS' if all(metric['parity_passed']
                    for case in report['cases'].values() for metric in case['comparisons'][label].values()) else 'NOT PASSED' for label in
                    ('publisher_vs_native','partitioned_vs_native','partitioned_vs_publisher')}
        report.update(status='completed',full_native_trajectory_parity='NOT EVALUATED; same-input cases do not replace original full-trajectory failures')
        report['artifact_sha256']={p.name:sha256(p) for p in out.glob('*.f32le')}
        save()
        with tempfile.TemporaryFile() as temp:
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ['report.json',*report['artifact_sha256']]:archive.write(out/name,name)
            temp.seek(0)
            with (out/'same-input-dit-review.zip').open('xb') as destination:
                import shutil
                shutil.copyfileobj(temp,destination)
        print('Private review ZIP:',out/'same-input-dit-review.zip',flush=True)
        print('Original strict comparison verdicts:',json.dumps(report['comparison_verdicts']),flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--native-archive',type=Path,required=True)
    parser.add_argument('--capture',type=Path,default=ROOT/'.runtime-cache/drums-reference30-v3')
    parser.add_argument('--publisher-dir',type=Path,default=ROOT/'.runtime-cache/coreai-generator-source-publisher')
    parser.add_argument('--base-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'.runtime-cache/drums-reference30-dit-inputs-v1')
    args=parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: five exact native inputs, steps1/2/6/11/26; five publisher CPU controls +70 BF16-storage FP32 partition calls.')
        print('No50-step generation, missing-state reconstruction, APG/VAE/source/Qwen, download, native export or baseline change.')
        return
    execute(args)


if __name__=='__main__':main()
