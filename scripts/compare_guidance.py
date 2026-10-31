"""Compare cached CFG 7 with one CFG 1 B generation, preserving input/seed/steps."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from audio import canonicalize
from provenance import sha256,write_json
from run import cached_job
from context_comparison import create_input
from export_glitch_feedback import latest_comparison
from mac_workflow import check_model_runtime,run_command
from runtime_environment import runtime_environment
from setup_runtime import runtime_python
from stream_process import run_logged


def prepare(run,output):
    """Snapshot B and its completed baseline; do not regenerate CFG 7 or reuse it as CFG 1."""
    run,output=Path(run).resolve(),Path(output).resolve()
    original=json.loads((run/'comparison.json').read_text())
    experiment=json.loads((run/'default_result/experiment.json').read_text())
    profile=original['profile']
    if original.get('status')!='decoded' or experiment.get('status')!='generated' or len(experiment['jobs'])!=1:
        raise ValueError('Expected completed decoder comparison with one B baseline')
    job=experiment['jobs'][0]
    folder=run/'default_result'/f"000_A_{job['instrument']}_seed{job['seed']}"
    baseline=cached_job(folder,job,experiment.get('execution_signature'))
    if baseline is None:
        raise ValueError('Missing completed CFG 7 baseline')
    params=baseline['settings']['params']
    expected={'task_type':'lego','inference_steps':50,'guidance_scale':7.0,'duration':30.0,
              'instruction':f"Generate the {profile['instrument']} track based on the audio context:",
              'caption':f"Isolated {profile['instrument']} accompaniment following the source harmony, timing and dynamics.",
              'thinking':False,'enable_normalization':False,'seed':job['seed'],
              'instrumental':True,'lyrics':'[Instrumental]','repainting_start':0.0,'repainting_end':-1,
              'use_cot_caption':False,'use_cot_metas':False,'use_cot_language':False}
    if (profile['model']!='acestep-v15-base' or profile['seeds']!=[job['seed']]
        or profile['instrument']!=job['instrument'] or job['duration']!=30
        or any(params.get(key)!=value for key,value in expected.items())):
        raise ValueError('Baseline settings differ from the fixed comparison protocol')
    source=run/'prepared/source.wav'
    if sha256(source)!=original['source_sha256'] or sha256(source)!=job['source_sha256']:
        raise ValueError('B input changed')
    if original['assets_sha256']!=sha256(ROOT/'configs/assets-base.json'):
        raise ValueError('Model manifest changed since baseline')
    output.mkdir(parents=True,exist_ok=False)
    prepared=create_input(output/'prepared',source,30,'Exact B input; only requested guidance changes to 1',copy_exact=True)
    if sha256(prepared/'source.wav')!=job['source_sha256']:
        raise ValueError('B snapshot differs')
    shutil.copyfile(folder/baseline.get('stem_file','stem.wav'),output/'baseline_cfg7.wav')
    report={'status':'planned','profile':profile,'baseline_settings':baseline['settings'],
            'source_sha256':job['source_sha256'],'baseline_sha256':sha256(output/'baseline_cfg7.wav'),
            'baseline_guidance':7.0,'candidate_guidance':1.0,'inference_steps':50,'seed':job['seed'],
            'note':'One new B generation; original source/seed/steps/caption retained. CFG 1 bypasses the amplified/batched guidance branch. Runtime and encoder sampling can vary; not proof of a single bug.'}
    write_json(output/'comparison.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve() if args.run else latest_comparison()
    output=args.out.resolve() if args.out else ROOT/'runs'/('guidance_comparison_'+uuid.uuid4().hex[:12])
    report=prepare(run,output)
    print('Folder pembanding:',output,flush=True)
    print('CFG 7 dari cache; CFG 1 memakai input/seed/50 langkah/caption yang sama.',flush=True)
    command=run_command(ROOT,report['profile'],output/'prepared',output/'cfg1_result',execute=args.execute)
    command+=['--max-job-seconds','30','--guidance-scale','1']
    if not args.execute:
        run_logged(command,check=True)
        print('PLAN ONLY: tidak ada generasi baru',flush=True)
        return
    try:
        print(json.dumps(check_model_runtime(runtime_python()),indent=2),flush=True)
        run_logged(command,env=runtime_environment(),check=True)
        experiment=json.loads((output/'cfg1_result/experiment.json').read_text())
        job=experiment['jobs'][0]
        directory=output/'cfg1_result'/f"000_A_{job['instrument']}_seed{job['seed']}"
        candidate=cached_job(directory,job,None)
        if candidate is None or candidate['settings']['params']['guidance_scale']!=1.0:
            raise ValueError('CFG 1 was not actually generated')
        if job['source_sha256']!=report['source_sha256']:
            raise ValueError('Candidate input differs from baseline')
        before=report['baseline_settings']['params']
        after=candidate['settings']['params']
        changed={key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
        if changed-{'guidance_scale','src_audio'} or candidate['settings']['config']!=report['baseline_settings']['config']:
            raise ValueError('Additional generation settings changed; comparison retained for inspection')
        if sha256(output/'baseline_cfg7.wav')!=report['baseline_sha256']:
            raise ValueError('Baseline changed during comparison')
        for label,path in (('cfg7',output/'baseline_cfg7.wav'),('cfg1',directory/candidate.get('stem_file','stem.wav'))):
            canonicalize(path,output/(label+'_listen.wav'),duration=30,gain=0.7,max_duration=30)
        report.update(status='generated',candidate_settings=candidate['settings'],candidate_sha256=candidate['stem_sha256'],listening_gain_both=0.7)
        print('Bandingkan cfg7_listen.wav dan cfg1_listen.wav pada folder di atas.',flush=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output/'comparison.json',report)


if __name__=='__main__':main()
