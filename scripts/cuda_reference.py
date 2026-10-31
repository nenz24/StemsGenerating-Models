"""Validate/export one exact B reference and run the existing ACE adapter on CUDA only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from audio import canonicalize
from provenance import COMMITS,sha256,write_json
from run import cached_job,plan
from compare_guidance import prepare
from export_glitch_feedback import latest_comparison
from mac_workflow import run_command
from runtime_environment import runtime_environment
from setup_runtime import runtime_python
from stream_process import run_logged


def notebook_source_digest(data):
    """Ignore Jupyter execution state and the documented dry-run toggle, retaining source checks."""
    book=json.loads(data)
    cells=[{'cell_type':cell['cell_type'],
            'source':(''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']).replace('DRY_RUN = True','DRY_RUN = False')}
           for cell in book['cells']]
    return hashlib.sha256(json.dumps(cells,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def verify_bundle(directory):
    """Verify extracted files before installation or executing the packaged source snapshot."""
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'bundle.json').read_text())
    if manifest.get('schema_version')!=1:
        raise ValueError('Unsupported CUDA bundle schema')
    for record in manifest['files']:
        path=(directory/record['path']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Bundle file missing/changed or outside its directory')
        actual=notebook_source_digest(path.read_bytes()) if 'notebook_source_sha256' in record else sha256(path)
        expected=record.get('notebook_source_sha256',record['sha256'])
        if actual!=expected:raise ValueError('Bundle content changed')
    return manifest


def export_bundle(run,archive):
    """Include committed bridge code and the exact cached B input; exclude all model weights."""
    archive=Path(archive).resolve()
    # git archive excludes local notebook outputs, ignored runs, data, credentials and downloaded models.
    source_zip=subprocess.check_output(['git','-C',str(ROOT),'archive','--format=zip','HEAD'])
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary)
        reference=base/'reference'
        report=prepare(run,reference)
        if report['seed']!=0:
            raise ValueError('First CUDA reference expects the original seed 0 baseline')
        report.update(upstream_commit=COMMITS['ace'],bridge_commit=revision,
                      assets_sha256=sha256(ROOT/'configs/assets-base.json'))
        write_json(reference/'comparison.json',report)
        with zipfile.ZipFile(io.BytesIO(source_zip)) as source:
            needed={'scripts/cuda_reference.py','notebooks/08_cuda_reference.ipynb',
                    'notebooks/09_colab_reference.ipynb','configs/assets-base.json'}
            if not needed.issubset(source.namelist()):
                raise ValueError('Pull the committed CUDA runner/notebook before exporting')
            if hashlib.sha256(source.read('configs/assets-base.json')).hexdigest()!=sha256(ROOT/'configs/assets-base.json'):
                raise ValueError('Working model manifest differs from the committed snapshot')
            files={f'source/{name}':source.read(name) for name in source.namelist() if not name.endswith('/')}
        for path in reference.rglob('*'):
            if path.is_file():files[str(path.relative_to(base))]=path.read_bytes()
        manifest={'schema_version':1,'bridge_commit':revision,'upstream_commit':COMMITS['ace'],
                  'source_sha256':report['source_sha256'],'duration':30,'seed':report['seed'],'guidance_scale':7.0,
                  'files':[{'path':name,'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(files.items())]}
        for record in manifest['files']:
            if record['path'].endswith('.ipynb'):
                record['notebook_source_sha256']=notebook_source_digest(files[record['path']])
        archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as output:
            for name,data in files.items():output.writestr(name,data)
            output.writestr('bundle.json',json.dumps(manifest,indent=2)+'\n')
    return archive


def check_cuda(python):
    """Inspect the isolated model runtime; refuse CPU/MPS fallback before weight download."""
    code="""import json,torch
assert torch.cuda.is_available(), 'CUDA unavailable in model runtime; check GPU allocation/driver'
p=torch.cuda.get_device_properties(0)
print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':p.name,'vram_bytes':p.total_memory,'capability':list(torch.cuda.get_device_capability(0))}))
"""
    result=subprocess.run([str(python),'-c',code],capture_output=True,text=True,env=runtime_environment())
    if result.returncode:
        raise RuntimeError('CUDA model runtime check failed:\n'+(result.stderr.strip() or result.stdout.strip()))
    return json.loads(result.stdout.strip())


def run_reference(reference,output,execute=False):
    """One CUDA job with original B source/seed/CFG/steps; keep MPS baseline for comparison."""
    reference,output=Path(reference).resolve(),Path(output).resolve()
    original=json.loads((reference/'comparison.json').read_text())
    if original['assets_sha256']!=sha256(ROOT/'configs/assets-base.json') or original.get('upstream_commit')!=COMMITS['ace']:
        raise ValueError('CUDA reference model manifest/revision changed')
    source=reference/'prepared/source.wav'
    baseline=reference/'baseline_cfg7.wav'
    if sha256(source)!=original['source_sha256'] or sha256(baseline)!=original['baseline_sha256']:
        raise ValueError('B source or MPS baseline changed')
    baseline_params=original['baseline_settings']['params']
    if baseline_params['guidance_scale']!=7 or baseline_params['inference_steps']!=50 or baseline_params['duration']!=30:
        raise ValueError('Expected original 30-second, CFG 7, 50-step baseline')
    profile=dict(original['profile'],device='cuda')
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'planned','source_sha256':original['source_sha256'],'baseline_sha256':original['baseline_sha256'],
            'bridge_commit':original.get('bridge_commit'),'upstream_commit':COMMITS['ace'],
            'requested':{'device':'cuda','model':profile['model'],'seed':profile['seeds'][0],'guidance_scale':7,'steps':50,'duration':30},
            'note':'One CUDA baseline, no sweep/metadata/long-input changes. Device dtypes and RNG streams may differ; this alone cannot prove an MPS bug.'}
    write_json(output/'cuda_reference.json',report)
    command=run_command(ROOT,profile,reference/'prepared',output/'generation',execute=execute)
    command+=['--max-job-seconds','30','--guidance-scale','7','--capture-decoder',str(output/'capture')]
    try:
        if execute:report['hardware']=check_cuda(runtime_python())
        run_logged(command,env=runtime_environment(),check=True)
        if not execute:
            print('PLAN ONLY: no CUDA inference executed',flush=True)
            return report
        experiment=json.loads((output/'generation/experiment.json').read_text())
        job=experiment['jobs'][0]
        folder=output/'generation'/f"000_A_{job['instrument']}_seed{job['seed']}"
        result=cached_job(folder,job,None)
        if result is None or job['source_sha256']!=original['source_sha256'] or result['device']!='cuda':
            raise ValueError('CUDA result/input verification failed')
        before,after=baseline_params,result['settings']['params']
        changed={key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
        if changed-{'src_audio'} or original['baseline_settings']['config']!=result['settings']['config']:
            raise ValueError('Unexpected generation parameter changes; result retained for inspection')
        capture=json.loads((output/'capture/capture.json').read_text())
        for name,path in (('mps_baseline',baseline),('cuda',folder/result.get('stem_file','stem.wav'))):
            canonicalize(path,output/(name+'_listen.wav'),duration=30,gain=0.7,max_duration=30)
        report.update(status='generated',settings=result['settings'],decoder_capture=capture,
                      candidate_sha256=result['stem_sha256'],listening_gain_both=0.7)
        shutil.copyfile(output/'generation/session.json',output/'session.json')
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output/'cuda_reference.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--run',type=Path,help='Completed Mac decoder comparison for export')
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.export:
        run=args.run.resolve() if args.run else latest_comparison()
        archive=args.archive or ROOT/'runs'/('cuda_reference_'+uuid.uuid4().hex[:12]+'.zip')
        print('Upload this bundle to your CUDA server:',export_bundle(run,archive))
        print('No model inference/download ran on this Mac. Bundle excludes weights and uncommitted notebook outputs.')
        return
    if args.reference is None:parser.error('--reference is required on the CUDA server')
    if (args.reference.parent/'bundle.json').is_file():verify_bundle(args.reference.parent)
    output=args.out or ROOT/'runs'/('cuda_reference_'+uuid.uuid4().hex[:12])
    run_reference(args.reference,output,args.execute)
    print('Output:',output)


if __name__=='__main__':main()
