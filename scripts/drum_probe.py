"""Drum probes or full-song single-pass using the pinned ACE-Step CUDA runtime."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import electric_full_song as single

PRESET = {
    'id':'acoustic-drums-v1','model':'acestep-v15-base','device':'cuda',
    'precision':'float32','instrument':'drums','seeds':[0],
    'guidance_scale':7.0,'inference_steps':50,'offload_to_cpu':True,
    'caption':'Isolated acoustic drum kit accompaniment. A restrained, natural groove with kick, snare and closed hi-hat, following the piano performance timing, phrasing and dynamics. Clear drum hits, tasteful fills and space between phrases. Drums only, no vocals or melodic instruments.',
}


def package(output, success):
    name='drum_feedback.zip'
    with zipfile.ZipFile(output/name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ['drum_probe.json','prepared/input.json','generation/experiment.json',
                         'generation/session.json','generation/000_A_drums_seed0/result.json',
                         'capture/capture.json','drums_preview.mp3','mix_preview.mp3']:
            if (output/relative).is_file():archive.write(output/relative,relative)
    if success:
        with zipfile.ZipFile(output/'drum_result.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in ['drum_probe.json','drums.wav','mix.wav']:
                archive.write(output/relative,relative)


def generate(root,source,output,*,start=0,duration=30,execute=False,reference_audio=None):
    root,source,output=[Path(p).resolve() for p in (root,source,output)]
    single.attach(root)
    from provenance import sha256,write_json
    from runtime_environment import runtime_environment
    from stream_process import run_logged
    from mac_workflow import run_command
    from compare_bass_tone import caption_worker,require_fp32
    from cuda_precision import FLOAT32_WORKER
    from run import cached_job,plan
    config=root/'configs/acoustic-drums.json'
    if config.is_file() and json.loads(config.read_text())!=PRESET:
        raise ValueError('Drum preset differs; preserve the old experiment and use a new preset/version')
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','instrument':'drums','preset':PRESET,
            'mode':'single_pass','model_calls_requested':1,
            'note':'One continuous drum context; no song chunking or stitching. VAE may tile internally. No bass regeneration or inferred meter/BPM claims.'}
    write_json(output/'drum_probe.json',report)
    try:
        metadata=single.prepare(source,output/'prepared',start=start,duration=duration)
        if reference_audio is not None:
            reference_audio=Path(reference_audio).resolve()
            # Freeze reference bytes; no normalization, EQ, retiming or random crop.
            frozen=output/('reference'+reference_audio.suffix.lower())
            original_ref_hash=sha256(reference_audio)
            shutil.copyfile(reference_audio,frozen)
            if sha256(frozen)!=original_ref_hash:raise ValueError('Reference changed during copy')
            reference_audio=frozen
            report['reference_audio']={'path':str(frozen),'sha256':original_ref_hash}
        report.update(status='planned',input=metadata)
        write_json(output/'drum_probe.json',report)
        if metadata['model_duration']<=30 and reference_audio is None:
            command=run_command(root,PRESET,output/'prepared',output/'generation',execute=execute)
            command+=['--max-job-seconds','30','--guidance-scale','7','--capture-decoder',str(output/'capture')]
            if execute:
                single.model_python(root)
                command=[command[0],'-c',caption_worker(FLOAT32_WORKER,PRESET['caption']),*command[1:]]
            run_logged(command,env=runtime_environment(root),check=True)
        elif execute:
            command=[str(single.model_python(root)),str(Path(__file__).resolve()),'--worker',
                     '--root',str(root),'--out',str(output)]
            run_logged(command,env=runtime_environment(root),check=True)
        else:
            generation=output/'generation';generation.mkdir()
            experiment=plan(output/'prepared','ace','drums',[0])
            experiment.update(max_job_seconds=600,device_requested='cuda',ace_model=PRESET['model'],
                              offload_to_cpu=True,guidance_scale=7,mode='single_pass')
            write_json(generation/'experiment.json',experiment)
        if not execute:return report
        experiment=json.loads((output/'generation/experiment.json').read_text())
        if experiment.get('status')!='generated' or len(experiment['jobs'])!=1:
            raise ValueError('Expected one completed drum job')
        job=experiment['jobs'][0];folder=output/'generation/000_A_drums_seed0'
        result=cached_job(folder,job,None)
        if result is None or result.get('device')!='cuda' or job.get('instrument')!='drums' or job['source_sha256']!=metadata['source_sha256']:
            raise ValueError('Drum device/input/result verification failed')
        params,config=result['settings']['params'],result['settings']['config']
        expected={'caption':PRESET['caption'],'seed':0,'guidance_scale':7,'inference_steps':50,
                  'duration':metadata['model_duration'],'task_type':'lego','instrumental':True,
                  'instruction':'Generate the drums track based on the audio context:',
                  'enable_normalization':False,'thinking':False,'use_cot_caption':False,
                  'use_cot_metas':False,'use_cot_language':False}
        if reference_audio is not None:
            expected['reference_audio']=str(reference_audio)
        if any(params.get(k)!=v for k,v in expected.items()):raise ValueError('Actual drum settings differ')
        if config.get('seeds')!=[0] or config.get('batch_size')!=1 or config.get('use_random_seed') is not False:
            raise ValueError('Unexpected drum seed/batch settings')
        capture=json.loads((output/'capture/capture.json').read_text());require_fp32(capture)
        if sha256(source)!=metadata['original_sha256'] or sha256(output/'prepared/source.wav')!=metadata['source_sha256']:
            raise ValueError('Input changed during drum generation')
        if reference_audio is not None and sha256(reference_audio)!=report['reference_audio']['sha256']:
            raise ValueError('Reference changed during generation')
        single.export_audio(folder/result['stem_file'],output/'drums.wav',metadata['valid_frames'])
        single.export_audio(folder/'mix.wav',output/'mix.wav',metadata['valid_frames'])
        for src,dst,gain in [('drums.wav','drums_preview.mp3',.7),('mix.wav','mix_preview.mp3',1)]:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(output/src),'-af',f'volume={gain}',
                            '-c:a','libmp3lame','-b:a','192k',str(output/dst)],check=True)
        report.update(status='generated',settings=result['settings'],decoder_capture=capture,
                      stem_sha256=sha256(output/'drums.wav'),mix_sha256=sha256(output/'mix.wav'),
                      inference_seconds=result.get('inference_seconds'),
                      peak_cuda_allocated_bytes=result.get('peak_cuda_allocated_bytes'))
        write_json(output/'drum_probe.json',report);package(output,True)
        return report
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(output/'drum_probe.json',report)
        try:package(output,False)
        except Exception:pass
        raise
    finally:write_json(output/'drum_probe.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--duration',type=float,default=30)
    group.add_argument('--full-song',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--reference-audio',type=Path)
    p.add_argument('--start',type=float,default=0);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.worker:
        reference=None
        report_path=a.out.resolve()/'drum_probe.json'
        if report_path.is_file():
            report=json.loads(report_path.read_text())
            if report.get('reference_audio'):
                import hashlib
                item=report['reference_audio'];reference=Path(item['path']).resolve()
                if not reference.is_relative_to(a.out.resolve()) or hashlib.sha256(reference.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Worker reference path/hash mismatch')
        single.execute_worker(a.root.resolve(),a.out.resolve(),preset=PRESET,instrument='drums',reference_audio=reference)
    else:
        if a.source is None:p.error('--source is required')
        print(json.dumps(generate(a.root,a.source,a.out,start=a.start,duration=None if a.full_song else a.duration,execute=a.execute,reference_audio=a.reference_audio),indent=2))
