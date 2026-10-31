"""One 30-second drum probe using the existing pinned ACE-Step CUDA runtime."""
import argparse
import json
from pathlib import Path
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


def generate(root,source,output,*,start=0,execute=False):
    root,source,output=[Path(p).resolve() for p in (root,source,output)]
    single.attach(root)
    from provenance import sha256,write_json
    from runtime_environment import runtime_environment
    from stream_process import run_logged
    from mac_workflow import run_command
    from compare_bass_tone import caption_worker,require_fp32
    from cuda_precision import FLOAT32_WORKER
    from run import cached_job
    config=root/'configs/acoustic-drums.json'
    if config.is_file() and json.loads(config.read_text())!=PRESET:
        raise ValueError('Drum preset differs; preserve the old experiment and use a new preset/version')
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','instrument':'drums','preset':PRESET,
            'note':'Independent 30-second instrument probe; no bass regeneration, full-song drum quality or inferred meter/BPM claims.'}
    write_json(output/'drum_probe.json',report)
    try:
        metadata=single.prepare(source,output/'prepared',start=start,duration=30)
        report.update(status='planned',input=metadata)
        command=run_command(root,PRESET,output/'prepared',output/'generation',execute=execute)
        command+=['--max-job-seconds','30','--guidance-scale','7','--capture-decoder',str(output/'capture')]
        if execute:
            single.model_python(root)
            command=[command[0],'-c',caption_worker(FLOAT32_WORKER,PRESET['caption']),*command[1:]]
        run_logged(command,env=runtime_environment(root),check=True)
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
                  'duration':30,'task_type':'lego','instrumental':True,
                  'instruction':'Generate the drums track based on the audio context:',
                  'enable_normalization':False,'thinking':False,'use_cot_caption':False,
                  'use_cot_metas':False,'use_cot_language':False}
        if any(params.get(k)!=v for k,v in expected.items()):raise ValueError('Actual drum settings differ')
        if config.get('seeds')!=[0] or config.get('batch_size')!=1 or config.get('use_random_seed') is not False:
            raise ValueError('Unexpected drum seed/batch settings')
        capture=json.loads((output/'capture/capture.json').read_text());require_fp32(capture)
        if sha256(source)!=metadata['original_sha256'] or sha256(output/'prepared/source.wav')!=metadata['source_sha256']:
            raise ValueError('Input changed during drum generation')
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
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start',type=float,default=0);p.add_argument('--execute',action='store_true')
    a=p.parse_args();print(json.dumps(generate(a.root,a.source,a.out,start=a.start,execute=a.execute),indent=2))
