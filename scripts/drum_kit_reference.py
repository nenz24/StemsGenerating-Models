"""One controlled drum-kit reference comparison; keep piano and preset unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import ace_context_chunks as chunks
import drum_probe as drum


def cached_chunk(folder,plan,index):
    progress=json.loads((folder/'progress.json').read_text())
    run=(folder/progress[str(index)]).resolve()
    if not run.is_relative_to(folder/f'chunk_{index:03d}'):raise ValueError('Cached chunk path escapes run')
    stem=chunks.verify_chunk(run,plan,plan['chunks'][index])
    return stem,json.loads((run/'drum_probe.json').read_text())


def archived_chunk(archive,out,plan,index):
    folder=out/f'baseline_chunk_{index:03d}';folder.mkdir()
    raw=f'chunks/chunk_{index:03d}_drums.wav'
    metadata=f'stitched/chunk_reports/{index:03d}.json'
    (folder/'drums.wav').write_bytes(archive.read(raw))
    report=json.loads(archive.read(metadata))['generation_report']
    chunks.write_json(folder/'drum_probe.json',report)
    return chunks.verify_chunk(folder,plan,plan['chunks'][index]),report


def package(out,success):
    for name,full in [('drum_kit_feedback.zip',False),('drum_kit_result.zip',True)]:
        if full and not success:continue
        with zipfile.ZipFile(out/name,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for relative in ['comparison.json','reference_preview.mp3','control_preview.mp3',
                'control_mix_preview.mp3','candidate/drum_probe.json',
                'candidate/drums_preview.mp3','candidate/mix_preview.mp3',
                'candidate/capture/capture.json',
                'candidate/generation/000_A_drums_seed0/result.json']:
                if (out/relative).is_file():z.write(out/relative,relative)
            if full:
                for relative in ['reference.wav','control.wav','candidate/drums.wav','candidate/mix.wav']:
                    z.write(out/relative,relative)


def compare(root,output,*,plan_path=None,result_archive=None,source=None,target=0,reference=1,execute=False,
            expected_control_sha=None,expected_reference_sha=None):
    root,output=Path(root).resolve(),Path(output).resolve()
    if (plan_path is None)==(result_archive is None):raise ValueError('Choose an existing plan or a result ZIP')
    if target==reference:raise ValueError('Reference and comparison chunks must be different')
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','target_chunk':target,'reference_chunk':reference,
            'note':'One reference-conditioned comparison. No full-song regeneration, EQ, kick replacement or baseline overwrite.'}
    chunks.write_json(output/'comparison.json',report)
    try:
        if plan_path is not None:
            folder,plan,_=chunks.load_plan(plan_path)
            source=chunks.generation_source(folder,plan)
            target_wav,control=cached_chunk(folder,plan,target)
            reference_wav,reference_report=cached_chunk(folder,plan,reference)
        else:
            if source is None:raise ValueError('Original piano is required with the result ZIP')
            source=Path(source).resolve()
            with zipfile.ZipFile(result_archive) as z:
                if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>256*1024**2:
                    raise ValueError('Invalid or oversized result archive')
                if any(i.file_size>100*1024**2 for i in z.infolist()):raise ValueError('Result member too large')
                required={'plan.json'}|{f'chunks/chunk_{i:03d}_drums.wav' for i in [target,reference]}|{f'stitched/chunk_reports/{i:03d}.json' for i in [target,reference]}
                if not required.issubset(z.namelist()):
                    raise ValueError('Use drum_chunked_result.zip with raw chunk WAVs and reports; feedback ZIP is insufficient')
                plan=json.loads(z.read('plan.json'))
                if chunks.sha256(source)!=plan['original_sha256']:raise ValueError('Piano differs from the saved experiment')
                target_wav,control=archived_chunk(z,output,plan,target)
                reference_wav,reference_report=archived_chunk(z,output,plan,reference)
        if chunks.instrument_of(plan)!='drums' or plan['preset']!=drum.PRESET:
            raise ValueError('Expected the original drum preset')
        if expected_control_sha is not None and chunks.sha256(target_wav)!=expected_control_sha:
            raise ValueError('Control differs from the listener-reviewed chunk 0')
        if expected_reference_sha is not None and chunks.sha256(reference_wav)!=expected_reference_sha:
            raise ValueError('Reference differs from the listener-reviewed chunk 1')
        if control['settings']['params'].get('reference_audio') is not None:
            raise ValueError('Control already uses a reference; choose the original unreferenced run')
        rchunk=plan['chunks'][reference];target_chunk=plan['chunks'][target]
        reference_seconds=min(30.,(rchunk['core_end']-rchunk['core_start'])/chunks.RATE)
        if reference_seconds<10:raise ValueError('Reference core shorter than 10 seconds')
        local_start=(rchunk['core_start']-rchunk['input_start'])/chunks.RATE+((rchunk['core_end']-rchunk['core_start'])/chunks.RATE-reference_seconds)/2
        drum.single.attach(root)
        drum.single.canonicalize_long(reference_wav,output/'reference.wav',start=local_start,
                                     duration=reference_seconds,gain=1,max_duration=600)
        import soundfile as sf
        import numpy as np
        audio,_=sf.read(output/'reference.wav')
        if not np.isfinite(audio).all() or np.max(np.abs(audio))<1e-4:raise ValueError('Reference is silent or invalid')
        shutil.copyfile(target_wav,output/'control.wav')
        for src,dst in [('reference.wav','reference_preview.mp3'),('control.wav','control_preview.mp3')]:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(output/src),'-af','volume=0.7',
                            '-c:a','libmp3lame','-b:a','192k',str(output/dst)],check=True)
        report.update(status='planned',original_source_sha256=plan['original_sha256'],
            control_stem_sha256=chunks.sha256(target_wav),reference_chunk_stem_sha256=chunks.sha256(reference_wav),
            reference_excerpt_sha256=chunks.sha256(output/'reference.wav'),reference_seconds=reference_seconds,
            reference_song_start_seconds=rchunk['input_start']/chunks.RATE+local_start,
            source_start_seconds=target_chunk['input_start']/chunks.RATE,
            source_duration_seconds=(target_chunk['input_end']-target_chunk['input_start'])/chunks.RATE,
            model_duration=target_chunk['model_duration'],preset=plan['preset'],control_generation_report=control)
        chunks.write_json(output/'comparison.json',report)
        candidate=drum.generate(root,source,output/'candidate',start=report['source_start_seconds'],
            duration=report['source_duration_seconds'],reference_audio=output/'reference.wav',execute=execute)
        if candidate['input']['source_sha256']!=control['input']['source_sha256']:
            raise ValueError('Candidate piano model input differs from the control')
        if not execute:return report
        old,new=control['settings'],candidate['settings']
        differences={k for k in set(old['params'])|set(new['params']) if old['params'].get(k)!=new['params'].get(k)}
        if differences-{'src_audio','reference_audio'} or old['config']!=new['config']:
            raise ValueError(f'Unexpected settings differences: {sorted(differences)}')
        report.update(status='generated',actual_parameter_differences=sorted(differences),
                      candidate_stem_sha256=candidate['stem_sha256'],same_piano_model_input=True)
        # Use exactly the candidate's prepared piano for a comparable control mix.
        from audio import listening_mix
        listening_mix(output/'candidate/prepared/original.wav',output/'control.wav',output/'control_mix.wav')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(output/'control_mix.wav'),
            '-c:a','libmp3lame','-b:a','192k',str(output/'control_mix_preview.mp3')],check=True)
        chunks.write_json(output/'comparison.json',report);package(output,True)
        return report
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        chunks.write_json(output/'comparison.json',report)
        try:package(output,False)
        except Exception:pass
        raise
    finally:chunks.write_json(output/'comparison.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan',type=Path);source.add_argument('--result-zip',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    print(json.dumps(compare(a.root,a.out,plan_path=a.plan,result_archive=a.result_zip,source=a.source,execute=a.execute),indent=2))
