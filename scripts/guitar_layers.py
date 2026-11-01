"""Two guitar layers on the same piano interval and the same frozen style reference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import soundfile as sf

import guitar_probe as guitar


def read(path, frames):
    audio, rate = sf.read(path, always_2d=True, dtype='float32')
    if rate != 32000 or audio.shape != (frames, 2) or not np.isfinite(audio).all():
        raise ValueError('Layer audio must be finite stereo 32kHz at the exact piano length')
    return audio


def export_mix(output, strum, pick, piano):
    combined = .45 * strum + .45 * pick
    mix = .5 * (piano + combined)
    peak = max(float(np.max(np.abs(combined))), float(np.max(np.abs(mix))))
    gain = min(1., .9 / peak) if peak else 1.
    for name, audio in [('guitars_layered', combined), ('piano_guitars_mix', mix)]:
        sf.write(output / (name + '.wav'), audio * gain, 32000, subtype='PCM_24')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(output / (name + '.wav')),
                        '-c:a','libmp3lame','-b:a','192k',str(output / (name + '.mp3'))], check=True)
    return gain


def generate(root, source, output, reference_audio, *, start=0, execute=False):
    root, source, output, reference_audio = [Path(p).resolve() for p in (root, source, output, reference_audio)]
    guitar.single.attach(root)
    from provenance import sha256, write_json
    output.mkdir(parents=True, exist_ok=False)
    reference_hash = sha256(reference_audio)
    frozen = output / ('style_reference' + reference_audio.suffix.lower())
    shutil.copyfile(reference_audio, frozen)
    if sha256(frozen) != reference_hash: raise ValueError('Layer style reference changed during copy')
    report = {'status':'preparing', 'source_sha256':sha256(source), 'start_seconds':start,
              'duration_seconds':30, 'reference_audio_sha256':reference_hash,
              'reference_file':frozen.name, 'reference_source_name':reference_audio.name,
              'reference_format':reference_audio.suffix.lower(), 'model_calls_requested':2,
              'layer_weights':{'strum':0.45,'pick':0.45},
              'note':'Two newly generated performances at the same piano position. Chunk 0 and chunk 2 from different song times are not overlaid. Shared reference requests sound identity; technique, harmony and quality still require listening.'}
    write_json(output / 'layers.json', report)
    try:
        layers = {}
        for name, variant in [('strum','strum-layer'),('pick','pick-layer')]:
            layers[name] = guitar.generate(root, source, output / name, start=start, duration=30,
                                          execute=execute, variant=variant, reference_audio=frozen)
        a, b = layers['strum'], layers['pick']
        if a['input'] != b['input'] or a['input']['duration'] != 30 or a['engine_sha256'] != b['engine_sha256']:
            raise ValueError('Both layers must use the same complete 30-second piano input and engine')
        if any(r.get('reference_audio',{}).get('sha256') != reference_hash for r in (a,b)):
            raise ValueError('Both layers must use the same style-reference bytes')
        if sha256(source) != report['source_sha256'] or sha256(frozen) != reference_hash:
            raise ValueError('Source or common style reference changed')
        report['input_sha256'] = a['input']['source_sha256']
        if not execute:
            report['status'] = 'planned'
            return report
        if any(r['status'] != 'generated' for r in (a,b)):
            raise ValueError('Both guitar layers must finish')
        if a['settings']['config'] != b['settings']['config']:
            raise ValueError('Layer generation configs differ')
        pa, pb = a['settings']['params'], b['settings']['params']
        differences = sorted(k for k in pa.keys() | pb.keys() if pa.get(k) != pb.get(k))
        if 'caption' not in differences or set(differences) - {'caption','src_audio','reference_audio'}:
            raise ValueError('Unexpected actual layer setting differences')
        frames = a['input']['valid_frames']
        strum, pick = [read(output / name / 'guitar.wav', frames) for name in ('strum','pick')]
        piano = read(output / 'strum/prepared/original.wav', frames)
        gain = export_mix(output, strum, pick, piano)
        report.update(status='generated', musical_quality='unreviewed', export_frames=frames,
                      actual_parameter_differences=differences, global_headroom_gain=gain,
                      strum_stem_sha256=a['stem_sha256'], pick_stem_sha256=b['stem_sha256'],
                      layered_wav_sha256=sha256(output / 'guitars_layered.wav'))
        write_json(output / 'layers.json', report)
        for filename, full in [('guitar_layers_feedback.zip',False),('guitar_layers_result.zip',True)]:
            with zipfile.ZipFile(output / filename, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(output / 'layers.json','layers.json')
                for name in ('guitars_layered','piano_guitars_mix'):
                    archive.write(output / (name + '.mp3'),name + '.mp3')
                    if full: archive.write(output / (name + '.wav'),name + '.wav')
                for role in ('strum','pick'):
                    for name in ['guitar_probe.json','guitar_preview.mp3','mix_preview.mp3'] + (['guitar.wav'] if full else []):
                        archive.write(output / role / name, role + '/' + name)
                if full: archive.write(frozen,frozen.name)
        return report
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'layers.json', report)


def generate_full(root, source, output, reference_audio, *, execute=False):
    """Two 90s-context guitar plans; resume verified chunks without repeating work."""
    import ace_context_chunks as chunks
    root, source, output, reference_audio = [Path(p).resolve() for p in (root,source,output,reference_audio)]
    expected = {'source_sha256':chunks.sha256(source), 'reference_audio_sha256':chunks.sha256(reference_audio)}
    if not output.exists():
        output.mkdir(parents=True)
        frozen = output / ('style_reference' + reference_audio.suffix.lower())
        shutil.copyfile(reference_audio, frozen)
        if chunks.sha256(frozen) != expected['reference_audio_sha256']: raise ValueError('Full-song reference changed during copy')
        report = dict(expected, status='preparing', reference_file=frozen.name, mode='full_song_context90',
                      layer_weights={'strum':.45,'pick':.45}, musical_quality='unreviewed')
        chunks.write_json(output/'layers_full.json', report)
        for role,variant in [('strum','strum-layer'),('pick','pick-layer')]:
            chunks.prepare(source,output/role,context=90,margin=6,fade=.58,
                           instrument='guitar',guitar_variant=variant,reference_audio=frozen)
        report['status']='planned';chunks.write_json(output/'layers_full.json',report)
    report=json.loads((output/'layers_full.json').read_text())
    if any(report.get(k)!=v for k,v in expected.items()) or report.get('status')=='preparing':
        raise ValueError('Full-song input/reference differs or preparation is incomplete; preserve the run')
    frozen=output/report['reference_file']
    if not frozen.resolve().is_relative_to(output) or chunks.sha256(frozen)!=expected['reference_audio_sha256']:
        raise ValueError('Frozen full-song style reference changed')
    plans={}
    for role,variant in [('strum','strum-layer'),('pick','pick-layer')]:
        _,plan,_=chunks.load_plan(output/role/'plan.json')
        if plan.get('guitar_variant')!=variant or plan['original_sha256']!=expected['source_sha256'] or plan.get('shared_reference',{}).get('sha256')!=expected['reference_audio_sha256'] or (plan['context_cap_seconds'],plan['margin_seconds'],plan['fade_seconds'])!=(90,6,.58):
            raise ValueError('Full-song role plan differs')
        plans[role]=plan
    if plans['strum']['chunks']!=plans['pick']['chunks']:
        raise ValueError('Guitar layer timelines must match')
    report['model_calls_total_if_uncached']=sum(len(p['chunks']) for p in plans.values())
    report['export_frames']=plans['strum']['total_frames']
    chunks.write_json(output/'layers_full.json',report)
    if not execute:return report
    if report.get('status')=='generated':
        mixed=(output/report['mix_export_folder']).resolve()
        if not mixed.is_relative_to(output) or chunks.sha256(mixed/'guitars_layered.wav')!=report['layered_wav_sha256'] or chunks.sha256(mixed/'piano_guitars_mix.wav')!=report['mix_wav_sha256']:
            raise ValueError('Completed layered export changed')
        package_full(output,report)
        return report
    for role in ('strum','pick'):
        chunks.run_chunks(root,output/role/'plan.json')
        done=output/role/'stitched/seams.json'
        if done.is_file():
            seam=json.loads(done.read_text())
            if chunks.sha256(output/role/'stitched/guitar_full.wav')!=seam['stem_sha256']:
                raise ValueError('Completed role export changed')
        else:chunks.stitch_run(output/role/'plan.json')
    frames=report['export_frames']
    strum,pick=[read(output/role/'stitched/guitar_full.wav',frames) for role in ('strum','pick')]
    piano=.7*read(output/'strum/source.wav',frames)
    mixed=Path(tempfile.mkdtemp(prefix='mix_attempt_',dir=output))
    gain=export_mix(mixed,strum,pick,piano)
    report.update(status='generated',global_headroom_gain=gain,
                  mix_export_folder=mixed.name,
                  layered_wav_sha256=chunks.sha256(mixed/'guitars_layered.wav'),
                  mix_wav_sha256=chunks.sha256(mixed/'piano_guitars_mix.wav'))
    chunks.write_json(output/'layers_full.json',report)
    package_full(output,report)
    return report


def package_full(output, report):
    mixed=output/report['mix_export_folder']
    for filename,full in [('guitar_layers_full_feedback.zip',False),('guitar_layers_full_result.zip',True)]:
        with zipfile.ZipFile(output/filename,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.write(output/'layers_full.json','layers_full.json')
            for name in ('guitars_layered','piano_guitars_mix'):
                z.write(mixed/(name+'.mp3'),name+'.mp3')
                if full:z.write(mixed/(name+'.wav'),name+'.wav')
            for role in ('strum','pick'):
                for name in ('plan.json','progress.json'):z.write(output/role/name,role+'/'+name)
                for p in sorted((output/role/'stitched').rglob('*')):
                    if p.is_file() and (p.suffix in ('.json','.mp3') or p.name.startswith('seam_') or full and p.name=='guitar_full.wav'):
                        z.write(p,role+'/stitched/'+str(p.relative_to(output/role/'stitched')))
            if full:z.write(output/report['reference_file'],report['reference_file'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference-audio',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--full-song',action='store_true')
    args = parser.parse_args()
    if args.full_song:
        result=generate_full(args.root,args.source,args.out,args.reference_audio,execute=args.execute)
    else:result=generate(args.root,args.source,args.out,args.reference_audio,start=args.start,execute=args.execute)
    print(json.dumps(result,indent=2))
