"""One clean-electric-guitar context (10–90s) on a pinned ACE-Step CUDA runtime."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import zipfile

import electric_full_song as single

PRESET = {
    'id': 'clean-electric-guitar-v1', 'model': 'acestep-v15-base',
    'device': 'cuda', 'precision': 'float32', 'instrument': 'guitar',
    'seeds': [0], 'guidance_scale': 7.0, 'inference_steps': 50,
    'offload_to_cpu': True,
    'caption': 'Isolated clean electric guitar accompaniment. A warm, clear, undistorted electric guitar played naturally with gentle picked chord voicings and sparse rhythmic strums. Follow the piano performance chords, timing, phrasing and dynamics, leaving space for the piano melody. Subtle pick attack, natural note decay and expressive articulation. Electric guitar only, no bass guitar, drums, piano or vocals.',
}


def load_engine(root):
    single.attach(root)
    # Guitar notebooks carry the current engine for older cached source bridges.
    # Load it only in this process; runtime source, weights and old runs stay intact.
    path = Path(__file__).with_name('guitar_run.py')
    if not path.is_file():
        path = Path(root) / 'tools/stem_eval/run.py'
    spec = importlib.util.spec_from_file_location('guitar_probe_engine', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def package(output, success):
    reports = ['guitar_probe.json', 'prepared/input.json', 'generation/experiment.json',
               'generation/session.json', 'generation/000_A_guitar_seed0/result.json',
               'capture/capture.json']
    for name, audio in [('guitar_feedback.zip', ['guitar_preview.mp3', 'mix_preview.mp3']),
                        ('guitar_result.zip', ['guitar.wav', 'mix.wav'])]:
        if name == 'guitar_result.zip' and not success:
            continue
        with zipfile.ZipFile(output / name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in reports + audio:
                if (output / relative).is_file():
                    archive.write(output / relative, relative)


def generate(root, source, output, *, start=0, duration=30, execute=False):
    if not math.isfinite(duration) or not 10 <= duration <= 90:
        raise ValueError('Guitar contexts must be 10–90 seconds; default probe is 30')
    root, source, output = [Path(p).resolve() for p in (root, source, output)]
    engine = load_engine(root)
    from provenance import sha256, write_json
    from runtime_environment import runtime_environment
    from stream_process import run_logged
    from compare_bass_tone import require_fp32
    config = root / 'configs/clean-electric-guitar.json'
    if config.is_file() and json.loads(config.read_text()) != PRESET:
        raise ValueError('Guitar preset differs; use a new preset/version')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'instrument': 'guitar', 'preset': PRESET,
              'mode': 'single_pass_probe', 'model_calls_requested': 1,
              'engine_sha256': sha256(Path(engine.__file__)),
              'note': 'Original piano conditioning only. No bass/drum regeneration. Guitar sound, chord following and timing need listener review; no iPhone/CoreAI claim.'}
    write_json(output / 'guitar_probe.json', report)
    try:
        metadata = single.prepare(source, output / 'prepared', start=start, duration=duration)
        report.update(status='planned', input=metadata)
        write_json(output / 'guitar_probe.json', report)
        if not execute:
            (output / 'generation').mkdir()
            experiment = engine.plan(output / 'prepared', 'ace', 'guitar', [0])
            experiment.update(max_job_seconds=90, device_requested='cuda',
                              ace_model=PRESET['model'], offload_to_cpu=True,
                              guidance_scale=7, mode='single_pass_probe')
            write_json(output / 'generation/experiment.json', experiment)
            return report
        command = [str(single.model_python(root)), str(Path(__file__).resolve()), '--worker',
                   '--root', str(root), '--out', str(output)]
        run_logged(command, env=runtime_environment(root), check=True)
        experiment = json.loads((output / 'generation/experiment.json').read_text())
        if experiment.get('status') != 'generated' or len(experiment['jobs']) != 1:
            raise ValueError('Expected one completed guitar job')
        job = experiment['jobs'][0]
        folder = output / 'generation/000_A_guitar_seed0'
        result = engine.cached_job(folder, job, None)
        if result is None or result.get('device') != 'cuda' or job.get('instrument') != 'guitar' or job['source_sha256'] != metadata['source_sha256']:
            raise ValueError('Guitar device/input/result verification failed')
        params, config = result['settings']['params'], result['settings']['config']
        expected = {'caption': PRESET['caption'], 'seed': 0, 'guidance_scale': 7,
                    'inference_steps': 50, 'duration': metadata['model_duration'],
                    'task_type': 'lego', 'instrumental': True,
                    'instruction': 'Generate the guitar track based on the audio context:',
                    'reference_audio': None, 'enable_normalization': False,
                    'thinking': False, 'use_cot_caption': False,
                    'use_cot_metas': False, 'use_cot_language': False}
        if any(params.get(k) != v for k, v in expected.items()):
            raise ValueError('Actual guitar settings differ')
        if config.get('seeds') != [0] or config.get('batch_size') != 1 or config.get('use_random_seed') is not False:
            raise ValueError('Unexpected guitar seed/batch settings')
        capture = json.loads((output / 'capture/capture.json').read_text())
        require_fp32(capture)
        if sha256(source) != metadata['original_sha256'] or sha256(output / 'prepared/source.wav') != metadata['source_sha256']:
            raise ValueError('Input changed during guitar generation')
        single.export_audio(folder / result['stem_file'], output / 'guitar.wav', metadata['valid_frames'])
        single.export_audio(folder / 'mix.wav', output / 'mix.wav', metadata['valid_frames'])
        for src, dst, gain in [('guitar.wav', 'guitar_preview.mp3', .7), ('mix.wav', 'mix_preview.mp3', 1)]:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(output / src),
                            '-af', f'volume={gain}', '-c:a', 'libmp3lame', '-b:a', '192k', str(output / dst)], check=True)
        report.update(status='generated', settings=result['settings'], decoder_capture=capture,
                      stem_sha256=sha256(output / 'guitar.wav'), mix_sha256=sha256(output / 'mix.wav'),
                      inference_seconds=result.get('inference_seconds'),
                      peak_cuda_allocated_bytes=result.get('peak_cuda_allocated_bytes'))
        write_json(output / 'guitar_probe.json', report)
        package(output, True)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'guitar_probe.json', report)
        try:
            package(output, False)
        except Exception:
            pass
        raise
    finally:
        write_json(output / 'guitar_probe.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        single.execute_worker(args.root.resolve(), args.out.resolve(), preset=PRESET,
                              instrument='guitar', engine=load_engine(args.root.resolve()))
    else:
        if args.source is None:
            parser.error('--source is required')
        print(json.dumps(generate(args.root, args.source, args.out, start=args.start,
                                  duration=args.duration, execute=args.execute), indent=2))
