#!/usr/bin/env python3
"""Replay only the VAE on captured native bass/guitar latent; never regenerate stems.

Default is plan-only. Full750 CPU decoding is a host diagnostic, not an iPhone
memory strategy. --include-window-control also replays the six original tiles.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import uuid
import zipfile

from export_coreai_decoder import ROOT, sha256
from package_coreai_instrument_transition_review import ROLES, FRAMES, MAX_ZIP, shapes, report_metadata


def read_review(path):
    import numpy as np
    path = Path(path)
    if path.stat().st_size > MAX_ZIP: raise ValueError('Review exceeds32MiB')
    with zipfile.ZipFile(path) as archive:
        names = {'review-manifest.json'} | {role+'/'+name for role in ROLES
                    for name in [*shapes(role), 'report.txt', 'last8s-listening.wav']}
        if len(archive.infolist()) != len(names) or set(archive.namelist()) != names:
            raise ValueError('Unexpected, missing or duplicated review entries')
        if archive.getinfo('review-manifest.json').file_size > 1024*1024:
            raise ValueError('Manifest exceeds1MiB')
        manifest = json.loads(archive.read('review-manifest.json'))
        expected = {'schema_version':1, 'profile':'private-instrument30-transition-review',
                    'absolute_tolerance':1e-4, 'relative_tolerance':1e-3, 'relative_rmse_tolerance':1e-3,
                    'new_inference_calls':0, 'native_exports':0, 'baseline_modified':False}
        if (any(manifest.get(k) != value for k,value in expected.items())
                or set(manifest.get('runs',{})) != set(ROLES)
                or set(manifest.get('artifacts',{})) != {role+'/'+n for role in ROLES for n in shapes(role)}):
            raise ValueError('Review contract differs; original bounds unchanged')
        result = {}
        for role in ROLES:
            info = manifest['runs'][role]
            run_id = str(uuid.UUID(info['run_id'])).upper()
            if archive.getinfo(role+'/report.txt').file_size > 1024*1024: raise ValueError('Report exceeds1MiB')
            raw_report = archive.read(role+'/report.txt')
            parsed = report_metadata(raw_report.decode(),role,run_id)
            if (any(info.get(k) != value for k,value in parsed.items())
                    or hashlib.sha256(raw_report).hexdigest() != info.get('report_sha256')):
                raise ValueError('Device report/manifest mismatch: '+role)
            arrays = {}
            for name,shape in shapes(role).items():
                key = role+'/'+name
                entry = manifest['artifacts'][key]
                if (entry.get('shape') != shape or entry.get('bytes') != math.prod(shape)*4
                        or entry.get('dtype') != 'float32_le' or archive.getinfo(key).file_size != entry['bytes']):
                    raise ValueError('Unexpected native tensor shape/size: '+key)
                raw = archive.read(key)
                if hashlib.sha256(raw).hexdigest() != entry.get('sha256'): raise ValueError('Native tensor SHA256 mismatch: '+key)
                value = np.frombuffer(raw,dtype='<f4').copy().reshape(shape)
                if not np.isfinite(value).all(): raise ValueError('Nonfinite native tensor: '+key)
                arrays[name] = value
            if (manifest['artifacts'][role+'/final-latent.f32le']['sha256'] != info['final_latent_sha256']
                    or not np.array_equal(arrays['final-latent.f32le'],arrays['native-step-50-latent.f32le'])):
                raise ValueError('Final latent/checkpoint50/report mismatch')
            if archive.getinfo(role+'/last8s-listening.wav').file_size != 58+8*48000*8:
                raise ValueError('Unexpected last8s WAV size')
            tail = archive.read(role+'/last8s-listening.wav')
            if hashlib.sha256(tail).hexdigest() != info.get('last8s_wav_sha256'):
                raise ValueError('Last8s WAV checksum mismatch')
            result[role] = {'latent':arrays['final-latent.f32le'], 'audio':arrays['raw-'+role+'.f32le'],
                            'tail_wav':tail, 'metadata':info}
    return result, manifest


def waveform_checks(actual, reference):
    import numpy as np
    import torch
    from export_coreai_dit_block import precision_metrics
    from coreai_drum_reference30_window_contract import tile_plan
    if list(actual.shape) != [1,2,FRAMES] or list(reference.shape) != [1,2,FRAMES]:
        raise ValueError('Expected full30s stereo waveforms')
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite waveform comparison')
    regions = {'overall':(0,FRAMES), 'last8s':(22*48000,FRAMES)}
    for i,tile in enumerate(tile_plan(),1): regions['core-'+str(i)] = (tile['core_start']*1920,tile['core_end']*1920)
    for i,tile in enumerate(tile_plan()[1:],1):
        center = tile['core_start']*1920
        regions['seam-'+str(i)] = (center-3840,center+3840)
    checks = {}
    for name,(a,b) in regions.items():
        left, right = actual[...,a:b], reference[...,a:b]
        metric = precision_metrics(left,right)
        if metric['finite']:
            error = np.abs(left.numpy().astype(np.float64)-right.numpy().astype(np.float64))
            bound = 1e-4+1e-3*np.abs(right.numpy().astype(np.float64))
            metric.update(pointwise_violations=int((error>bound).sum()), values=int(error.size), worst_bound_ratio=float((error/bound).max()))
        checks[name] = metric
    return {'strict_verdict':'PASS' if all(m['parity_passed'] for m in checks.values()) else 'NOT PASSED',
            'comparisons_passed':sum(m['parity_passed'] for m in checks.values()), 'comparisons_target':len(regions), 'checks':checks}


def write_listening_page(out, include_windows):
    cards = []
    for role,label in [('bass','Bass'),('guitar','Gitar')]:
        rows = [('01-native-last8s.wav','Dari iPhone'),('03-cpu-full-last8s.wav','Decode CPU penuh')]
        if include_windows: rows.insert(1,('02-cpu-window-last8s.wav','Decode CPU berjendela'))
        clips = ''.join(f'<div><p>{title}</p><audio controls preload="metadata" src="{role}/{name}"></audio><output>Posisi lagu: 22,0 detik</output></div>' for name,title in rows)
        cards.append(f'<section><h2>{label}</h2>{clips}</section>')
    html = ('<!doctype html><meta charset="utf-8"><title>Perbandingan transisi bass dan gitar</title>'
            '<style>body{font:17px system-ui;max-width:820px;margin:32px auto;padding:0 20px;background:#f5f5f5;color:#222}'
            'section{background:white;padding:20px;margin:20px 0;border-radius:12px}audio{width:100%}output{display:block;margin:4px 0 20px;color:#555}</style>'
            '<h1>Perbandingan bagian akhir</h1><p>Cuplikan detik 22–30 dari hasil yang sama. Putar dengan volume yang sama dan perhatikan perpindahan akor.</p>'
            + ''.join(cards) + '<p><a href="report.json">Laporan pemeriksaan</a></p>'
            '<script>document.querySelectorAll("audio").forEach(p=>{const o=p.nextElementSibling;'
            'const update=()=>o.textContent="Posisi lagu: "+(22+p.currentTime).toFixed(1)+" detik";'
            'p.addEventListener("timeupdate",update);p.addEventListener("seeked",update);'
            'p.addEventListener("play",()=>document.querySelectorAll("audio").forEach(q=>{if(q!==p)q.pause()}));});</script>')
    (out/'listen.html').write_text(html)


def execute(args):
    import torch
    from compare_coreai_drum_lyrics import local_checkpoint
    from coreai_drum_reference30_window_contract import VAE_SHA, tile_plan, dependency_bounds
    from prepare_coreai_drum_reference30_window_decoder import load_decoder, verify_architecture
    from coreai_listening_audio import write_float_wav
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError('Existing decode review preserved: '+str(args.out))
    if any(args.out.resolve().is_relative_to(ROOT/p) for p in ('baselines','prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Use an inactive output outside baselines/models')
    if args.archive.resolve().is_relative_to(args.out.resolve()): raise ValueError('Do not write inside the review input folder')
    archive_sha256 = sha256(args.archive)
    received, manifest = read_review(args.archive)
    if sha256(args.archive) != archive_sha256: raise ValueError('Review archive changed while reading')
    record = next(r for r in json.loads((ROOT/'configs/assets-base.json').read_text())['files'] if r['filename']=='vae/diffusion_pytorch_model.safetensors')
    if record['digest'] != VAE_SHA: raise ValueError('Pinned VAE identity changed')
    checkpoint = local_checkpoint(record,args.vae_checkpoint)
    args.out.mkdir(parents=True)
    report = {'schema_version':1, 'profile':'instrument30-same-latent-decoder-audit', 'status':'verifying_audio',
              'archive_sha256':archive_sha256, 'vae_sha256':VAE_SHA,
              'absolute_tolerance':1e-4, 'relative_tolerance':1e-3, 'relative_rmse_tolerance':1e-3,
              'cpu_compute_dtype':'float32', 'sampler_weight_storage_dtype':'bfloat16',
              'baseline_modified':False, 'sampler_calls':0, 'Qwen_calls':0, 'source_encoder_calls':0,
              'native_exports':0, 'cpu_decoder_calls':0, 'include_window_control':args.include_window_control,
              'full_trajectory_parity':'NOT EVALUATED; original drum gate unchanged', 'musical_acceptance':'NOT EVALUATED; listening required',
              'scope':'Same actual native final latent only; host full750 is a diagnostic, not a proposed iPhone memory strategy',
              'roles':{}, 'torch':torch.__version__, 'artifact_sha256':{}}
    def save():
        temporary = args.out/'report.json.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(args.out/'report.json')
    save()
    try:
        for role,value in received.items():
            folder = args.out/role
            folder.mkdir()
            info = value['metadata']
            # Reconstruct original full WAV and verify before any model is loaded.
            with tempfile.TemporaryDirectory(dir=args.out) as temporary:
                proof = Path(temporary)/'proof.wav'
                write_float_wav(proof,value['audio'].reshape(-1),FRAMES,info['listening_gain_float32'])
                if sha256(proof) != info['solo_wav_sha256']: raise ValueError('Raw/WAV/report binding mismatch: '+role)
            tail = folder/'01-native-last8s.wav'
            write_float_wav(tail,value['audio'][...,22*48000:].reshape(-1),8*48000,info['listening_gain_float32'])
            if sha256(tail) != info['last8s_wav_sha256']: raise ValueError('Native tail differs from captured bytes')
            report['roles'][role] = {'run_id':info['run_id'], 'device_report_sha256':info['report_sha256'],
                                    'native_final_latent_sha256':info['final_latent_sha256'],
                                    'condition_sha256':info['condition_sha256'], 'listening_gain_float32':info['listening_gain_float32'],
                                    'input_integrity':'VERIFIED', 'comparisons':{}}
        torch.set_num_threads(2)
        decoder = load_decoder(checkpoint)
        verify_architecture(decoder)
        for role,value in received.items():
            latent = torch.from_numpy(value['latent']).transpose(1,2).contiguous()
            native = torch.from_numpy(value['audio'])
            assembled = None
            with torch.inference_mode():
                if args.include_window_control:
                    assembled = torch.empty(1,2,FRAMES)
                    coverage = [0]*750
                    for i,tile in enumerate(tile_plan()):
                        dependency_bounds(tile)
                        decoded = decoder(latent[...,tile['window_start']:tile['window_start']+256].contiguous())
                        report['cpu_decoder_calls'] += 1
                        if list(decoded.shape) != [1,2,491520] or not torch.isfinite(decoded).all(): raise ValueError('Invalid CPU window output')
                        a,b,c = tile['core_start']*1920,tile['core_end']*1920,tile['crop_start']*1920
                        assembled[...,a:b] = decoded[...,c:c+b-a]
                        for frame in range(tile['core_start'],tile['core_end']): coverage[frame] += 1
                        del decoded
                        print(f'{role}: CPU window{i+1}/6',flush=True)
                    if set(coverage) != {1}: raise ValueError('Every output frame must be covered exactly once')
                print(role+': decode full750 once; no sampler run',flush=True)
                begin = time.monotonic()
                full = decoder(latent)
                report['cpu_decoder_calls'] += 1
                if list(full.shape) != [1,2,FRAMES] or not torch.isfinite(full).all(): raise ValueError('Invalid CPU full output')
                decode_seconds = time.monotonic()-begin
            comparisons = {'native_vs_cpu_full':waveform_checks(native,full)}
            if assembled is not None:
                comparisons.update(native_vs_cpu_window=waveform_checks(native,assembled), cpu_window_vs_cpu_full=waveform_checks(assembled,full))
            report['roles'][role].update(comparisons=comparisons, full_decode_seconds=decode_seconds)
            outputs = [('03-cpu-full-last8s.wav',full)]
            if assembled is not None: outputs.append(('02-cpu-window-last8s.wav',assembled))
            for name,audio in outputs:
                write_float_wav(args.out/role/name,audio[...,22*48000:].reshape(-1).numpy(),8*48000,value['metadata']['listening_gain_float32'])
            for label,result in comparisons.items(): print(role,label,result['strict_verdict'],result['comparisons_passed'],'/',result['comparisons_target'],flush=True)
            del full, assembled
            save()
        write_listening_page(args.out,args.include_window_control)
        report['artifact_sha256'] = {str(p.relative_to(args.out)):sha256(p) for p in args.out.rglob('*') if p.is_file() and p.name!='report.json'}
        report['status'] = 'completed'
        save()
        print('Listening page:',args.out/'listen.html',flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--archive',type=Path,default=ROOT/'.runtime-cache/bass-guitar-transition-review-v1.zip')
    parser.add_argument('--vae-checkpoint',type=Path)
    parser.add_argument('--include-window-control',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'.runtime-cache/instrument-transition-decode-v1')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: verify saved native bass/guitar outputs; CPU FP32 VAE full750 on the same latent, optional six-window control; no sampler, export, download or activation.')
        return
    execute(args)


if __name__ == '__main__': main()
