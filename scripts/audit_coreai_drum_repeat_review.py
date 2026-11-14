#!/usr/bin/env python3
"""Audit a same-input native drum repeat ZIP; repeatability is not music/host parity."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

from package_coreai_full_song_quality_review import floats,is_sha,comparison,LIMIT


def audit(path,candidate=False):
    generated='candidate' if candidate else 'repeat'
    if Path(path).stat().st_size>LIMIT:raise ValueError('Review exceeds32MiB')
    with zipfile.ZipFile(path) as z:
        items=[i for i in z.infolist() if not i.is_dir()];names=[i.filename for i in items]
        if len(items)>100 or sum(i.file_size for i in items)>64*1024*1024 or len(names)!=len(set(names)):
            raise ValueError('Oversized/duplicated payload')
        if any(n.startswith('/') or '..' in Path(n).parts for n in names):raise ValueError('Unsafe archive path')
        manifests=[n for n in names if n.endswith('review-manifest.json') and not n.startswith('__MACOSX/')]
        if len(manifests)!=1:raise ValueError('Select one review manifest')
        if z.getinfo(manifests[0]).file_size>1024*1024:raise ValueError('Oversized manifest')
        prefix=manifests[0][:-len('review-manifest.json')];m=json.loads(z.read(manifests[0]));data={}
        expected={'schema_version':1,'profile':'full-song-one-window-repeatability-v1','instrument':'drums','new_generation_windows':1,
            'new_inference_calls':756,'native_exports':0,'baseline_modified':False,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,
            'relative_rmse_tolerance':1e-3,'sampler_weight_storage_dtype':'bfloat16','compute_dtype':'float32','vae_dtype':'float32',
            'inference_steps':50,'guidance_scale':7,'apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
            'music_acceptance':'NOT EVALUATED BY TEST'}
        if candidate:
            expected.update(profile='full-song-guidance-candidate-v1',guidance_scale=4,baseline_guidance_scale=7,candidate_euler_control_checks=50,
                            candidate_update='Float32 c+0.5*(native_guided7-c); Float32 Euler; native momentum/cap/projection unchanged')
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Repeat scope/settings/bounds differ')
        if not is_sha(m.get('execution_sha256')) or not is_sha(m.get('job_reference_sha256')):raise ValueError('Invalid source identity')
        for name,entry in m['artifacts'].items():
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Invalid artifact path')
            raw=z.read(prefix+name)
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Artifact SHA/size differs')
            data[name]=raw
        actual={n[len(prefix):] for n in names if n.startswith(prefix) and not n.startswith('__MACOSX/')}
        if actual!=set(data)|{'review-manifest.json'}:raise ValueError('Unexpected payload')
    shapes={'input/context.f32le':[1,750,128],'input/noise.f32le':[1,750,64],'input/condition.f32le':[1,127,2048],
            'input/timesteps.f32le':[51],generated+'/final-momentum.f32le':[1,750,64]}
    if 'input/piano-focus.f32le' in data:shapes['input/piano-focus.f32le']=[1,2,m['focus']['frames']]
    if candidate and 'input/piano-focus.f32le' not in data:raise ValueError('Candidate piano comparison missing')
    for label in ('captured',generated):
        shapes[label+'/final-latent.f32le']=[1,750,64];shapes[label+'/raw-drums.f32le']=[1,2,1440000]
    arrays={}
    for name,shape in shapes.items():
        e=m['artifacts'][name]
        if e.get('shape')!=shape or e.get('dtype')!='float32_le' or len(data[name])!=math.prod(shape)*4:raise ValueError('Tensor shape/type differs')
        arrays[name]=floats(data[name])
    for name in ('context','noise','condition','timesteps'):
        if m['artifacts']['input/'+name+'.f32le']['sha256']!=m['used_input_sha256'][name]:raise ValueError('Used input hash differs')
    times=arrays['input/timesteps.f32le']
    if times[0]!=1 or times[-1]!=0 or any(a<=b for a,b in zip(times,times[1:])):raise ValueError('Timesteps differ')
    for name,key in [('captured/raw-drums.f32le','original_captured_raw_sha256'),('captured/final-latent.f32le','original_captured_latent_sha256'),
                     ('captured/report.txt','original_captured_report_sha256')]:
        if m['artifacts'][name]['sha256']!=m[key]:raise ValueError('Captured source identity differs')
    for label in ('captured',generated):
        report=data[label+'/report.txt'].decode()
        if ('LIVE final latent SHA256: '+m['artifacts'][label+'/final-latent.f32le']['sha256']) not in report.splitlines():raise ValueError('Latent/report binding differs')
        if candidate and label==generated and 'Candidate FP32 Euler control: PASS50/50' not in report:raise ValueError('Candidate arithmetic control missing')
    focus=m['focus'];start,n=focus['local_start_sample'],focus['frames']
    if (type(start) is not int or type(n) is not int or not 0<=start<start+n<=1440000 or n>96000
            or m['latent_end']-m['latent_start']!=750 or focus['global_start_sample']!=m['latent_start']*1920+start):raise ValueError('Focus timeline differs')
    gain=struct.unpack('<f',struct.pack('<f',focus['listening_gain']))[0]
    if not math.isfinite(gain) or not 0<gain<=1:raise ValueError('Invalid listening gain')
    checks={}
    for name in ('final-latent','raw-drums'):
        checks[name]=comparison(arrays[generated+'/'+name+'.f32le'],arrays['captured/'+name+'.f32le'])
    def focus_wave(name):
        wav=data[name]
        header=(b'RIFF'+struct.pack('<I',50+n*8)+b'WAVEfmt '+struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0)
                +b'fact'+struct.pack('<II',4,n)+b'data'+struct.pack('<I',n*8))
        if wav[:58]!=header or len(wav)!=58+n*8:raise ValueError('Focus WAV format differs')
        value=floats(wav[58:]);planar=value[::2];planar.extend(value[1::2]);return planar
    for label in ('captured',generated):
        raw=arrays[label+'/raw-drums.f32le'];selected=raw[start:start+n];selected.extend(raw[1440000+start:1440000+start+n])
        scaled=array.array('f',(v*gain for v in selected))
        checks[label+'_focus_wav_binding']={'passed':scaled==focus_wave(label+'/focus.wav')}
        if 'input/piano-focus.f32le' in arrays:
            combined=array.array('f',(p+d for p,d in zip(arrays['input/piano-focus.f32le'],selected)))
            halved=array.array('f',(v*.5 for v in combined));mixed=array.array('f',(v*gain for v in halved))
            checks[label+'_piano_mix_binding']={'passed':mixed==focus_wave(label+'/piano-with-drums-focus.wav')}
    repeated=checks['final-latent']['passed'] and checks['raw-drums']['passed']
    if candidate:
        if 'repeatability_passed' in m or m.get('candidate_vs_baseline_within_bounds') is not repeated:raise ValueError('Candidate difference scope differs')
        baseline_gain=struct.unpack('<f',struct.pack('<f',m.get('baseline_listening_gain',0)))[0]
        if not math.isfinite(baseline_gain) or not 0<gain<=baseline_gain<=1:raise ValueError('Candidate shared gain differs')
        return {'profile':'full-song-guidance-candidate-audit','checks':checks,'candidate_vs_baseline_within_bounds':repeated,
                'focus_wav_bindings_passed':all(checks[label+key]['passed'] for label in ('captured',generated) for key in ('_focus_wav_binding','_piano_mix_binding')),
                'scope':'Candidate output differences are expected; neither within-bounds nor outside-bounds means musical success/failure',
                'music_acceptance':'NOT EVALUATED BY AUDIT','baseline_modified':False}
    if m.get('repeatability_passed') is not repeated:raise ValueError('Claimed repeatability differs from original bounds')
    return {'profile':'full-song-one-window-repeat-audit','checks':checks,'repeatability_passed':repeated,
            'focus_wav_bindings_passed':all(checks[label+'_focus_wav_binding']['passed'] for label in ('captured','repeat')),
            'scope':'Repeat vs captured output only; this does not establish host trajectory parity, musical acceptance or a cause of run differences',
            'music_acceptance':'NOT EVALUATED BY AUDIT','baseline_modified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--candidate',action='store_true',help='Explicitly audit separate guidance4 candidate; never interpret it as repeatability')
    args=p.parse_args()
    try:print(json.dumps(audit(args.archive,args.candidate),indent=2))
    except (ValueError,OSError,KeyError,TypeError,zipfile.BadZipFile) as error:p.exit(1,'Audit stopped: '+str(error)+'\n')


if __name__=='__main__':main()
