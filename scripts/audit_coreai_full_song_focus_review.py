#!/usr/bin/env python3
"""Verify a small iPhone focus ZIP and selected raw/WAV binding; no inference."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

from package_coreai_full_song_quality_review import floats,is_sha,comparison,FRAMES,LIMIT


def audit(path):
    path=Path(path)
    if path.stat().st_size>LIMIT:raise ValueError('Focus ZIP exceeds32MiB')
    with zipfile.ZipFile(path) as archive:
        files=[i for i in archive.infolist() if not i.is_dir()]
        if len(files)>300 or sum(i.file_size for i in files)>64*1024*1024:raise ValueError('Oversized focus ZIP payload')
        names=[i.filename for i in files]
        if len(names)!=len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names):raise ValueError('Unsafe/duplicate archive paths')
        manifests=[n for n in names if n.endswith('review-manifest.json') and not n.startswith('__MACOSX/')]
        if len(manifests)!=1:raise ValueError('Select one focus review manifest')
        name=manifests[0];prefix=name[:-len('review-manifest.json')]
        manifest_bytes=archive.read(name)
        if len(manifest_bytes)>1024*1024:raise ValueError('Manifest exceeds1MiB')
        m=json.loads(manifest_bytes)
        expected={'schema_version':1,'profile':'private-full-song-quality-review-v1','mode':'device-focus',
                  'new_inference_calls':0,'native_exports':0,'baseline_modified':False,'absolute_tolerance':1e-4,
                  'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,'full_song_numerical_parity':'NOT EVALUATED',
                  'music_acceptance':'NOT EVALUATED BY EXPORT'}
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Focus scope/baseline/bounds differ')
        execution,job=m.get('execution_sha256'),m.get('job_reference_sha256')
        if not is_sha(execution) or not is_sha(job):raise ValueError('Invalid execution/job identity')
        payload={}
        for n,entry in m['artifacts'].items():
            if n.startswith('/') or '..' in Path(n).parts:raise ValueError('Invalid artifact path')
            raw=archive.read(prefix+n)
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Artifact size/SHA mismatch: '+n)
            payload[n]=raw
        # CRC is checked by ZipFile.read, including a Foundation wrapper directory.
        actual={n[len(prefix):] for n in names if n.startswith(prefix) and not n.startswith('__MACOSX/')}
        if actual!=set(payload)|{'review-manifest.json'}:raise ValueError('Unexpected archive payload')
    done=json.loads(payload['completed.json']);focus=m['focus'];a,n=focus['start_sample'],focus['frames']
    if (done.get('schema_version')!=1 or done.get('execution_sha256')!=execution or done.get('job_reference_sha256')!=job
            or done.get('completed_windows')!=m['completed_windows'] or done.get('target_samples48')!=m['completed_receipt']['target_samples48']
            or not all(is_sha(v) for v in done.get('artifact_sha256',{}).values())
            or type(a) is not int or type(n) is not int or not 0<=a<a+n<=done['target_samples48'] or n>2*48000):raise ValueError('Completed receipt/focus geometry differs')
    gain=struct.unpack('<f',struct.pack('<f',done['listening_gain']))[0]
    if not math.isfinite(gain) or not 0<gain<=1:raise ValueError('Invalid Float32 listening gain')
    windows=m['schedule'];indices=[w['index'] for w in windows if max(a,w['sample_start48'])<min(a+n,w['sample_end48'])]
    if (indices!=focus['window_indices'] or not indices or any(w['index']!=i for i,w in enumerate(windows))
            or any(not 0<=w['sample_start48']<w['sample_end48']<=done['target_samples48'] or w['sample_end48']-w['sample_start48']>FRAMES for w in windows)):
        raise ValueError('Focus schedule differs')
    checks={}
    def tensor(name,shape):
        entry=m['artifacts'][name]
        if entry.get('shape')!=shape or entry.get('dtype')!='float32_le' or len(payload[name])!=math.prod(shape)*4:raise ValueError('Tensor shape/type differs')
        return floats(payload[name])
    def wave(role):
        name='listening/focus/'+role+'.wav';entry=m['artifacts'][name];data=payload[name];count=entry['frames'];start=entry['start_sample']
        header=(b'RIFF'+struct.pack('<I',50+count*8)+b'WAVEfmt '+struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0)
                +b'fact'+struct.pack('<II',4,count)+b'data'+struct.pack('<I',count*8))
        if (data[:58]!=header or len(data)!=58+count*8 or not start<=a<a+n<=start+count
                or entry['source_file_sha256']!=done['artifact_sha256'][role+'.wav']):raise ValueError('Focus WAV identity/geometry differs')
        values=floats(data[58+(a-start)*8:58+(a-start+n)*8]);result=values[::2];result.extend(values[1::2]);return result
    for role in ('bass','drums','guitar'):
        assembled=tensor(role+'/assembled-focus.f32le',[1,2,n]);receipt=json.loads(payload[role+'/assembled.json'])
        if (receipt.get('schema_version')!=1 or receipt.get('execution_sha256')!=execution or receipt.get('instrument')!=role
                or receipt.get('target_samples48')!=done['target_samples48'] or not is_sha(receipt.get('raw_sha256'))
                or m['artifacts'][role+'/assembled-focus.f32le']['source_file_sha256']!=receipt['raw_sha256']
                or m['artifacts'][role+'/assembled-focus.f32le'].get('global_start_sample')!=a):raise ValueError('Raw stem provenance differs')
        total=[0.]*(2*n);weights=[0.]*n
        for i in indices:
            w=windows[i];b=max(a,w['sample_start48']);e=min(a+n,w['sample_end48']);count=e-b;prefix=role+'/window-%03d/'%i
            saved=json.loads(payload[prefix+'completed.json']);report=payload[prefix+'report.txt'];latent=payload[prefix+'final-latent.f32le']
            if (saved.get('schema_version')!=1 or saved.get('execution_sha256')!=execution or saved.get('job_reference_sha256')!=job
                    or saved.get('instrument')!=role or saved.get('window_index')!=i or hashlib.sha256(report).hexdigest()!=saved['report_sha256']
                    or ('LIVE final latent SHA256: '+hashlib.sha256(latent).hexdigest()) not in report.decode().splitlines()):raise ValueError('Window latent/report/checkpoint binding differs')
            tensor(prefix+'final-latent.f32le',[1,750,64])
            entry=m['artifacts'][prefix+'focus.f32le'];values=tensor(prefix+'focus.f32le',[1,2,count])
            if entry.get('global_start_sample')!=b or entry.get('local_start_sample')!=b-w['sample_start48'] or entry.get('source_file_sha256')!=saved['raw_sha256']:raise ValueError('Window raw range provenance differs')
            left=0 if i==0 else min(w['sample_end48'],windows[i-1]['sample_end48'])-w['sample_start48']
            right=0 if i+1==len(windows) else w['sample_end48']-max(w['sample_start48'],windows[i+1]['sample_start48'])
            for j in range(count):
                offset=b+j-w['sample_start48'];factor=1.;length=w['sample_end48']-w['sample_start48']
                if left>0 and offset<left:factor*=math.sin(.5*math.pi*(offset+.5)/left)**2
                if right>0 and offset>=length-right:factor*=math.cos(.5*math.pi*(offset-(length-right)+.5)/right)**2
                k=b-a+j;weights[k]+=factor;total[k]+=values[j]*factor;total[n+k]+=values[count+j]*factor
        if not all(math.isfinite(v) and v>0 for v in weights):raise ValueError('Uncovered focus raw samples')
        replay=array.array('f',(total[i]/weights[i%n] for i in range(2*n)))
        checks[role+'_raw_assembly']=comparison(replay,assembled)
        checks[role+'_wav_binding']={'passed':array.array('f',(v*gain for v in assembled))==wave(role),'scope':'Exact Float32 raw*saved gain on exported range'}
    # Piano/band WAVs are readable and source-receipt bound, without claiming a
    # source/mixture numerical comparison from missing full original source data.
    wave('piano');wave('piano-with-band')
    return {'profile':'full-song-device-focus-audit','scope':'Saved selected raw/WAV/OLA binding only; no original source/decoder/DiT parity',
            'checks':checks,'selected_checks_passed':all(c['passed'] for c in checks.values()),'new_inference_calls':0,
            'baseline_modified':False,'native_decoder_parity':'NOT EVALUATED','music_acceptance':'NOT EVALUATED BY AUDIT'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',required=True,type=Path);args=parser.parse_args()
    try:print(json.dumps(audit(args.archive),indent=2))
    except (ValueError,OSError,KeyError,TypeError,zipfile.BadZipFile) as error:parser.exit(1,'Audit stopped: '+str(error)+'\n')


if __name__=='__main__':main()
