#!/usr/bin/env python3
"""Export small existing-audio/full-song seam evidence; no model execution or DSP repair."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

from coreai_listening_audio import write_float_wav
from export_coreai_decoder import ROOT, sha256
from prepare_coreai_full_song import schedule

RATE=48000
FRAMES=1440000
LIMIT=32*1024*1024
ROLES=('bass','drums','guitar')
WAVS=('piano.wav','bass.wav','drums.wav','guitar.wav','piano-with-band.wav')


def is_sha(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def regular(path,size=None):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():raise ValueError('Missing or linked file: '+str(path))
    if size is not None and path.stat().st_size!=size:raise ValueError('Wrong-size file: '+path.name)
    return path


def metadata(path):
    path=regular(path)
    if path.stat().st_size>1024*1024:raise ValueError('Metadata exceeds1MiB')
    return json.loads(path.read_bytes())


def floats(raw):
    values=array.array('f');values.frombytes(raw)
    if sys.byteorder!='little':values.byteswap()
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite selected audio/tensor')
    return values


def raw_range(path,plane,start,count):
    regular(path,plane*8)
    if not 0<=start<start+count<=plane:raise ValueError('Invalid stereo range')
    values=array.array('f')
    with Path(path).open('rb') as stream:
        for channel in range(2):
            stream.seek((channel*plane+start)*4)
            block=stream.read(count*4)
            if len(block)!=count*4:raise ValueError('Incomplete audio range')
            values.extend(floats(block))
    return values


class Wave:
    """Bounded RIFF reader for the app's Float32 and ordinary PCM16 stereo exports."""
    def __init__(self,path):
        self.path=regular(path);size=self.path.stat().st_size;fmt=None;data=None
        with self.path.open('rb') as stream:
            header=stream.read(12)
            if len(header)!=12 or header[:4]!=b'RIFF' or header[8:]!=b'WAVE' or struct.unpack('<I',header[4:8])[0]+8!=size:
                raise ValueError('Expected complete RIFF WAV')
            offset=12
            while offset+8<=size:
                stream.seek(offset);tag,n=struct.unpack('<4sI',stream.read(8));position=offset+8
                if position+n>size:raise ValueError('Truncated WAV chunk')
                if tag==b'fmt ':
                    if fmt is not None or not 16<=n<=64:raise ValueError('Invalid WAV format chunk')
                    fmt=struct.unpack('<HHIIHH',stream.read(16))
                if tag==b'data':
                    if data is not None:raise ValueError('Multiple WAV data chunks')
                    data=(position,n)
                offset=position+n+(n%2)
        if fmt is None or data is None:raise ValueError('Missing WAV format/audio')
        encoding,channels,rate,byte_rate,align,bits=fmt
        if (channels!=2 or rate!=RATE or (encoding,bits) not in ((3,32),(1,16))
                or align!=2*bits//8 or byte_rate!=RATE*align or data[1]%align or data[1]==0):
            raise ValueError('Select stereo48kHz Float32 or PCM16 WAV')
        self.encoding,self.bits,self.align=encoding,bits,align
        self.offset,self.frames=data[0],data[1]//align

    def read(self,start,count):
        if not 0<=start<start+count<=self.frames:raise ValueError('Invalid WAV range')
        with self.path.open('rb') as stream:
            stream.seek(self.offset+start*self.align);raw=stream.read(count*self.align)
        if len(raw)!=count*self.align:raise ValueError('Incomplete WAV range')
        if self.encoding==3:values=floats(raw)
        else:
            integers=array.array('h');integers.frombytes(raw)
            if sys.byteorder!='little':integers.byteswap()
            values=array.array('f',(x/32768 for x in integers))
        result=values[::2];result.extend(values[1::2]);return result


def regions(frames,seconds):
    count=min(frames,seconds*RATE)
    return [(name,start,count) for name,start in [('beginning',0),('middle',(frames-count)//2),('end',frames-count)]]


def focus_range(frames,seconds,length):
    if not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or not 0<=seconds<frames/RATE:
        raise ValueError('Focus time must lie inside the saved song')
    count=min(frames,length*RATE);start=max(0,min(round(seconds*RATE)-count//2,frames-count))
    return start,count


def contributors(windows,start,count):
    return [w['index'] for w in windows if max(start,w['sample_start48'])<min(start+count,w['sample_end48'])]


def discover(source,execution=None):
    source=Path(source).absolute()
    if not source.is_dir() or any(p.is_symlink() for p in (source,*source.parents)):raise ValueError('Select an existing unlinked download folder')
    if execution is not None and not is_sha(execution):raise ValueError('Invalid --execution-sha')
    found=[];seen=0;errors=[]
    for current,dirs,_ in os.walk(source,followlinks=False,onerror=errors.append):
        seen+=1;current=Path(current)
        if seen>4096:raise ValueError('Search limit reached; select one downloaded container directly')
        if current.parent.name=='FullSongGeneration' and is_sha(current.name):
            if (execution is None or current.name==execution) and (current/'completed.json').is_file():found.append(current)
            dirs[:]=[];continue
        if len(current.relative_to(source).parts)>=8:dirs[:]=[]
        else:dirs[:]=sorted(n for n in dirs if n not in {'.git','Library','Caches','node_modules','.Trash','__pycache__'} and not (current/n).is_symlink())
    if errors:raise ValueError('Incomplete folder search; select the downloaded container directly')
    if not found:raise ValueError('Completed FullSongGeneration not found in '+str(source)+'. Download the current CoreAIProbe container; do not repeat generation.')
    if len(found)>1:
        # Identical downloaded receipts may resolve; verify the selected bytes below.
        identities={(p.name,sha256(regular(p/'completed.json')),sha256(regular(p/'report.txt'))) for p in found}
        if len(identities)!=1:raise ValueError('Several different runs/copies found; select one container or use --execution-sha')
    return sorted(found)[0]


def inspect(folder):
    done=metadata(folder/'completed.json');execution=folder.name
    if (done.get('schema_version')!=1 or done.get('execution_sha256')!=execution or not is_sha(execution)
            or not is_sha(done.get('job_reference_sha256')) or type(done.get('target_samples48')) is not int
            or not 0<done['target_samples48']<=(2**32-1-50)//8
            or not isinstance(done.get('listening_gain'),(float,int)) or not math.isfinite(done['listening_gain'])
            or not 0<done['listening_gain']<=1 or set(done.get('artifact_sha256',{}))!=set(WAVS)
            or not all(is_sha(v) for v in done['artifact_sha256'].values())):raise ValueError('Invalid completed collection receipt')
    frames=done['target_samples48'];plan=schedule(frames*2//3)
    if plan['target_samples48']!=frames or done.get('completed_windows')!=len(plan['windows'])*3 or len(plan['windows'])<2:
        raise ValueError('Not a complete multiple-window song schedule')
    report=regular(folder/'report.txt')
    if report.stat().st_size>1024*1024:raise ValueError('Report exceeds1MiB')
    text=report.read_text()
    if ('Runtime: CoreAI full-song prepared-input diagnostic' not in text or 'Full-song collection: COMPLETED' not in text):
        raise ValueError('Select the completed full-song diagnostic')
    waves={}
    for name,expected in done['artifact_sha256'].items():
        file=regular(folder/name,58+frames*8)
        if sha256(file)!=expected:raise ValueError('Completed WAV SHA mismatch: '+name)
        waves[name]=Wave(file)
        if waves[name].frames!=frames or waves[name].encoding!=3:raise ValueError('Full-song WAV format differs')
    captures={};stems={}
    for role in ROLES:
        receipt=metadata(folder/(role+'-assembled.json'));file=regular(folder/('raw-'+role+'.f32le'),frames*8)
        if (receipt.get('schema_version')!=1 or receipt.get('execution_sha256')!=execution or receipt.get('instrument')!=role
                or receipt.get('target_samples48')!=frames or sha256(file)!=receipt.get('raw_sha256')):
            raise ValueError('Raw stem receipt/SHA mismatch: '+role)
        stems[role]=file
        for w in plan['windows']:
            parent=folder/role/('window-%03d'%w['index']);saved=metadata(parent/'completed.json');child=saved.get('child_directory','')
            try:identifier=str(uuid.UUID(child.removeprefix('attempt-'))).upper()
            except (ValueError,AttributeError):raise ValueError('Invalid window attempt directory')
            if (child!='attempt-'+identifier or saved.get('schema_version')!=1 or saved.get('execution_sha256')!=execution
                    or saved.get('job_reference_sha256')!=done['job_reference_sha256'] or saved.get('instrument')!=role
                    or saved.get('window_index')!=w['index'] or not is_sha(saved.get('raw_sha256')) or not is_sha(saved.get('report_sha256'))):
                raise ValueError('Window checkpoint identity mismatch')
            child=parent/child;raw=regular(child/('raw-'+role+'.f32le'),FRAMES*8);report=regular(child/'report.txt')
            if report.stat().st_size>1024*1024 or sha256(report)!=saved['report_sha256']:raise ValueError('Window report SHA mismatch')
            captures[role,w['index']]={'folder':child,'raw':raw,'receipt':saved,'pointer':parent/'completed.json'}
    return done,plan,waves,stems,captures


def comparison(actual,expected):
    if len(actual)!=len(expected) or not actual:raise ValueError('Invalid comparison arrays')
    squared=energy=0.;maximum=ratio=0.;violations=0
    for a,b in zip(actual,expected):
        if not math.isfinite(a) or not math.isfinite(b):raise ValueError('Nonfinite comparison')
        error=abs(a-b);bound=1e-4+1e-3*abs(b)
        maximum=max(maximum,error);ratio=max(ratio,error/bound);violations+=error>bound;squared+=error*error;energy+=b*b
    n=len(actual);rmse=math.sqrt(squared/n);normalized=rmse/max(math.sqrt(energy/n),1e-12)
    return {'passed':violations==0 and normalized<=1e-3,'max_absolute_error':maximum,'rmse':rmse,
            'normalized_rmse':normalized,'pointwise_violations':violations,'values':n,'worst_bound_ratio':ratio}


def assembled_range(windows,parts,start,count):
    """Independent range replay of the shipped Swift Float64 normalized OLA."""
    total=[0.]*(2*count);weight=[0.]*count
    for index,values in parts.items():
        w=windows[index];a=w['sample_start48'];b=w['sample_end48']
        left=0 if index==0 else min(b,windows[index-1]['sample_end48'])-a
        right=0 if index+1==len(windows) else b-max(a,windows[index+1]['sample_start48'])
        for j in range(count):
            offset=start+j-a;n=b-a;factor=1.
            if left>0 and offset<left:factor*=math.sin(.5*math.pi*(offset+.5)/left)**2
            if right>0 and offset>=n-right:factor*=math.cos(.5*math.pi*(offset-(n-right)+.5)/right)**2
            weight[j]+=factor
            total[j]+=values[j]*factor;total[count+j]+=values[count+j]*factor
    if not all(math.isfinite(v) and v>0 for v in weight):raise ValueError('Uncovered OLA range')
    return array.array('f',(total[i]/weight[i%count] for i in range(2*count)))


def package(out,audio=None,source=None,execution=None,focus_seconds=None):
    out=Path(out).absolute()
    if out.exists() or out.is_symlink():raise FileExistsError('Existing output preserved: '+str(out))
    if (audio is None)==(source is None):raise ValueError('Choose one audio file or container folder')
    if focus_seconds is not None and (source is None or not math.isfinite(focus_seconds) or focus_seconds<0):
        raise ValueError('A finite nonnegative focus time requires a container source')
    folder=None
    if source is not None:
        folder=discover(source,execution)
        documents=folder.parent.parent
        container=documents.parent.parent if documents.parent.name=='AppData' else documents.parent
        if out.resolve().is_relative_to(container.resolve()):raise ValueError('Output must be outside the downloaded container')
    if any(out.resolve().is_relative_to(ROOT/p) for p in ('baselines','prototypes/ios27-coreai/ModelCandidates')):raise ValueError('Output must be outside models/baselines')
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest={'schema_version':1,'profile':'private-full-song-quality-review-v1','new_inference_calls':0,'native_exports':0,
        'baseline_modified':False,'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
        'full_song_numerical_parity':'NOT EVALUATED','music_acceptance':'NOT EVALUATED BY THIS TOOL',
        'scope':'Selected existing audio ranges and transport/assembly checks; no musical/root-cause/repair claim',
        'privacy':'Private user audio/tensors/reports; never commit this archive or its contents','clips':[],'artifacts':{},'checks':{}}
    with tempfile.TemporaryDirectory(prefix='.full-song-quality-',dir=out.parent) as scratch:
        scratch=Path(scratch)
        def record(name,**attributes):
            path=scratch/name
            manifest['artifacts'][name]={'sha256':sha256(path),'bytes':path.stat().st_size,**attributes}
        def save_bytes(name,raw,**attributes):
            path=scratch/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);record(name,**attributes)
        def wav_clip(wave,name,start,count):
            path=scratch/name;path.parent.mkdir(parents=True,exist_ok=True)
            write_float_wav(path,wave.read(start,count),count)
            record(name,source_file_sha256=manifest['source_file_sha256'] if folder is None else done['artifact_sha256'][wave.path.name],start_sample=start,frames=count)
            manifest['clips'].append({'path':name,'start_seconds':start/RATE,'duration_seconds':count/RATE,'additional_gain':1.})
        if folder is None:
            wave=Wave(audio);manifest.update(mode='audio-only',source_file_sha256=sha256(wave.path),frames=wave.frames,
                missing_evidence='Individual instruments, raw windows, latent and checkpoints NOT PROVIDED')
            for name,start,count in regions(wave.frames,8):wav_clip(wave,name+'.wav',start,count)
            if sha256(wave.path)!=manifest['source_file_sha256']:raise ValueError('Source WAV changed while reading; no output published')
        else:
            done,plan,waves,stems,captures=inspect(folder);frames=done['target_samples48'];windows=plan['windows']
            gain=struct.unpack('<f',struct.pack('<f',done['listening_gain']))[0]
            manifest.update(mode='container',completed_receipt=done,schedule=plan['windows'],listening_gain_float32=gain)
            listening=regions(frames,3)
            focus=None
            if focus_seconds is not None:
                a,n=focus_range(frames,focus_seconds,3);listening[-1]=('focus',a,n)
                a,n=focus_range(frames,focus_seconds,2)
                focus={'requested_seconds':focus_seconds,'start_sample':a,'frames':n,'window_indices':contributors(windows,a,n)}
                manifest['focus']=focus
            save_bytes('device-report.txt',(folder/'report.txt').read_bytes())
            save_bytes('completed.json',(folder/'completed.json').read_bytes())
            for name,wave in waves.items():
                for region,start,count in listening:wav_clip(wave,'listening/'+region+'/'+name,start,count)
            previous,last=windows[-2],windows[-1]
            begin=last['sample_start48'];end=min(previous['sample_end48'],last['sample_end48']);middle=(begin+end)//2
            boundaries={begin,end,*[v for w in windows for v in (w['sample_start48'],w['sample_end48']) if begin<=v<=end]}
            left=max(v for v in boundaries if v<=middle);right=min(v for v in boundaries if v>middle)
            count=min(3*RATE,right-left);start=max(left,min(middle-count//2,right-count))
            indices=[w['index'] for w in windows if w['sample_start48']<=start and w['sample_end48']>=start+count]
            selected=sorted({0,len(windows)//2,len(windows)-1,*indices,*(focus['window_indices'] if focus else [])})
            manifest['seam']={'start_sample':start,'frames':count,'window_indices':indices,'selected_latent_windows':selected}
            for role in ROLES:
                parts={}
                for i in selected:
                    cap=captures[role,i];child=cap['folder']
                    if sha256(cap['raw'])!=cap['receipt']['raw_sha256']:raise ValueError('Window raw SHA mismatch')
                    latent=regular(child/'final-latent.f32le',750*64*4);raw=latent.read_bytes();floats(raw)
                    text=(child/'report.txt').read_text()
                    if not re.search(r'^LIVE final latent SHA256: '+hashlib.sha256(raw).hexdigest()+r'$',text,re.M):raise ValueError('Final latent/report SHA mismatch')
                    prefix=role+'/window-%03d/'%i
                    save_bytes(prefix+'final-latent.f32le',raw,shape=[1,750,64],dtype='float32_le')
                    save_bytes(prefix+'report.txt',(child/'report.txt').read_bytes())
                    save_bytes(prefix+'completed.json',cap['pointer'].read_bytes())
                    if i in indices:
                        values=raw_range(cap['raw'],FRAMES,start-windows[i]['sample_start48'],count);parts[i]=values
                        if sys.byteorder!='little':values.byteswap()
                        save_bytes(prefix+'overlap.f32le',values.tobytes(),shape=[1,2,count],dtype='float32_le',global_start_sample=start,local_start_sample=start-windows[i]['sample_start48'])
                        if sys.byteorder!='little':values.byteswap()
                actual=raw_range(stems[role],frames,start,count)
                little=array.array('f',actual)
                if sys.byteorder!='little':little.byteswap()
                save_bytes(role+'/assembled-overlap.f32le',little.tobytes(),shape=[1,2,count],dtype='float32_le',global_start_sample=start)
                manifest['checks'][role+'_selected_overlap_replay']=comparison(assembled_range(windows,parts,start,count),actual)
                if focus:
                    a,n=focus['start_sample'],focus['frames'];values=raw_range(stems[role],frames,a,n)
                    little=array.array('f',values)
                    if sys.byteorder!='little':little.byteswap()
                    receipt=metadata(folder/(role+'-assembled.json'))
                    save_bytes(role+'/assembled-focus.f32le',little.tobytes(),shape=[1,2,n],dtype='float32_le',
                        global_start_sample=a,source_file_sha256=receipt['raw_sha256'])
                    save_bytes(role+'/assembled.json',(folder/(role+'-assembled.json')).read_bytes())
                    if len(focus['window_indices'])==1:
                        i=focus['window_indices'][0]
                        original=raw_range(captures[role,i]['raw'],FRAMES,a-windows[i]['sample_start48'],n)
                        manifest['checks'][role+'_focus_single_window_binding']={'passed':values==original,
                            'scope':'Exact raw values; focus is outside generation overlap; not decoder/trajectory parity'}
                for region,a,n in listening:
                    raw=raw_range(stems[role],frames,a,n)
                    scaled=array.array('f',(x*gain for x in raw))
                    # Exact IEEE Float32 gain replay, bound to the receipt-verified device WAV.
                    manifest['checks'][role+'_'+region+'_wav_binding']={'passed':scaled==waves[role+'.wav'].read(a,n),'scope':'Selected range only; exact Float32 values'}
        if not all(check['passed'] for check in manifest['checks'].values()):manifest['sampled_transport_or_assembly']='NOT PASSED; preserve evidence'
        elif manifest['checks']:manifest['sampled_transport_or_assembly']='PASS for selected checks only; music/root cause NOT EVALUATED'
        archive_path=scratch/'review.zip'
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name in manifest['artifacts']:archive.write(scratch/name,name)
            archive.writestr('review-manifest.json',json.dumps(manifest,indent=2)+'\n')
        if archive_path.stat().st_size>LIMIT:raise ValueError('Archive exceeds32MiB; no final output published')
        os.link(archive_path,out)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio',type=Path,help='Existing piano-with-band WAV')
    group.add_argument('--choose-audio',action='store_true',help='Choose the existing WAV in macOS')
    group.add_argument('--source',type=Path,help='Downloaded container, Documents, execution folder or Downloads')
    group.add_argument('--choose-container',action='store_true')
    parser.add_argument('--execution-sha',help='Exact execution SHA from the original full-song report')
    parser.add_argument('--focus-seconds',type=float,help='Container only: replace silent tail with a3s focus and include2s raw evidence')
    parser.add_argument('--out',type=Path,help='New ZIP, outside the container; default chooses the next unused version')
    args=parser.parse_args()
    if args.execution_sha and not (args.source or args.choose_container):parser.error('--execution-sha requires a container source')
    if args.focus_seconds is not None and not (args.source or args.choose_container):parser.error('--focus-seconds requires a container source')
    if args.choose_audio or args.choose_container:
        if sys.platform!='darwin':parser.error('Chooser requires macOS; use --audio/--source')
        command='POSIX path of (choose '+('file with prompt "Pilih WAV piano-with-band hasil yang sudah ada"' if args.choose_audio else 'folder with prompt "Pilih container CoreAIProbe yang sudah di-download"')+')'
        result=subprocess.run(['osascript','-e',command],capture_output=True,text=True)
        if result.returncode:parser.exit(1,'Selection canceled; existing data unchanged.\n')
        if args.choose_audio:args.audio=Path(result.stdout.rstrip('\r\n'))
        else:args.source=Path(result.stdout.rstrip('\r\n'))
    out=args.out
    if out is None:
        version=1
        while True:
            out=ROOT/('.runtime-cache/full-song-quality-review-v%d.zip'%version)
            if not out.exists() and not out.is_symlink():break
            version+=1
    try:manifest=package(out,args.audio,args.source,args.execution_sha,args.focus_seconds)
    except (ValueError,OSError,KeyError,TypeError) as error:parser.exit(1,'Review not created: '+str(error)+'\n')
    print('Private review:',out,';',out.stat().st_size,'bytes;',manifest['mode'])
    print('Existing outputs only; no generation, export, baseline or audio repair. Send this ZIP.')
    if sys.platform=='darwin':subprocess.run(['open','-R',str(out)],check=False)


if __name__=='__main__':main()
