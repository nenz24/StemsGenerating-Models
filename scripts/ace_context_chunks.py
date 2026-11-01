"""ACE-Step bounded-context experiments and measured, sample-aligned stitching.

Reuses the checked bass/drum workers. Plan/probe default to no inference.
Needs numpy, scipy, soundfile, ffmpeg, plus the existing CUDA runtime for --execute.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import correlate, find_peaks

import electric_full_song as single

RATE = 32000
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def layout(total, context=90, margin=6, fade=.58, cues=None, energy=None):
    values=(context,margin,fade)
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values)
        or not 10<=context<=600 or not 0<fade<margin or margin<1 or context-2*margin<10):
        raise ValueError('Need context 10–600s, margin>=1s, fade in (0,margin), core>=10s')
    if not isinstance(total,int) or not 10*RATE<=total<=600*RATE:
        raise ValueError('Source must be 10–600 seconds')
    # Integer model durations include end padding. Never exceed the requested cap.
    maximum=math.floor(context)*RATE
    pad=round(margin*RATE); core=maximum-2*pad
    if core<10*RATE:
        raise ValueError('Integer context cap leaves less than 10s of core audio')
    count=1 if total<=maximum else max(2,math.ceil((total-2*pad)/core))
    capacities=[core+pad if i in [0,count-1] else core for i in range(count)]
    if count==1:capacities=[maximum]
    cuts=[0]; policies=[]
    cues=[] if cues is None else cues
    if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t)
           or not 0<t<total/RATE for t in cues) or cues!=sorted(set(cues)):
        raise ValueError('Boundary times must be finite, unique, sorted and inside the song')
    for i in range(1,count):
        remaining=total-cuts[-1]
        target=cuts[-1]+round(remaining*capacities[i-1]/sum(capacities[i-1:]))
        lo=max(cuts[-1]+10*RATE,total-sum(capacities[i:]))
        hi=min(cuts[-1]+capacities[i-1],total-(count-i)*10*RATE)
        if lo>hi:raise ValueError('No feasible boundary under the requested context/margin limits')
        target=min(hi,max(lo,target))
        lo,hi=max(lo,target-2*RATE),min(hi,target+2*RATE)
        options=[round(t*RATE) for t in cues if lo<=round(t*RATE)<=hi]
        if options:
            cut=min(options,key=lambda t:abs(t-target)); policy='provided_boundary'
        else:
            candidates=list(range(lo,hi+1,RATE//10)) or [min(hi,max(lo,target))]
            cut=min(candidates,key=lambda t:(energy(t) if energy else 0)+.002*abs(t-target)/RATE)
            policy='low_energy_near_target' if energy else 'balanced_time'
        cuts.append(cut); policies.append(policy)
    cuts.append(total)
    chunks=[]
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        start,end=max(0,a-pad),min(total,b+pad)
        duration=(end-start)/RATE
        assert math.ceil(duration)<=context
        chunks.append({'index':i,'core_start':a,'core_end':b,'input_start':start,'input_end':end,
                       'model_duration':math.ceil(duration)})
    return {'sample_rate':RATE,'total_frames':total,'context_cap_seconds':context,
            'margin_seconds':margin,'fade_seconds':fade,'boundary_policies':policies,'chunks':chunks}


def read_audio(path):
    data,sr=sf.read(path,always_2d=True,dtype='float32')
    if sr!=RATE or data.shape[1]!=2 or not np.isfinite(data).all():
        raise ValueError('Need finite stereo 32kHz audio')
    return data


def instrument_of(plan):
    instrument=plan.get('instrument',plan.get('preset',{}).get('instrument','bass'))
    if instrument not in ('bass','drums'):raise ValueError('Supported chunk instruments: bass, drums')
    return instrument


def worker_for(instrument):
    if instrument=='bass':return single
    if instrument=='drums':
        import drum_probe
        return drum_probe
    raise ValueError('Supported chunk instruments: bass, drums')


def files_for(plan):
    return ('drum_probe.json','drums.wav','drums_full') if instrument_of(plan)=='drums' else ('single_pass.json','electric_bass.wav','electric_bass')


def prepare(source, output, context=90, margin=6, fade=.58, cues_file=None, *, instrument='bass'):
    worker=worker_for(instrument)
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    original_hash=sha256(source)
    generation_input=output/('input'+source.suffix.lower())
    shutil.copyfile(source,generation_input)
    if sha256(generation_input)!=original_hash:
        raise ValueError('Original input changed while preparing the plan')
    clean=output/'source.wav'
    # Canonical audio is for planning/mixing only. Generate from the preserved
    # original so preprocessing matches the listener-approved single-pass probes.
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(source),'-ar',str(RATE),
                    '-ac','2','-c:a','pcm_s16le',str(clean)],check=True)
    audio=read_audio(clean)
    cues=None
    if cues_file:
        metadata=json.loads(Path(cues_file).read_text())
        if metadata.get('source_sha256')!=sha256(source):
            raise ValueError('Boundary metadata does not match the original source hash')
        cues=metadata.get('downbeats',[])+metadata.get('sections',[])
        cues=sorted(set(cues))
    def energy(frame):
        block=audio[max(0,frame-RATE//8):min(len(audio),frame+RATE//8)]
        return float(np.sqrt(np.mean(block**2)))
    plan=layout(len(audio),context,margin,fade,cues,energy)
    plan.update(schema_version=2,status='planned',original_sha256=original_hash,source_file='source.wav',
                source_sha256=sha256(clean),preset=worker.PRESET,instrument=instrument,
                generation_source_file=generation_input.name,generation_source_sha256=original_hash,
                note='No inferred meter/downbeats. Context cap includes margins and integer end padding. Independent chunks; continuity and musical quality unverified.')
    write_json(output/'plan.json',plan)
    return plan


def generation_source(folder,plan):
    path=(folder/plan.get('generation_source_file',plan['source_file'])).resolve()
    expected=plan.get('generation_source_sha256',plan['source_sha256'])
    if not path.is_relative_to(folder.resolve()) or sha256(path)!=expected:
        raise ValueError('Generation source changed or escapes the run')
    return path


def load_plan(path):
    path=Path(path).resolve();plan=json.loads(path.read_text())
    source=path.parent/plan['source_file']
    if not source.resolve().is_relative_to(path.parent) or sha256(source)!=plan['source_sha256']:
        raise ValueError('Canonical source changed or escapes the run')
    if plan['preset']!=worker_for(instrument_of(plan)).PRESET or single.frames(source)!=plan['total_frames']:
        raise ValueError('Preset or source duration differs from plan')
    generation_source(path.parent,plan)
    return path.parent,plan,source


def verify_chunk(folder,plan,chunk):
    report_file,stem_file,_=files_for(plan)
    report=json.loads((folder/report_file).read_text())
    stem=folder/stem_file;metadata=report.get('input',{})
    if (report.get('status')!='generated' or report.get('preset')!=plan['preset']
        or metadata.get('original_sha256')!=plan.get('generation_source_sha256',plan['source_sha256'])
        or metadata.get('start')!=chunk['input_start']/RATE
        or metadata.get('requested_duration')!=(chunk['input_end']-chunk['input_start'])/RATE
        or metadata.get('source_gain')!=.7 or metadata.get('model_duration')!=chunk['model_duration']
        or single.frames(stem)!=chunk['input_end']-chunk['input_start']
        or sha256(stem)!=report.get('stem_sha256')):
        raise ValueError('Chunk provenance/settings/duration/hash mismatch')
    return stem


def run_chunks(root,plan_path):
    folder,plan,source=load_plan(plan_path)
    worker=worker_for(instrument_of(plan));report_file,_,_=files_for(plan)
    source=generation_source(folder,plan)
    import fcntl
    with (folder/'.generation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress_file=folder/'progress.json'
        progress=json.loads(progress_file.read_text()) if progress_file.exists() else {}
        for chunk in plan['chunks']:
            key=str(chunk['index']);parent=folder/f'chunk_{chunk["index"]:03d}'
            parent.mkdir(exist_ok=True)
            if key in progress:
                cached=(folder/progress[key]).resolve()
                if not cached.is_relative_to(parent): raise ValueError('Cached path escapes chunk')
                verify_chunk(cached,plan,chunk)
                print('Reuse verified chunk',key,flush=True);continue
            # Recover a finished attempt after an interrupted progress write.
            completed=None
            for attempt in sorted(parent.glob('attempt_*')):
                report=attempt/report_file
                if report.is_file() and json.loads(report.read_text()).get('status')=='generated':
                    verify_chunk(attempt,plan,chunk);completed=attempt;break
            if completed is None:
                single.model_python(Path(root))  # fail before creating an empty attempt
                number=0
                while (parent/f'attempt_{number:03d}').exists():number+=1
                completed=parent/f'attempt_{number:03d}'
                worker.generate(root,source,completed,start=chunk['input_start']/RATE,
                    duration=(chunk['input_end']-chunk['input_start'])/RATE,execute=True)
                verify_chunk(completed,plan,chunk)
            progress[key]=str(completed.relative_to(folder));write_json(progress_file,progress)
    return progress


def db(value):
    return float(20*np.log10(max(float(value),1e-12)))


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio,dtype=np.float64))))


def pitch(audio):
    mono=audio.mean(axis=1)[::8].astype(np.float64)
    if len(mono)<64 or rms(mono)<1e-4:return {'hz':None,'periodicity':0.}
    mono-=mono.mean()
    ac=correlate(mono,mono,mode='full',method='fft')[len(mono)-1:]
    ac/=np.maximum(1,len(mono)-np.arange(len(mono)))
    low,high=round(4000/250),min(round(4000/35),len(ac)-1)
    peaks,_=find_peaks(ac[low:high]);peaks+=low
    if not len(peaks) or ac[0]<=0:return {'hz':None,'periodicity':0.}
    best=float(ac[peaks].max());lag=int(min(p for p in peaks if ac[p]>=best*.95))
    confidence=float(np.clip(ac[lag]/ac[0],0,1))
    return {'hz':4000/lag if confidence>=.55 else None,'periodicity':confidence}


def fade_weights(count):
    angle=np.linspace(0,np.pi/2,count,dtype=np.float64)
    return np.cos(angle)[:,None],np.sin(angle)[:,None]


def seam_window(left,right,lchunk,rchunk,center,width):
    start=center-width//2;end=start+width
    a=left[start-lchunk['input_start']:end-lchunk['input_start']]
    b=right[start-rchunk['input_start']:end-rchunk['input_start']]
    if a.shape!=(width,2) or b.shape!=(width,2):raise ValueError('Fade extends beyond both contexts')
    wa,wb=fade_weights(width);blend=a*wa+b*wb
    ea,eb=rms(a),rms(b);ref=math.sqrt((ea*ea+eb*eb)/2)
    corr=float(np.mean(a.astype(np.float64)*b)/(ea*eb)) if min(ea,eb)>1e-7 else None
    ratio=db(rms(blend)/ref) if ref>1e-7 else 0.
    # Prefer quiet, correlated transitions, penalize both cancellation and buildup.
    cost=(ea+eb)+.02*abs(ratio)+(.02*(1-corr) if corr is not None else 0)
    return start,end,blend,{'center_seconds':center/RATE,'left_rms_dbfs':db(ea),
        'right_rms_dbfs':db(eb),'correlation':corr,'blend_relative_rms_db':ratio,
        'predicted_level_change_db':db(eb/max(ea,1e-12)),'cost':float(cost)}


def stitch_arrays(plan,audios):
    chunks=plan['chunks'];width=round(plan['fade_seconds']*RATE)
    if len(chunks)!=len(audios):raise ValueError('Every chunk must be present')
    output=np.zeros((plan['total_frames'],2),dtype=np.float32)
    for chunk,audio in zip(chunks,audios):
        if audio.shape!=(chunk['input_end']-chunk['input_start'],2) or not np.isfinite(audio).all():
            raise ValueError('Chunk audio shape/nonfinite mismatch')
        a,b=chunk['core_start'],chunk['core_end']
        output[a:b]=audio[a-chunk['input_start']:b-chunk['input_start']]
    seams=[]
    for i in range(1,len(chunks)):
        left,right=chunks[i-1],chunks[i];nominal=right['core_start']
        search=round(min(1.5,plan['margin_seconds']/3)*RATE)
        candidates=([nominal] if plan['boundary_policies'][i-1]=='provided_boundary'
                    else range(nominal-search,nominal+search+1,RATE//20))
        # Retain at least half the context margin outside the fade at each edge.
        trim=round(plan['margin_seconds']*RATE/2)
        candidates=[c for c in candidates if c-width//2>=right['input_start']+trim
                    and c+(width-width//2)<=left['input_end']-trim]
        if not candidates:raise ValueError('No seam with sufficient edge discard')
        best=min((seam_window(audios[i-1],audios[i],left,right,c,width) for c in candidates),key=lambda x:x[3]['cost'])
        a,b,blend,stats=best
        # If moved, extend the correct chunk's trusted interior to the new fade.
        output[nominal-search:a]=audios[i-1][nominal-search-left['input_start']:a-left['input_start']]
        output[b:nominal+search+width]=audios[i][b-right['input_start']:nominal+search+width-right['input_start']]
        output[a:b]=blend
        before=audios[i-1][max(0,a-left['input_start']-RATE//3):a-left['input_start']]
        after=audios[i][b-right['input_start']:b-right['input_start']+RATE//3]
        if instrument_of(plan)=='drums':
            lp=rp={'hz':None,'periodicity':None,'note':'Pitch continuity is not a drum-groove measure'}
        else:lp,rp=pitch(before),pitch(after)
        cents=1200*math.log2(rp['hz']/lp['hz']) if lp['hz'] and rp['hz'] else None
        stats.update(index=i,nominal_seconds=nominal/RATE,fade_start_frame=a,fade_end_frame=b,
                     left_pitch=lp,right_pitch=rp,pitch_change_cents=cents,
                     manual_review=(stats['blend_relative_rms_db'] < -3 or abs(stats['predicted_level_change_db'])>6
                                    or cents is not None and abs(cents)>150),
                     musical_review_required=True,
                     note='Signal flags do not evaluate groove/timbre. Equal-power does not guarantee phase continuity.')
        seams.append(stats)
    peak=float(np.max(np.abs(output)))
    gain=min(1.,.95/peak) if peak else 1.
    return output*gain,{'seams':seams,'global_headroom_gain':gain,'pre_gain_peak':peak,
                        'note':'Signal diagnostics only; no automatic claim of chord accuracy, timbre consistency or musical quality.'}


def stitch_run(plan_path):
    folder,plan,source=load_plan(plan_path)
    report_file,_,stem_name=files_for(plan)
    progress=json.loads((folder/'progress.json').read_text())
    audios=[];verified=[]
    for chunk in plan['chunks']:
        run=(folder/progress[str(chunk['index'])]).resolve()
        if not run.is_relative_to(folder/f'chunk_{chunk["index"]:03d}'):raise ValueError('Chunk path escapes run')
        stem=verify_chunk(run,plan,chunk)
        audios.append(read_audio(stem));verified.append((run,stem))
    output=folder/'stitched'
    if output.exists():raise ValueError('Stitched output exists; preserve it before another stitch')
    audio,report=stitch_arrays(plan,audios)
    output.mkdir()
    sf.write(output/f'{stem_name}.wav',audio,RATE,subtype='PCM_24')
    sf.write(output/'mix_full.wav',.5*(.7*read_audio(source)+audio),RATE,subtype='PCM_24')
    for stem in [stem_name,'mix_full']:
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(output/f'{stem}.wav'),'-c:a','libmp3lame','-b:a','192k',str(output/f'{stem}.mp3')],check=True)
    for seam in report['seams']:
        center=round(seam['center_seconds']*RATE)
        sf.write(output/f'seam_{seam["index"]:03d}.wav',audio[max(0,center-3*RATE):min(len(audio),center+3*RATE)],RATE,subtype='PCM_24')
    (output/'chunk_reports').mkdir()
    for chunk,(run,stem) in zip(plan['chunks'],verified):
        start=max(0,(chunk['core_start']-chunk['input_start'])/RATE)
        length=min(8.,(chunk['core_end']-chunk['core_start'])/RATE)
        write_json(output/'chunk_reports'/f'{chunk["index"]:03d}.json',
            {'chunk':chunk,'stem_sha256_verified':sha256(stem),
             'preview_song_start_seconds':chunk['core_start']/RATE,
             'preview_seconds':length,'generation_report':json.loads((run/report_file).read_text())})
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-ss',str(start),'-i',str(stem),
            '-t',str(length),'-c:a','libmp3lame','-b:a','192k',str(output/f'chunk_{chunk["index"]:03d}_before_stitch.mp3')],check=True)
    report.update(status='stitched',instrument=instrument_of(plan),export_frames=len(audio),
                  stem_file=f'{stem_name}.wav',stem_sha256=sha256(output/f'{stem_name}.wav'))
    write_json(output/'seams.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--source',type=Path,required=True);plan.add_argument('--out',type=Path,required=True)
    plan.add_argument('--context-seconds',type=float,default=90);plan.add_argument('--margin-seconds',type=float,default=6)
    plan.add_argument('--fade-seconds',type=float,default=.58);plan.add_argument('--cues',type=Path)
    plan.add_argument('--instrument',choices=['bass','drums'],default='bass')
    for name in ['probe','run','stitch']:
        s=sub.add_parser(name);s.add_argument('--plan',type=Path,required=True)
        if name!='stitch':s.add_argument('--root',type=Path,default=ROOT);s.add_argument('--execute',action='store_true')
        if name=='probe':s.add_argument('--duration',type=int,choices=[60,90,120],required=True);s.add_argument('--start',type=float,default=0)
    args=p.parse_args()
    if args.command=='plan':
        result=prepare(args.source,args.out,args.context_seconds,args.margin_seconds,args.fade_seconds,args.cues,instrument=args.instrument)
    elif args.command=='probe':
        folder,plan,source=load_plan(args.plan)
        parent=folder/f'probe_{args.duration:03d}';parent.mkdir(exist_ok=True)
        index=0
        while (parent/f'attempt_{index:03d}').exists():index+=1
        result=worker_for(instrument_of(plan)).generate(args.root,generation_source(folder,plan),parent/f'attempt_{index:03d}',start=args.start,duration=args.duration,execute=args.execute)
    elif args.command=='run':
        if args.execute:result=run_chunks(args.root,args.plan)
        else:_,result,_=load_plan(args.plan)
    else:result=stitch_run(args.plan)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
