"""Revoice an existing piano_bass analysis as a phrased finger-bass performance.

Retains the previous harmony/timeline. Seeded phrasing is a performance heuristic,
not a learned bassist model. SoundFont rendering cannot add unsupported sampled
slides, hammer-ons or true dead-note articulations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mido
import numpy as np
import soundfile as sf

from piano_bass import render


def arrange(report, seed=24):
    rng = np.random.default_rng(seed)
    chords = report['chords']
    base = report['notes']
    lookup = {round(n['start'], 5):n for n in base}
    duration = report['duration_seconds']
    notes=[]
    for i,chord in enumerate(chords):
        n=lookup.get(round(chord['start'],5))
        if n is None:
            continue
        start,end=chord['start'],chord['end']
        beat=end-start
        same_next=(i+1<len(chords) and chord['chord']==chords[i+1]['chord']
                   and round(chords[i+1]['start'],5) in lookup)
        same_prev=i>0 and chord['chord']==chords[i-1]['chord']
        phase=i%8  # phrase convention; does not claim inferred meter/downbeat
        pitch=n['midi']
        # Sparse entrances, occasional sustained notes, leave space between phrases.
        if phase==1 and same_prev:
            continue
        if start<16 and i%2 and same_prev:
            continue
        length=.87*beat
        if phase==0 and same_next:
            length=1.65*beat
        elif phase in [3,7]:
            length=.57*beat
        elif phase==2:
            length=.68*beat
        velocity=int(np.clip(n['velocity']+rng.integers(-5,6)+(5 if phase in [0,4] else -2),43,99))
        delayed=start+float(rng.uniform(.003,.018))
        notes.append({'start':delayed,'end':min(duration,delayed+length),'midi':pitch,'velocity':velocity,'articulation':'sustain' if length>beat else 'finger'})
        if start>16 and same_next and phase in [2,5,7] and beat>.28:
            # A soft offbeat rearticulation, or a chord-safe fifth; no random fills.
            fifth=pitch+7 if pitch+7<=48 else pitch-5
            extra_pitch=fifth if phase==5 else pitch
            extra_start=start+beat*(.73 if phase==7 else .76)+float(rng.uniform(-.006,.006))
            notes[-1]['end']=min(notes[-1]['end'],extra_start-.035)
            notes.append({'start':extra_start,'end':min(duration,end-.025),'midi':extra_pitch,'velocity':max(36,velocity-(19 if phase==7 else 12)),'articulation':'soft_offbeat'})
    notes.sort(key=lambda n:n['start'])
    for a,b in zip(notes,notes[1:]):
        a['end']=min(a['end'],b['start']-.012)
    return [n for n in notes if n['end']-n['start']>.025]


def write_midi(notes, path):
    mid=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack();mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',tempo=500000))
    track.append(mido.Message('program_change',program=33))
    events=[]
    for n in notes:
        for t,kind in [(n['start'],'note_on'),(n['end'],'note_off')]:
            events.append((round(t*960),kind,n))
    last=0
    for tick,kind,n in sorted(events,key=lambda e:(e[0],e[1]=='note_on')):
        track.append(mido.Message(kind,note=n['midi'],velocity=n['velocity'] if kind=='note_on' else 0,time=tick-last));last=tick
    mid.save(path)


def process(analysis, source, out, soundfont):
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads(analysis.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report['source_sha256']:
        raise ValueError('Source does not match original harmony analysis')
    notes=arrange(report)
    write_midi(notes,out/'bass.mid')
    print(f'Render {len(notes)} phrased notes with {soundfont.name}',flush=True)
    render(notes,report['duration_seconds'],out/'bass_dry.wav',soundfont,dry=True)
    # Restrained DI/amp treatment: remove subsonics, retain finger attack, mild drive.
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'bass_dry.wav'),'-af','highpass=f=35,equalizer=f=100:t=q:w=0.8:g=2,equalizer=f=750:t=q:w=1:g=2,lowpass=f=4200,acompressor=threshold=0.18:ratio=2.5:attack=15:release=120:makeup=1.3,asoftclip=type=tanh:threshold=0.85:output=0.95','-c:a','pcm_s24le',str(out/'bass.wav')],check=True)
    # Match isolated bass RMS to the previous demo for a more useful comparison.
    audio,sr=sf.read(out/'bass.wav');target=.1286
    gain=min(target/(np.sqrt(np.mean(audio**2))+1e-9),.8/(np.max(np.abs(audio))+1e-9))
    sf.write(out/'bass.wav',audio*gain,sr,subtype='PCM_24')
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-i',str(out/'bass.wav'),'-filter_complex','[0:a]volume=0.78[p];[1:a]volume=0.55[b];[p][b]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.92:level=0:latency=1','-ar','44100','-c:a','pcm_s24le',str(out/'piano_with_bass.wav')],check=True)
    for name in ['bass','piano_with_bass']:
        subprocess.run(['ffmpeg','-v','error','-i',str(out/f'{name}.wav'),'-c:a','libmp3lame','-b:a','192k',str(out/f'{name}.mp3')],check=True)
    subprocess.run(['ffmpeg','-v','error','-ss','24','-i',str(out/'piano_with_bass.wav'),'-t','30','-c:a','libmp3lame','-b:a','192k',str(out/'preview_30s.mp3')],check=True)
    report.update(notes=notes,note_count=len(notes),arrangement='phrased_finger_v1',seed=24,soundfont=str(soundfont),soundfont_sha256=hashlib.sha256(soundfont.read_bytes()).hexdigest(),analysis_source=str(analysis))
    report.pop('render',None)
    report['limitations']='Preserved automatic chord estimates; heuristic phrasing on an assumed eight-beat cycle. Sampled GM instrument, without recorded slides, hammer-ons or round-robin articulations. Musical realism requires listening review.'
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    print('Finished',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['analysis','source','out','soundfont']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    process(args.analysis.resolve(),args.source.resolve(),args.out.resolve(),args.soundfont.resolve())
