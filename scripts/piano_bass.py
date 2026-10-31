"""CPU piano harmony analysis -> editable MIDI -> sampled electric bass.

Dependencies: librosa, numpy, scipy, soundfile, mido; ffmpeg and libfluidsynth.
Chord labels are estimates, not a verified transcription. No model downloads.
"""
import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
from pathlib import Path
import subprocess

import librosa
import mido
import numpy as np
import soundfile as sf


CHORD_TYPES = [('', (0, 4, 7)), ('m', (0, 3, 7))]


def chord_templates():
    templates = []
    for _, intervals in CHORD_TYPES:
        for root in range(12):
            t = np.full(12, -.25)
            for interval in intervals:
                t[(root+interval)%12] = 1.
            templates.append(t/np.linalg.norm(t))
    return np.array(templates)


def render(notes, duration, destination, soundfont, dry=False):
    lib = ct.CDLL(ctypes.util.find_library('fluidsynth'))
    signatures = {
        'new_fluid_settings': (ct.c_void_p, []),
        'fluid_settings_setnum': (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_double]),
        'fluid_settings_setint': (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
        'new_fluid_synth': (ct.c_void_p, [ct.c_void_p]),
        'fluid_synth_sfload': (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
        'fluid_synth_program_change': (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int]),
        'fluid_synth_noteon': (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]),
        'fluid_synth_noteoff': (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int]),
        'fluid_synth_write_float': (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int]),
        'delete_fluid_synth': (None, [ct.c_void_p]),
        'delete_fluid_settings': (None, [ct.c_void_p]),
    }
    for name, (restype, argtypes) in signatures.items():
        getattr(lib, name).restype = restype
        getattr(lib, name).argtypes = argtypes
    settings = lib.new_fluid_settings()
    lib.fluid_settings_setnum(settings, b'synth.sample-rate', 44100.)
    lib.fluid_settings_setnum(settings, b'synth.gain', .65)
    if dry:
        lib.fluid_settings_setint(settings, b'synth.reverb.active', 0)
        lib.fluid_settings_setint(settings, b'synth.chorus.active', 0)
    synth = lib.new_fluid_synth(settings)
    try:
        if lib.fluid_synth_sfload(synth, str(soundfont).encode(), 1) < 0:
            raise RuntimeError('SoundFont loading failed')
        lib.fluid_synth_program_change(synth, 0, 33)  # GM Electric Bass (finger)
        total = round(duration * 44100)
        audio = np.zeros((total, 2), dtype=np.float32)
        events = []
        for n in notes:
            events.extend([(round(n['start']*44100), 1, n), (round(n['end']*44100), 0, n)])
        cursor = 0
        for frame, on, n in sorted(events, key=lambda e: (e[0], e[1])) + [(total, 0, None)]:
            frame = min(frame, total)
            if frame > cursor:
                chunk = np.empty((frame-cursor, 2), dtype=np.float32)
                status = lib.fluid_synth_write_float(synth, len(chunk), chunk.ctypes.data, 0, 2, chunk.ctypes.data, 1, 2)
                if status < 0:
                    raise RuntimeError('SoundFont rendering failed')
                audio[cursor:frame] = chunk
                cursor = frame
            if n:
                if on:
                    lib.fluid_synth_noteon(synth, 0, n['midi'], n['velocity'])
                else:
                    lib.fluid_synth_noteoff(synth, 0, n['midi'])
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
        peak = float(np.max(np.abs(audio)))
        audio *= .70/peak
        sf.write(destination, audio, 44100, subtype='PCM_24')
        return {'raw_peak': peak, 'saved_peak': float(np.max(np.abs(audio)))}
    finally:
        lib.delete_fluid_synth(synth)
        lib.delete_fluid_settings(settings)


def process(source, out, soundfont):
    out.mkdir(parents=True, exist_ok=False)
    print('Decode and analyze piano', flush=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-ac', '1', '-ar', '22050', str(out/'analysis.wav')], check=True)
    y, sr = sf.read(out/'analysis.wav')
    duration = len(y)/sr
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, trim=False)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(beats) < 8:
        raise RuntimeError('Insufficient detected beats; needs manual timing')
    period = float(np.median(np.diff(beats)))
    print(f'Estimated tempo {float(np.asarray(tempo).item()):.1f} BPM; {len(beats)} beats', flush=True)
    # CQT separates note octaves. Compress energy and reduce harmonic overtones.
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz('C1'), n_bins=84))
    salience = cqt.copy()
    for offset, weight in [(12,.30), (19,.15), (24,.10)]:
        salience[offset:] -= weight*cqt[:-offset]
    salience = np.maximum(salience, 0)**.7
    chroma = salience.reshape(7,12,-1).sum(axis=0)
    low = salience[:36].reshape(3,12,-1).sum(axis=0)
    # Equal chord-tone weights avoid mistaking the loudest melody note for root.
    templates = chord_templates()
    state_count = len(templates)
    boundaries = np.r_[beats, duration]
    boundaries = np.unique(np.clip(boundaries,0,duration))
    scores, strengths = [], []
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        a,b = round(start*sr/hop), max(round(start*sr/hop)+1, round(end*sr/hop))
        c = chroma[:,a:b].mean(axis=1)
        # Remove the broad spectral floor before evaluating note membership.
        c = np.maximum(c - np.percentile(c, 35), 0)
        l = low[:,a:b].mean(axis=1)
        bass_weight = .35*min(1., float(np.sum(l)/(np.sum(c)+1e-8))*3)
        scores.append(templates@(c/(np.linalg.norm(c)+1e-8)) + bass_weight*np.tile(l/(np.max(l)+1e-8),len(CHORD_TYPES)))
        strengths.append(float(np.sqrt(np.mean(y[round(start*sr):round(end*sr)]**2))))
    scores = np.array(scores)
    transition = np.full((state_count,state_count), -.22)
    np.fill_diagonal(transition,0)
    dp = scores[0].copy(); back = []
    for score in scores[1:]:
        candidate = dp[:,None]+transition
        back.append(candidate.argmax(axis=0))
        dp = candidate.max(axis=0)+score
    states = [int(dp.argmax())]
    for prev in reversed(back): states.append(int(prev[states[-1]]))
    states.reverse()
    names = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    notes, chords = [], []
    threshold = max(.002, np.percentile(strengths,75)*.06)
    previous = 40
    for i,(start,end,state,energy) in enumerate(zip(boundaries[:-1],boundaries[1:],states,strengths)):
        root = state%12
        ranked = np.sort(scores[i])
        confidence = float(ranked[-1]-ranked[-2])
        chords.append({'start':float(start),'end':float(end),'chord':names[root]+CHORD_TYPES[state//12][0],'score_margin':confidence})
        if energy < threshold: continue
        # Conservative root-led part. Fifth only on stable harmony with piano evidence.
        pitchclass = root
        if i%4 == 2 and i>0 and states[i-1]==state:
            a,b = round(start*sr/hop), max(round(start*sr/hop)+1,round(end*sr/hop))
            c=chroma[:,a:b].mean(axis=1)
            if c[(root+7)%12] > .7*c[root]: pitchclass=(root+7)%12
        choices = [n for n in range(28,49) if n%12==pitchclass]
        pitch = min(choices,key=lambda n:abs(n-previous)+.25*abs(n-38))
        previous=pitch
        velocity=int(np.clip(69+12*np.log2(energy/(np.median(strengths)+1e-8)+1e-8),48,92))
        notes.append({'start':float(start),'end':float(start+.84*(end-start)),'midi':pitch,'velocity':velocity})
    mid=mido.MidiFile(ticks_per_beat=480); track=mido.MidiTrack(); mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo',tempo=500000,time=0))
    track.append(mido.Message('program_change',program=33,time=0))
    events=[]
    for n in notes:
        events.extend([(round(n['start']*960),'note_on',n),(round(n['end']*960),'note_off',n)])
    last=0
    for tick,kind,n in sorted(events,key=lambda e:(e[0],e[1]=='note_on')):
        track.append(mido.Message(kind,note=n['midi'],velocity=n['velocity'] if kind=='note_on' else 0,time=tick-last)); last=tick
    mid.save(out/'bass.mid')
    print(f'Rendering {len(notes)} bass notes',flush=True)
    peak=render(notes,duration,out/'bass.wav',soundfont)
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-i',str(out/'bass.wav'),'-filter_complex','[0:a]volume=0.78[p];[1:a]volume=0.55[b];[p][b]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:latency=1','-ar','44100','-c:a','pcm_s24le',str(out/'piano_with_bass.wav')],check=True)
    for name in ['bass','piano_with_bass']:
        subprocess.run(['ffmpeg','-v','error','-i',str(out/f'{name}.wav'),'-c:a','libmp3lame','-b:a','192k',str(out/f'{name}.mp3')],check=True)
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'duration_seconds':duration,'estimated_bpm':float(np.asarray(tempo).item()),'beat_count':len(beats),'note_count':len(notes),'soundfont':str(soundfont),'soundfont_sha256':hashlib.sha256(soundfont.read_bytes()).hexdigest(),'render':peak,'limitations':'Automatic chord estimates, not verified transcription; no meter or downbeat detection. Relative score margins are not calibrated probabilities. Sampled GM finger bass, not a recorded performance.','chords':chords,'notes':notes}
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['chords','notes']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--soundfont',type=Path,default=Path('/usr/share/sounds/sf2/TimGM6mb.sf2'))
    args=parser.parse_args()
    process(args.source.resolve(),args.out.resolve(),args.soundfont.resolve())
