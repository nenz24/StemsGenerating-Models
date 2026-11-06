from pathlib import Path
import io,json,zipfile,os
os.environ['MPLCONFIGDIR']='/tmp/drum-apg-matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft,welch,resample_poly,find_peaks
p=Path('work/drum-apg-waveforms');r=json.loads((p/'report.json').read_text());gain=r['common_listening_gain'];fs=48000;n=491520
raw=np.fromfile(p/'candidate-audio.f32le',dtype='<f4').reshape(2,n).T.astype(np.float64)
control=wavfile.read(p/'01-control-original-apg-headroom.wav')[1].astype(np.float64)/gain
source=np.fromfile('work/drum-waveforms/audio.f32le',dtype='<f4').reshape(2,n).T.astype(np.float64)
with zipfile.ZipFile('baselines/acoustic-drums-30-v1/drum_result.zip') as z:
 sr,a=wavfile.read(io.BytesIO(z.read('drums.wav')));approved=resample_poly(a.astype(np.float64)/32768,3,2)[:n]
signals={'source_piano':source,'approved_30s_first10.24':approved,'control_apg2.5':control,'candidate_apg1.46059':raw}
results={}
fig,axes=plt.subplots(4,2,figsize=(15,12),layout='constrained')
for row,(label,x) in enumerate(signals.items()):
 mono=x.mean(axis=1);rms=float(np.sqrt(np.mean(x*x)));delta=np.max(abs(np.diff(x,axis=0)),axis=1)
 # All bands below16k to avoid interpreting old32k export's cutoff as an artifact.
 f,power=welch(x,fs,nperseg=8192,axis=0);power=power.sum(axis=1)
 energy={f'{lo}-{hi}':float(power[(f>=lo)&(f<hi)].sum()/power[(f<16000)].sum()) for lo,hi in [(0,150),(150,1000),(1000,5000),(5000,16000)]}
 block=x.reshape(512,960,2);envelope=np.sqrt(np.mean(block**2,axis=(1,2)))
 derivative=np.diff(x,axis=0);phase40=np.mean(derivative[:(len(derivative)//1920)*1920].reshape(-1,1920,2)**2,axis=(0,2))
 phase80=np.mean(derivative[:(len(derivative)//3840)*3840].reshape(-1,3840,2)**2,axis=(0,2))
 peaks,_=find_peaks(envelope,distance=4,prominence=.1*envelope.max())
 results[label]={'rms':rms,'peak':float(abs(x).max()),'adjacent_delta_max':float(delta.max()),'largest_delta_seconds':float((delta.argmax()+1)/fs),'band_energy_fraction_below16k':energy,'repeated_frames_exact_count':int(np.all(np.diff(x,axis=0)==0,axis=1).sum()),'zero_frames':int(np.all(x==0,axis=1).sum()),'phase40ms_derivative_max_to_median':float(phase40.max()/np.median(phase40)),'phase80ms_derivative_max_to_median':float(phase80.max()/np.median(phase80)),'envelope_peak_times_seconds':((peaks+.5)*.02).tolist()}
 axes[row,0].plot(np.arange(n)/fs,mono,linewidth=.3);axes[row,0].set_title(label);axes[row,0].set_xlim(0,10.24);axes[row,0].set_ylabel('Float32 amplitude')
 f,t,S=stft(mono,fs,nperseg=1024,noverlap=768)
 im=axes[row,1].pcolormesh(t,f,20*np.log10(np.maximum(abs(S)/max(rms,1e-12),1e-7)),vmin=-65,vmax=-10,shading='auto',cmap='magma');axes[row,1].set_ylim(0,16000);axes[row,1].set_ylabel('Hz');axes[row,1].set_title('STFT relative to own RMS (descriptive only)')
fig.colorbar(im,ax=axes[:,1],label='dB relative to own RMS');fig.savefig(p/'waveform-spectra.png',dpi=130)
# Compare sub-millisecond transients at largest change, without assuming they are clicks.
center=results['candidate_apg1.46059']['largest_delta_seconds'];start=max(0,center-.004);end=min(10.24,center+.004)
fig,axes=plt.subplots(3,1,figsize=(14,8),layout='constrained')
for ax,(label,x) in zip(axes,[('control',control),('candidate',raw),('piano',source)]):
 a,b=int(start*fs),int(end*fs);ax.plot(np.arange(a,b)/fs,x[a:b,0],label='L');ax.plot(np.arange(a,b)/fs,x[a:b,1],label='R',alpha=.65);ax.set_title(label);ax.legend();ax.grid(alpha=.2)
fig.savefig(p/'largest-transition-zoom.png',dpi=130)
z=np.fromfile(p/'candidate-latent.f32le',dtype='<f4').reshape(256,64).astype(np.float64)
results['candidate_latent']={'min':float(z.min()),'max':float(z.max()),'mean':float(z.mean()),'std':float(z.std()),'interframe_rms':float(np.sqrt(np.mean(np.diff(z,axis=0)**2))),'finite':bool(np.isfinite(z).all())}
results['scope']='Descriptive analysis, not a click detector. Natural percussion has sharp changes and noise. Colab is a different30s context, input preparation and RNG; no exact musical reference/parity. Phase-fold measures can reflect chance/percussion periodicity, not proof of frame-boundary defects.'
(p/'attack-audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
