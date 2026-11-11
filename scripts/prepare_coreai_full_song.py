#!/usr/bin/env python3
"""Prepare full-song source PCM and a bounded scheduling plan; never run models.

Default prints a plan only. --execute decodes source once with the existing
gain0.7/32kHz/stereo/nearest-even frontend. Existing baselines stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT=Path(__file__).resolve().parents[1]
INSTRUMENTS=('bass','drums','guitar')


def schedule(frames32):
    if type(frames32) is not int or frames32<=0:
        raise ValueError('Positive exact32kHz PCM frame count required')
    samples48=(frames32*3+1)//2
    latent_frames=(samples48+1919)//1920
    span=max(0,latent_frames-750)
    count=(span+549)//550+1
    starts=[i*span//(count-1) for i in range(count)] if count>1 else [0]
    windows=[]
    for i,start in enumerate(starts):
        end=start+750
        windows.append({'index':i,'latent_start':start,'latent_end':end,
                        'sample_start48':start*1920,'sample_end48':min(end*1920,samples48),
                        'right_padding_samples48':max(0,end*1920-samples48),
                        'source_valid_frames':max(0,min(end,latent_frames)-start),
                        'right_padding_latent_frames':max(0,end-latent_frames),
                        'overlap_with_previous_frames':0 if i==0 else starts[i-1]+750-start})
    runs=count*len(INSTRUMENTS)
    return {'schema_version':1,'status':'PLAN ONLY; no model execution',
            'frames32':frames32,'target_samples48':samples48,'latent_frames':latent_frames,
            'seconds':frames32/32000,'instruments':list(INSTRUMENTS),'window_frames':750,
            'minimum_overlap_frames':200,'windows_per_instrument':count,'windows':windows,
            'proposed_order':'one instrument at a time, windows in increasing order; shared sampler weights',
            'proposed_calls':{'generation_runs':runs,'sampling_steps':50*runs,
                              'DiT_partition_calls':700*runs,'APG_calls':50*runs,'window_decoder_calls':6*runs},
            'planned_lifecycle':{'maximum_sampling_handles':15,'release_sampler_before_decoder':True,
                                 'one_retained_decoder_asset':True,'process_guard_mib':2000,
                                 'native_eviction_guaranteed':False,'full_song_peak_measured':False},
            'precision':{'DiT_learned_storage':'bfloat16','compute_APG_IO_VAE':'float32'},
            'tolerances':{'absolute':1e-4,'relative':1e-3,'normalized_rmse':1e-3},
            'overlap_policy':'Future normalized overlap-add on raw outputs; retain exact clip length. No claim of musical seam acceptance.',
            'noise_policy':'Capture actual global per-instrument noise and source epsilon arrays; slice overlap consistently. Seed alone is not cross-platform provenance.',
            'gain_policy':'Source gain0.7 once; compare raw outputs before one shared whole-song listening gain, never per-window automatic gain.',
            'latency_estimate':None,'latency_scope':'No linear extrapolation from short independent runs; measure the full workload.',
            'ready_for_native_execution':False,'baseline_modified':False,
            'unmet_gates':['original full50-step numerical gate remains independent and is not overridden',
                           'bounded native source encoder on changing full-song input with verified context and epsilon',
                           'bass/drums/guitar conditioning compatible with the fixed127-token contract',
                           'actual global noise/epsilon captures and per-window provenance',
                           'normalized overlap-add validation and listening across seams',
                           'full-song resource, thermal and latency measurements']}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def pcm_prefix(path,limit):
    with wave.open(str(path),'rb') as source:
        if (source.getnchannels(),source.getsampwidth(),source.getframerate())!=(2,2,32000):
            raise ValueError('Baseline prefix must be stereo32kHz PCM16')
        count=source.getnframes()
        if count>limit:raise ValueError('Baseline prefix exceeds prepared song')
        return source.readframes(count),count


def prepare(source,out,baseline_prefix=None):
    import numpy as np
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Existing full-song preparation preserved: '+str(out))
    if not source.is_file():raise FileNotFoundError(source)
    if any(out.is_relative_to(p) for p in (source.parent,ROOT/'baselines',ROOT/'prototypes/ios27-coreai/ModelCandidates')):
        raise ValueError('Use an inactive output outside source/baselines/models')
    source_sha=sha256(source)
    out.mkdir(parents=True)
    report={'schema_version':1,'status':'preparing','source_sha256':source_sha,'source_gain':.7,
            'baseline_modified':False,'model_calls':0,'native_exports':0,'profile_activated':False}
    def save():
        temp=out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'report.json')
    save()
    try:
        # Stream to disk, then quantize bounded blocks. No full waveform array.
        with tempfile.TemporaryFile() as errors:
            process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(source),'-ss','0',
                '-af','volume=0.7','-ar','32000','-ac','2','-f','f32le','pipe:1'],stdout=subprocess.PIPE,stderr=errors)
            frames=0;float_digest=hashlib.sha256();pcm_digest=hashlib.sha256();pending=b''
            try:
                with (out/'source32.wav').open('xb') as target:
                    with wave.open(target,'wb') as wav:
                        wav.setparams((2,2,32000,0,'NONE','not compressed'))
                        while chunk:=process.stdout.read(1024*1024):
                            float_digest.update(chunk)
                            pending+=chunk;length=len(pending)//8*8
                            if not length:continue
                            values=np.frombuffer(pending[:length],dtype='<f4')
                            if not np.isfinite(values).all():raise ValueError('Nonfinite decoded source')
                            pcm=np.rint(values.astype(np.float64)*32768).clip(-32768,32767).astype('<i2').tobytes()
                            wav.writeframesraw(pcm);pcm_digest.update(pcm);frames+=len(pcm)//4
                            pending=pending[length:]
                code=process.wait()
                if code or pending or frames<=0:
                    errors.seek(0)
                    raise ValueError('Invalid FFmpeg source decode: '+errors.read(4096).decode(errors='replace'))
            finally:
                process.stdout.close()
                if process.poll() is None:process.terminate();process.wait()
        if sha256(source)!=source_sha:raise ValueError('Source changed during preparation')
        report.update(source32_frames=frames,source32_float_sha256=float_digest.hexdigest(),
                      source32_pcm_sha256=pcm_digest.hexdigest(),source32_wav_sha256=sha256(out/'source32.wav'),
                      quantization='np.rint(Float64(Float32)*32768), nearest-even; clip[-32768,32767]',
                      baseline_prefix_comparison='NOT EVALUATED')
        if baseline_prefix is not None:
            expected,count=pcm_prefix(baseline_prefix,frames)
            with wave.open(str(out/'source32.wav'),'rb') as prepared:actual=prepared.readframes(count)
            report['baseline_prefix_comparison']='PASS' if actual==expected else 'NOT PASSED'
            report['baseline_prefix_frames']=count
            report['baseline_prefix_pcm_sha256']=hashlib.sha256(actual).hexdigest()
            if actual!=expected:raise ValueError('Full-source prefix differs from unchanged baseline; no models executed')
        plan=schedule(frames)
        (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        report.update(status='source_and_schedule_prepared',generation_status='NOT EXECUTED',
                      full_song_numerical_parity='NOT EVALUATED',ready_for_native_execution=False,
                      windows_per_instrument=plan['windows_per_instrument'],generation_runs=plan['proposed_calls']['generation_runs'])
        save();return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames32',type=int,help='Exact decoded stereo32kHz frame count for a read-only schedule')
    source_group=parser.add_mutually_exclusive_group()
    source_group.add_argument('--source',type=Path)
    source_group.add_argument('--choose-source',action='store_true',help='Choose input with a macOS file dialog; no typed source path needed')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--baseline-prefix',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        if args.frames32 is not None:print(json.dumps(schedule(args.frames32),indent=2))
        else:print('PLAN ONLY: full source32 PCM + fixed750-frame bass/drums/guitar schedule. Zero model calls; --execute requires --source or --choose-source, and a new --out.')
        return
    if args.out is None:parser.error('--execute requires a new --out')
    if args.choose_source:
        if sys.platform!='darwin':parser.error('--choose-source requires macOS; use --source elsewhere')
        choice=subprocess.run(['osascript','-e','POSIX path of (choose file with prompt "Pilih audio sumber lagu penuh")'],
                              text=True,capture_output=True)
        if choice.returncode:parser.exit(1,'Source selection canceled or failed; no output created.\n')
        args.source=Path(choice.stdout.rstrip('\r\n'))
    if args.source is None:parser.error('--execute requires --source or --choose-source')
    report=prepare(args.source,args.out,args.baseline_prefix)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
