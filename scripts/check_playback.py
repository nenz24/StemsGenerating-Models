"""Make low-level/sub-bass playback comparisons from cached audio, without model inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/stem_eval'))
from audio import RATE,canonicalize,read_pcm
from provenance import sha256,write_json
from system_tools import require_tool
from export_glitch_feedback import latest_comparison


def make_previews(run,output):
    """Preserve the reference, attenuate 18 dB, and compare removal of sub-bass at the same gain."""
    capture=json.loads((run/'capture/capture.json').read_text())
    raw=run/'capture/default_float.wav'
    if capture.get('status')!='captured' or sha256(raw)!=capture['default_float_sha256']:
        raise ValueError('Cached default decoder output changed')
    output.mkdir(parents=True,exist_ok=False)
    reference=output/'reference.wav'
    canonicalize(raw,reference,start=3,duration=10,gain=0.7)
    gain=10**(-18/20)
    canonicalize(reference,output/'quiet_18db.wav',duration=10,gain=gain)
    subprocess.run([require_tool('ffmpeg'),'-nostdin','-v','error','-n','-i',str(reference),'-t','10',
        '-af',f'highpass=f=80:p=2,volume={gain}','-ar',str(RATE),'-ac','2','-c:a','pcm_s16le',
        str(output/'highpass80_quiet.wav')],check=True)
    if any(len(read_pcm(output/name))!=2*10*RATE for name in ('reference.wav','quiet_18db.wav','highpass80_quiet.wav')):
        raise ValueError('Playback comparison lengths differ')
    write_json(output/'report.json',{'source_sha256':sha256(raw),'start_in_B_seconds':3,'duration':10,
        'reference_gain':0.7,'extra_gain_both_tests':gain,'highpass_hz':80,
        'note':'Diagnostic only. High-pass removes wanted bass too. Keep device volume unchanged. This does not prove speaker distortion or repair generation.'})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    run=args.run.resolve() if args.run else latest_comparison()
    output=args.out.resolve() if args.out else run/('playback_'+uuid.uuid4().hex[:8])
    make_previews(run,output)
    print('Folder:',output)
    print('Bandingkan reference.wav, quiet_18db.wav, highpass80_quiet.wav dengan volume perangkat tetap.')
    print('Coba file yang sama melalui headphone bila tersedia. High-pass hanya diagnosis, bukan hasil bass final.')
    print('Tidak ada inferensi, unduhan, atau perubahan hasil lama.')


if __name__=='__main__':main()
