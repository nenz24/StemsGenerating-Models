"""Small stdlib-only Float32 listening helpers; never modify model references."""
import array
import math
from pathlib import Path
import struct
import sys


def read_planar(path, frames):
    if type(frames) is not int or frames<=0:raise ValueError('Invalid frame count')
    data=Path(path).read_bytes()
    if len(data)!=8*frames: raise ValueError('Unexpected stereo Float32 reference size')
    values=array.array('f');values.frombytes(data)
    if sys.byteorder!='little':values.byteswap()
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite reference audio')
    return values


def headroom_gain(peak):
    if not math.isfinite(peak) or peak<0:raise ValueError('Invalid peak')
    return min(1.0,10**(-3/20)/peak) if peak else 1.0


def write_float_wav(path, planar, frames, gain=1.0):
    if (type(frames) is not int or frames<=0 or len(planar)!=2*frames or not math.isfinite(gain) or not 0<gain<=1
            or not all(math.isfinite(v) for v in planar)):
        raise ValueError('Invalid listening WAV data/gain')
    interleaved=array.array('f')
    for i in range(frames):interleaved.extend((planar[i]*gain,planar[frames+i]*gain))
    if sys.byteorder!='little':interleaved.byteswap()
    pcm=interleaved.tobytes()
    header=(b'RIFF'+struct.pack('<I',50+len(pcm))+b'WAVEfmt '+struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0)
            +b'fact'+struct.pack('<II',4,frames)+b'data'+struct.pack('<I',len(pcm)))
    with Path(path).open('xb') as stream:stream.write(header);stream.write(pcm)
