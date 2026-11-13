"""Normalized overlap-add for fixed30s generated windows; no model/audio filtering.

This reconstructs a timeline from overlapping outputs. Synthetic reconstruction
tests do not establish musical continuity between independently generated clips.
"""
import math


def validate_windows(windows, target_samples):
    if type(target_samples) is not int or target_samples <= 0 or not windows:
        raise ValueError('A positive timeline and nonempty window schedule are required')
    previous = None
    covered = 0
    for i, window in enumerate(windows):
        a,b,pad = window['sample_start48'],window['sample_end48'],window['right_padding_samples48']
        if (type(window['index']) is not int or window['index'] != i or any(type(v) is not int for v in (a,b,pad))
                or a < 0 or b <= a or b > target_samples or pad < 0 or b-a+pad != 1440000
                or (previous is not None and a <= previous) or a > covered):
            raise ValueError('Invalid, unordered or uncovered30s window schedule')
        previous,covered = a,max(covered,b)
    if covered != target_samples: raise ValueError('Schedule does not reach exact song end')


def window_weights(windows, index):
    import numpy as np
    window = windows[index]
    a,b = window['sample_start48'],window['sample_end48']
    result = np.ones(b-a,dtype=np.float64)
    # Half-sample cosine ramps have positive endpoint weights. Multiplication
    # also supports three-way overlap; final normalization is always necessary.
    if index:
        overlap = min(b,windows[index-1]['sample_end48'])-a
        if overlap > 0:
            position = (np.arange(overlap,dtype=np.float64)+.5)/overlap
            result[:overlap] *= np.sin(.5*math.pi*position)**2
    if index+1 < len(windows):
        overlap = b-max(a,windows[index+1]['sample_start48'])
        if overlap > 0:
            position = (np.arange(overlap,dtype=np.float64)+.5)/overlap
            result[-overlap:] *= np.cos(.5*math.pi*position)**2
    if not np.isfinite(result).all() or not (result > 0).all(): raise ValueError('Invalid overlap weights')
    return result


class OverlapAssembler:
    def __init__(self, windows, target_samples):
        import numpy as np
        validate_windows(windows,target_samples)
        self.windows = [dict(window) for window in windows]
        self.target_samples = target_samples
        self.total = np.zeros((2,target_samples),dtype=np.float64)
        self.weight = np.zeros(target_samples,dtype=np.float64)
        self.completed = set()

    def add(self,index,audio):
        import numpy as np
        if type(index) is not int or not 0 <= index < len(self.windows) or index in self.completed:
            raise ValueError('Invalid or duplicate window; existing accumulator preserved')
        audio = np.asarray(audio)
        if audio.shape != (2,1440000) or not np.isfinite(audio).all():
            raise ValueError('Expected finite planar stereo30s output')
        window = self.windows[index]
        a,b = window['sample_start48'],window['sample_end48']
        weights = window_weights(self.windows,index)
        self.total[:,a:b] += audio[:,:b-a].astype(np.float64)*weights
        self.weight[a:b] += weights
        self.completed.add(index)

    def finish(self):
        import numpy as np
        if len(self.completed) != len(self.windows) or not (self.weight > 0).all():
            raise ValueError('Missing windows or uncovered samples; no completed output')
        value = (self.total/self.weight).astype(np.float32)
        if not np.isfinite(value).all(): raise ValueError('Nonfinite assembled output')
        return value
