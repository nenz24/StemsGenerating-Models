"""Supported fixed-shape preview contracts; default remains the 5.12 s baseline."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PreviewWindow:
    frames: int
    instrument: str = 'bass'

    def __post_init__(self):
        if type(self.frames) is not int or self.frames not in (128, 256):
            raise ValueError('Only 128/256 latent frames (5.12/10.24 s) are supported')
        if self.instrument not in ('bass','drums','guitar-strum','guitar-pick') or (self.frames == 128 and self.instrument != 'bass'):
            raise ValueError('Instrument preview requires an allowlisted 10.24-second profile')

    @property
    def weight_storage_dtype(self): return 'float16' if self.frames == 128 else 'bfloat16'
    @property
    def apg_projection_precision(self):
        return 'float32_validated_against_original_float64'
    @property
    def samples(self): return self.frames * 1920
    @property
    def seconds(self): return self.samples / 48000
    @property
    def hidden_tokens(self): return self.frames // 2
    @property
    def text_tokens(self): return 105 if self.frames == 128 else (111 if self.instrument in ('drums','guitar-pick') else 106)
    @property
    def condition_tokens(self): return self.text_tokens + 11 + 1
    @property
    def suffix(self): return '512' if self.frames == 128 else '1024'
    @property
    def source_bundle(self): return 'source-context' if self.frames == 128 else 'source-context-1024'
    @property
    def condition_bundle(self): return 'bass-conditioning' if self.frames == 128 else self.instrument+'-conditioning-1024'
    @property
    def preview_bundle(self): return self.instrument+'-preview-' + self.suffix
    @property
    def resource_profile(self): return 'bass-source' if self.frames == 128 else self.instrument+'-source-1024'

    @property
    def preset_file(self): return {'bass':'electric-bass.json','drums':'acoustic-drums.json','guitar-strum':'clean-electric-guitar-strum-layer.json','guitar-pick':'clean-electric-guitar-pick-layer.json'}[self.instrument]
    @property
    def instruction(self): return 'Generate the '+('guitar' if self.instrument.startswith('guitar-') else self.instrument)+' track based on the audio context:'
    @property
    def preset_sha256(self): return {'bass':'4bacacb4f9fb1a9e66c98958869e2756878e6420910b2af065bea6522cb6ef63','drums':'bcfd73641fddbda1fb0975febfe28cccb3a3121ec48cec1d877933e64a6d70ca','guitar-strum':'885c989481abca270fcea2baa77cd843c73efcde4ad3e2f6d42ad9eeee05d104','guitar-pick':'7aaff6ec3d7fb3a8c439d4b6d9e2baca3edc65835e3b59e64891d9b6694303bb'}[self.instrument]

    @classmethod
    def for_seconds(cls, seconds, instrument='bass'):
        if type(seconds) not in (int, float) or seconds not in (5.12, 10.24):
            raise ValueError('Preview duration must be 5.12 or 10.24 seconds')
        return cls(128 if seconds == 5.12 else 256, instrument)

    @classmethod
    def for_bundle(cls, profile, kind):
        candidates = [SHORT] + [cls(256,i) for i in ('bass','drums','guitar-strum','guitar-pick')]
        return next((w for w in candidates if getattr(w,kind)==profile),None)

SHORT = PreviewWindow(128)
LONG = PreviewWindow(256)
