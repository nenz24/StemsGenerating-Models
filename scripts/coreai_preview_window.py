"""Supported fixed-shape preview contracts; default remains the 5.12 s baseline."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PreviewWindow:
    frames: int

    def __post_init__(self):
        if type(self.frames) is not int or self.frames not in (128, 256):
            raise ValueError('Only 128/256 latent frames (5.12/10.24 s) are supported')

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
    def text_tokens(self): return 105 if self.frames == 128 else 106
    @property
    def condition_tokens(self): return self.text_tokens + 11 + 1
    @property
    def suffix(self): return '512' if self.frames == 128 else '1024'
    @property
    def source_bundle(self): return 'source-context' if self.frames == 128 else 'source-context-1024'
    @property
    def condition_bundle(self): return 'bass-conditioning' if self.frames == 128 else 'bass-conditioning-1024'
    @property
    def preview_bundle(self): return 'bass-preview-' + self.suffix
    @property
    def resource_profile(self): return 'bass-source' if self.frames == 128 else 'bass-source-1024'

    @classmethod
    def for_seconds(cls, seconds):
        if type(seconds) not in (int, float) or seconds not in (5.12, 10.24):
            raise ValueError('Preview duration must be 5.12 or 10.24 seconds')
        return cls(128 if seconds == 5.12 else 256)

SHORT = PreviewWindow(128)
LONG = PreviewWindow(256)
