"""Offline comparison only: no source admission, DiT, decoder or activation."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import torch

root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from generate_coreai_drum_reference import pcm16, resample
from compare_coreai_drum_source_gain import load_encoder
from compare_coreai_drum_lyrics import local_checkpoint
from export_coreai_source_context import posterior_context
from export_coreai_dit_block import precision_metrics

torch.set_num_threads(2)
original = root / 'work/drum-reference30-v1/frontend'
uploaded = root / 'work/drum-mac-source-pcm-review-v1/source32.wav'
evidence = json.loads((root / 'reviews/coreai-drums-reference30-pcm-review/source-sensitivity.json').read_text())
pcm = pcm16(uploaded.read_bytes())
assert hashlib.sha256(pcm).hexdigest() == evidence['Mac_PCM_sha256']
for name, expected in evidence['reference_tensor_sha256'].items():
    assert hashlib.sha256((original / (name + '.f32le')).read_bytes()).hexdigest() == expected
epsilon = torch.from_numpy(np.fromfile(original / 'source-epsilon.f32le', dtype='<f4').copy().reshape(1, 64, 750))
mask = torch.from_numpy(np.fromfile(original / 'source-mask.f32le', dtype='<f4').copy().reshape(1, 750, 64))
record = next(r for r in json.loads((root / 'configs/assets-base.json').read_text())['files']
              if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
checkpoint = local_checkpoint(record, root / 'work/drum-waveforms/vae/diffusion_pytorch_model.safetensors')
encoder = load_encoder(checkpoint)
with torch.no_grad():
    outputs = posterior_context(encoder(resample(pcm)), epsilon, mask)
metrics = {}
for name, value in zip(['source-mean', 'source-std', 'source-latent', 'source-context'], outputs):
    reference = torch.from_numpy(np.fromfile(original / (name + '.f32le'), dtype='<f4').copy().reshape(value.shape))
    metrics[name] = precision_metrics(value, reference)
print(json.dumps({'scope': __doc__, 'source_component_sensitivity': metrics}, indent=2))
