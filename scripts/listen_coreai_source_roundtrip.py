#!/usr/bin/env python3
"""Create local piano/source-reconstruction WAVs from an existing 10.24 s bundle.

One CPU FP32 VAE decode, no encoder/DiT, model export or checkpoint download.
Original PCM, source latent, model resources and saved generations stay unchanged.
"""
import argparse
import json
from pathlib import Path

from coreai_listening_audio import headroom_gain, read_planar, write_float_wav
from coreai_preview_window import LONG
from export_coreai_decoder import ROOT, sha256
from export_coreai_source_context import verify_bundle as verify_source
from compare_coreai_drum_lyrics import local_checkpoint


def reconstruct(source, out, explicit_checkpoint=None):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'Existing listening result preserved: {out}')
    if (out.resolve().is_relative_to(source.resolve())
            or out.resolve().is_relative_to((ROOT / 'prototypes/ios27-coreai/ModelCandidates').resolve())):
        raise ValueError('Listening output must be outside ModelCandidates/source bundle')
    print('Verifying existing source bundle; no model export', flush=True)
    ref = verify_source(source, LONG)
    source_identity = sha256(source / 'reference.json')
    piano = read_planar(source / 'audio.f32le', LONG.samples)
    record = next(r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
                  if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    report = {'schema_version': 1, 'status': 'running',
              'mode': 'One CPU FP32 VAE decode of saved source latent; no encoder/DiT/native export',
              'source_reference_sha256': source_identity,
              'source_pcm_sha256': ref['artifact_sha256']['audio.f32le'],
              'source_latent_sha256': ref['artifact_sha256']['source_latents-reference.f32le'],
              'checkpoint_sha256': record['digest'], 'sample_rate': 48000, 'channels': 2,
              'frames': LONG.samples, 'audio_seconds': LONG.seconds,
              'source_and_reconstruction_use_common_attenuation': True,
              'original_files_modified': False, 'encoder_calls': 0, 'sampling_steps': 0,
              'decoder_calls': 0, 'native_exports': 0, 'listening_assessment': 'NOT EVALUATED'}
    out.mkdir(parents=True, exist_ok=False)

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    save()
    try:
        print('Verifying pinned local VAE checkpoint; no download fallback', flush=True)
        checkpoint = local_checkpoint(record, explicit_checkpoint)
        import numpy as np
        import torch
        from safetensors import safe_open
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
        torch.set_num_threads(2)
        report['torch_version'] = torch.__version__
        cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
        with torch.device('meta'):
            decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                     audio_channels=cfg['audio_channels'],
                                     upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                     channel_multiples=cfg['channel_multiples']).eval()
        with safe_open(str(checkpoint), framework='pt', device='cpu') as weights:
            state = {k[len('decoder.'):]: weights.get_tensor(k)
                     for k in weights.keys() if k.startswith('decoder.')}
        decoder.load_state_dict(state, strict=True, assign=True)
        del state
        decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
        latent = np.fromfile(source / 'source_latents-reference.f32le', dtype='<f4')
        if latent.size != LONG.frames * 64 or not np.isfinite(latent).all():
            raise ValueError('Invalid saved source latent')
        latent = latent.reshape(1, LONG.frames, 64).transpose(0, 2, 1).copy()
        print('Decoding saved piano latent once; no generation/sampling', flush=True)
        with torch.no_grad():
            audio = decoder(torch.from_numpy(latent)).contiguous()
        report['decoder_calls'] = 1
        if list(audio.shape) != [1, 2, LONG.samples] or not torch.isfinite(audio).all():
            raise ValueError('Invalid source reconstruction shape/values')
        reconstructed = audio.reshape(-1).tolist()
        peaks = {'piano_source': max(abs(v) for v in piano),
                 'piano_reconstruction': max(abs(v) for v in reconstructed)}
        gain = headroom_gain(max(peaks.values()))
        write_float_wav(out / '01-piano-source.wav', piano, LONG.samples, gain)
        write_float_wav(out / '02-piano-reconstruction.wav', reconstructed, LONG.samples, gain)
        for name in ['audio.f32le', 'source_latents-reference.f32le']:
            if sha256(source / name) != ref['artifact_sha256'][name]:
                raise ValueError(f'Source file changed during comparison: {name}')
        if sha256(source / 'reference.json') != source_identity:
            raise ValueError('Source reference changed during comparison')
        report.update(status='completed_listening_copies', raw_sample_peaks=peaks,
                      common_linear_gain=gain,
                      artifact_sha256={p.name: sha256(p) for p in out.glob('*.wav')})
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print('Ready:', out, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vae-checkpoint', type=Path, help='Optional existing pinned local VAE checkpoint')
    args = parser.parse_args()
    reconstruct(ROOT / 'prototypes/ios27-coreai/ModelCandidates/source-context-1024',
                args.out, args.vae_checkpoint)


if __name__ == '__main__':
    main()
