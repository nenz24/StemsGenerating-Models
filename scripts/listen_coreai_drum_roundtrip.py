#!/usr/bin/env python3
"""Compare approved Colab percussion with its pinned FP32 VAE reconstruction.

One encoder and one decoder call, no DiT or native export. This is a lossy codec
control, not regeneration of the approved latent or a musical acceptance gate.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import wave
import zipfile

from compare_coreai_drum_lyrics import local_checkpoint
from coreai_listening_audio import headroom_gain, write_float_wav
from export_coreai_decoder import ROOT, sha256

BASELINE = ROOT / 'baselines/acoustic-drums-30-v1'
ARCHIVE_SHA256 = '70400534aac9e2ac14db1d7fc2bdaec31585cec9285e8fe5b60222fd202b8433'
DRUM_SHA256 = '29d409250f6104936b7fea215f5b1798c37f52bc8b91b3668003567952476fce'
SAMPLES = 491520


def approved_audio(archive):
    """Verify the archived WAV; resample its first 10.24 s without normalization."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Approved drum archive checksum mismatch; preserved')
    with zipfile.ZipFile(archive) as bundle:
        members = [i for i in bundle.infolist() if i.filename == 'drums.wav']
        if len(members) != 1 or members[0].file_size != 3840078:
            raise ValueError('Unexpected approved drum archive member')
        raw = bundle.read(members[0])
    if hashlib.sha256(raw).hexdigest() != DRUM_SHA256:
        raise ValueError('Approved drum WAV checksum mismatch')
    with wave.open(io.BytesIO(raw)) as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getnframes()) != (2, 32000, 2, 960000):
            raise ValueError('Unexpected approved drum PCM format')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0',
                              '-ar', '48000', '-ac', '2', '-f', 'f32le', 'pipe:1'],
                             input=raw, capture_output=True, check=True).stdout
    import numpy as np
    values = np.frombuffer(decoded, dtype='<f4')
    if values.size != 2 * 1440000 or not np.isfinite(values).all():
        raise ValueError('Invalid resampled approved drum audio')
    return values.reshape(-1, 2)[:SAMPLES].T.copy()[None]


def reconstruct(out, explicit_checkpoint=None, archive=None):
    out = Path(out)
    archive = Path(archive) if archive else BASELINE / 'drum_result.zip'
    if out.exists():
        raise FileExistsError(f'Existing listening result preserved: {out}')
    if any(out.resolve().is_relative_to(p.resolve()) for p in
           [ROOT / 'prototypes/ios27-coreai/ModelCandidates', ROOT / 'baselines', archive.parent]):
        raise ValueError('Listening output must be outside model/baseline/input folders')
    audio = approved_audio(archive)
    manifest = json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
    record = next(r for r in manifest if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    config_record = next(r for r in manifest if r['filename'] == 'vae/config.json')
    raw_config = (ROOT / 'configs/coreai-vae-config.json').read_bytes()
    if hashlib.sha1(f'blob {len(raw_config)}\0'.encode() + raw_config).hexdigest() != config_record['digest']:
        raise ValueError('Pinned VAE configuration checksum mismatch')
    report = {'schema_version': 1, 'status': 'running',
              'experiment': 'approved-percussion-fp32-vae-roundtrip',
              'scope': 'Lossy codec control on first 10.24s of approved 30s Colab PCM16/32kHz export; not original latent replay or CUDA parity',
              'archive_sha256': ARCHIVE_SHA256, 'approved_wav_sha256': DRUM_SHA256,
              'prepared_pcm_sha256': hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
              'checkpoint_sha256': record['digest'], 'sample_rate': 48000,
              'audio_seconds': 10.24, 'latent_frames': 256,
              'posterior': 'sample; explicit CPU generator seed 31; not the original Colab latent',
              'source_preparation': 'ffmpeg full 30s PCM16 stereo 32kHz export -> stereo Float32 48kHz, then first 10.24s; no normalization',
              'encoder_calls': 0, 'decoder_calls': 0, 'sampling_steps': 0, 'native_exports': 0,
              'baseline_modified': False, 'listening_assessment': 'NOT EVALUATED'}
    out.mkdir(parents=True, exist_ok=False)

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    save()
    try:
        print('Verifying pinned local VAE; no checkpoint download', flush=True)
        checkpoint = local_checkpoint(record, explicit_checkpoint)
        import numpy as np
        import torch
        from safetensors import safe_open
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder, OobleckDecoder, OobleckDiagonalGaussianDistribution
        torch.set_num_threads(2)
        report['torch_version'] = torch.__version__
        report['ffmpeg_version'] = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        cfg = json.loads(raw_config)

        def load(module, prefix):
            with safe_open(str(checkpoint), framework='pt', device='cpu') as weights:
                state = {k[len(prefix):]: weights.get_tensor(k) for k in weights.keys() if k.startswith(prefix)}
            module.load_state_dict(state, strict=True, assign=True)
            return module.to(device='cpu', dtype=torch.float32).eval().requires_grad_(False)

        print('Encoding approved drum excerpt once', flush=True)
        with torch.device('meta'):
            encoder = OobleckEncoder(encoder_hidden_size=cfg['encoder_hidden_size'], audio_channels=cfg['audio_channels'],
                                     downsampling_ratios=cfg['downsampling_ratios'], channel_multiples=cfg['channel_multiples'])
        encoder = load(encoder, 'encoder.')
        with torch.no_grad():
            parameters = encoder(torch.from_numpy(audio))
            report['encoder_calls'] = 1
            latent = OobleckDiagonalGaussianDistribution(parameters).sample(generator=torch.Generator(device='cpu').manual_seed(31)).contiguous()
        if list(latent.shape) != [1, 64, 256] or not torch.isfinite(latent).all():
            raise ValueError('Invalid approved drum latent')
        latent.numpy().astype('<f4').tofile(out / 'reencoded-latent.f32le')
        del encoder, parameters
        gc.collect()
        print('Decoding re-encoded approved drums once; no DiT', flush=True)
        with torch.device('meta'):
            decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
                                     audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
                                     channel_multiples=cfg['channel_multiples'])
        decoder = load(decoder, 'decoder.')
        with torch.no_grad():
            reconstructed = decoder(latent).contiguous()
        report['decoder_calls'] = 1
        if list(reconstructed.shape) != [1, 2, SAMPLES] or not torch.isfinite(reconstructed).all():
            raise ValueError('Invalid approved drum reconstruction')
        reconstructed = reconstructed.numpy()
        peaks = {'approved': float(np.max(np.abs(audio))), 'reconstruction': float(np.max(np.abs(reconstructed)))}
        gain = headroom_gain(max(peaks.values()))
        for name, data in [('01-approved-drums.wav', audio), ('02-approved-drums-vae.wav', reconstructed)]:
            write_float_wav(out / name, data.reshape(-1), SAMPLES, gain)
        x, y = audio.astype(np.float64), reconstructed.astype(np.float64)
        report['lossy_roundtrip_metrics_not_acceptance'] = {
            'relative_rms_error': float(np.sqrt(np.mean((x-y)**2)/np.mean(x*x))),
            'channel_correlations': [float(np.corrcoef(x[0,i], y[0,i])[0,1]) for i in range(2)]}
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError('Approved input archive changed during comparison')
        report.update(status='completed_listening_copies', raw_sample_peaks=peaks, common_linear_gain=gain,
                      artifact_sha256={p.name: sha256(p) for p in out.iterdir() if p.suffix in ['.wav', '.f32le']})
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
    parser.add_argument('--vae-checkpoint', type=Path, help='Optional existing pinned local VAE')
    args = parser.parse_args()
    reconstruct(args.out, args.vae_checkpoint)


if __name__ == '__main__':
    main()
