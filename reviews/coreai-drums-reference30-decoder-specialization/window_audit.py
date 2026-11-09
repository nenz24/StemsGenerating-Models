"""Actual saved-latent window256 vs accepted full750 decoder audit; no export."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


def tiles():
    result = []
    for start in range(0, 750, 128):
        window = min(max(start - 64, 0), 750 - 256)
        end = min(start + 128, 750)
        result.append({'window_start': window, 'window_frames': 256,
                       'core_start': start, 'core_end': end, 'crop_start': start - window})
    return result


def dependency_bounds(tile):
    """Conservative inverse-convolution bounds, checking every artificial edge.

    This specific architecture has local Snake activations, Conv1d and
    ConvTranspose1d only. True recording boundaries share padding with full750;
    interior boundaries must never be reached by the retained output cone.
    """
    strides = [10, 6, 4, 4, 2]
    size = 256 * 1920
    lo = tile['crop_start'] * 1920
    hi = (tile['core_end'] - tile['window_start']) * 1920 - 1
    def clip(a, b, length):
        if a < 0 and tile['window_start'] != 0:
            raise ValueError('Retained output touches artificial left padding')
        if b >= length and tile['window_start'] + 256 != 750:
            raise ValueError('Retained output touches artificial right padding')
        return max(a, 0), min(b, length - 1)
    lo, hi = clip(lo - 3, hi + 3, size)  # final kernel7 convolution
    for stride in reversed(strides):
        for dilation in (9, 3, 1):
            lo, hi = clip(lo - 3 * dilation, hi + 3 * dilation, size)
        padding = (stride + 1) // 2
        lo = (lo + padding - (2 * stride - 1) + stride - 1) // stride
        hi = (hi + padding) // stride
        size //= stride
        lo, hi = clip(lo, hi, size)
    lo, hi = clip(lo - 3, hi + 3, 256)  # first kernel7 convolution
    return [tile['window_start'] + lo, tile['window_start'] + hi]


def verify_architecture(decoder):
    import torch
    from diffusers.models.autoencoders.autoencoder_oobleck import (
        OobleckDecoder, OobleckDecoderBlock, OobleckResidualUnit, Snake1d)
    allowed = (OobleckDecoder, OobleckDecoderBlock, OobleckResidualUnit, Snake1d,
               torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.ModuleList)
    if any(not isinstance(m, allowed) for m in decoder.modules()):
        raise ValueError('Unknown temporal/global operator; receptive-field proof invalid')
    for conv in (decoder.conv1, decoder.conv2):
        assert (conv.kernel_size, conv.padding, conv.stride, conv.dilation) == ((7,), (3,), (1,), (1,))
    assert len(decoder.block) == 5
    for block, stride in zip(decoder.block, [10, 6, 4, 4, 2]):
        conv = block.conv_t1
        assert (conv.kernel_size, conv.padding, conv.stride, conv.dilation) == ((2*stride,), ((stride+1)//2,), (stride,), (1,))
        for unit, dilation in zip([block.res_unit1, block.res_unit2, block.res_unit3], [1, 3, 9]):
            assert (unit.conv1.kernel_size, unit.conv1.padding, unit.conv1.dilation, unit.conv1.stride) == ((7,), (3*dilation,), (dilation,), (1,))
            assert (unit.conv2.kernel_size, unit.conv2.padding, unit.conv2.dilation, unit.conv2.stride) == ((1,), (0,), (1,), (1,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--vae-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Existing audit preserved')
    if (args.out.resolve().is_relative_to(args.capture.resolve()) or any(args.out.resolve().is_relative_to(ROOT / p)
            for p in ('baselines', 'prototypes/ios27-coreai/ModelCandidates'))):
        raise ValueError('Preserve capture/baselines/native bundles')
    import torch
    from safetensors import safe_open
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder
    from audit_coreai_drum_reference30 import verify_capture, read_tensor
    from export_coreai_decoder import verify_checkpoint
    from export_coreai_dit_block import precision_metrics
    torch.set_num_threads(2)
    capture = verify_capture(args.capture)
    record = next(r for r in json.loads((ROOT / 'configs/assets-base.json').read_text())['files']
                  if r['filename'] == 'vae/diffusion_pytorch_model.safetensors')
    verify_checkpoint(args.vae_checkpoint, record)
    assert record['digest'] == capture['checkpoint_sha256']['vae']
    cfg = json.loads((ROOT / 'configs/coreai-vae-config.json').read_text())
    with torch.device('meta'):
        decoder = OobleckDecoder(channels=cfg['decoder_channels'], input_channels=cfg['decoder_input_channels'],
            audio_channels=cfg['audio_channels'], upsampling_ratios=list(reversed(cfg['downsampling_ratios'])),
            channel_multiples=cfg['channel_multiples']).eval()
    with safe_open(str(args.vae_checkpoint), framework='pt', device='cpu') as stream:
        weights = {k[len('decoder.'):]: stream.get_tensor(k) for k in stream.keys() if k.startswith('decoder.')}
    decoder.load_state_dict(weights, strict=True, assign=True)
    del weights
    decoder.to(device='cpu', dtype=torch.float32).requires_grad_(False)
    verify_architecture(decoder)
    latent = read_tensor(args.capture, 'final-latent.f32le').transpose(1, 2).contiguous()
    gold = read_tensor(args.capture, 'final-audio.f32le')
    assembled = torch.zeros_like(gold)
    coverage = [0] * 750
    rows = []
    with torch.no_grad():
        for index, tile in enumerate(tiles()):
            bounds = dependency_bounds(tile)
            start = tile['window_start']
            inputs = latent[..., start:start+256].contiguous()
            print(f'Decode window {index+1}/6; start{start}; keep{tile["core_start"]}:{tile["core_end"]}', flush=True)
            decoded = decoder(inputs)
            assert list(decoded.shape) == [1, 2, 491520] and torch.isfinite(decoded).all()
            a, b = tile['core_start']*1920, tile['core_end']*1920
            crop = decoded[..., tile['crop_start']*1920:tile['crop_start']*1920+(b-a)]
            assembled[..., a:b] = crop
            for frame in range(tile['core_start'], tile['core_end']): coverage[frame] += 1
            metrics = precision_metrics(crop, gold[..., a:b])
            rows.append({**tile, 'latent_dependency_global_inclusive': bounds,
                         'input_sha256': hashlib.sha256(inputs.numpy().astype('<f4').tobytes()).hexdigest(),
                         'comparison': metrics})
            del decoded, crop
    assert set(coverage) == {1}
    seams = []
    for tile in tiles()[1:]:
        center = tile['core_start'] * 1920
        seams.append({'sample': center, 'comparison': precision_metrics(
            assembled[..., center-3840:center+3840], gold[..., center-3840:center+3840])})
    overall = precision_metrics(assembled, gold)
    result = {'schema_version': 1, 'scope': __doc__, 'checkpoint_sha256': record['digest'],
              'versions': {n: importlib.metadata.version(n) for n in ('torch', 'diffusers')},
              'architecture_guard_passed': True, 'all_artificial_padding_excluded': True,
              'coverage_exactly_once': True, 'zero_padding_added': False, 'crossfade': False,
              'decoder_calls': 6, 'DiT_calls': 0, 'native_exports': 0, 'device_runs': 0,
              'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-3, 'relative_rmse_tolerance': 1e-3,
              'tiles': rows, 'seam_regions': seams, 'overall': overall,
              'all_checks_passed': overall['parity_passed'] and all(r['comparison']['parity_passed'] for r in rows+seams),
              'native_window_support': 'NOT EVALUATED', 'full_native_trajectory': 'NOT EVALUATED'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(overall), flush=True)
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__ == '__main__': main()
