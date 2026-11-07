"""Pinned publisher, random16dim24-layer model: geometry/control only, no music."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from export_coreai_dit_block import records, precision_metrics, convert_storage16
from export_coreai_denoiser import load_full_upstream
from export_coreai_bass_preview import partitions, stage_plan
from prepare_coreai_drum_reference30 import WINDOW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher-folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Existing audit preserved')
    torch.set_num_threads(2)
    publisher = records()
    paths = {n: args.publisher_folder / n for n in
             ('modeling_acestep_v15_base.py', 'configuration_acestep_v15.py', 'config.json')}
    for name, path in paths.items():
        raw = path.read_bytes()
        assert len(raw) == publisher[name]['size']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == publisher[name]['digest']
    cfg, kind, rotary = load_full_upstream(paths)
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.head_dim = 8
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg._attn_implementation = 'eager'
    torch.manual_seed(750127)
    model = kind(cfg).eval().requires_grad_(False)
    model.rotary_emb = rotary(cfg, device='cpu')
    mask = sys.modules[kind.__module__].create_4d_mask
    inputs = {'noise': torch.randn(1, 750, 64), 'context': torch.randn(1, 750, 128),
              'condition': torch.randn(1, 127, 16), 'timestep': torch.tensor([.72])}
    null = torch.randn(1, 1, 16)
    def original():
        return model(hidden_states=inputs['noise'].repeat(2, 1, 1),
                     timestep=inputs['timestep'].expand(2), timestep_r=inputs['timestep'].expand(2),
                     attention_mask=None, encoder_attention_mask=None,
                     encoder_hidden_states=torch.cat([inputs['condition'], null.expand_as(inputs['condition'])]),
                     context_latents=inputs['context'].repeat(2, 1, 1), use_cache=True,
                     past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()))[0]
    def partitioned():
        state = dict(inputs)
        for part, stage in zip(partitions(model, null, mask, WINDOW), stage_plan(WINDOW)):
            values = part(*(state[n] for n in stage['input_shapes']))
            values = values if isinstance(values, tuple) else (values,)
            for name, value in zip(stage['output_shapes'], values):
                state[stage['output_bindings'][name]] = value
        return state['velocity']
    checks = {}
    with torch.no_grad():
        checks['FP32_same_weights_partition_vs_full'] = precision_metrics(partitioned(), original())
        convert_storage16(model, 'bfloat16')
        checks['BF16_same_weights_partition_vs_full'] = precision_metrics(partitioned(), original())
    result = {'schema_version': 1, 'scope': __doc__, 'latent_frames': 750, 'hidden_tokens': 375,
              'condition_tokens': 127, 'layers': 24, 'alternating_sliding_and_full_attention': True,
              'hidden_size': 16, 'head_dim': 8, 'output_shape': [2, 750, 64],
              'comparisons': checks, 'all_checks_passed': all(x['parity_passed'] for x in checks.values()),
              'original_cache_scope': 'Fresh first-step cross-attention cache vs uncached partitions',
              'learned_checkpoint_used': False, 'Apple_exports': 0, 'device_runs': 0,
              'native_parity': 'NOT EVALUATED', 'full_trajectory_parity': 'NOT EVALUATED'}
    with args.out.open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['all_checks_passed']: raise SystemExit(1)


if __name__ == '__main__': main()
