#!/usr/bin/env python3
"""Export an actual CoreAI .aimodel following Apple's coreai-torch quickstart."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'prototypes/ios27-coreai/ModelCandidates')
    args = parser.parse_args()
    path = args.out / 'affine.aimodel'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing asset: {path}')

    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    class Affine(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + 0.25

    model = Affine().eval()
    x = torch.tensor([[0.0, 0.25, -0.5, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        expected = model(x).flatten().tolist()
    exported = torch.export.export(model, args=(x,)).run_decompositions(get_decomp_table())
    program = (TorchConverter().add_exported_program(
        exported, input_names=['x'], output_names=['y'], entrypoint_name='main').to_coreai())
    args.out.mkdir(parents=True, exist_ok=True)
    program.save_asset(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise RuntimeError('Exporter produced no asset files')
    reference = {
        'asset_file': path.name, 'function_name': 'main',
        'input_name': 'x', 'output_name': 'y',
        'input_shape': [1, 4], 'output_shape': [1, 4],
        'input_values': x.flatten().tolist(), 'expected_values': expected,
        'absolute_tolerance': 1e-6,
        'exporter': 'coreai-torch ' + importlib.metadata.version('coreai-torch'),
        'coreai_core_version': importlib.metadata.version('coreai-core'),
        'torch_version': torch.__version__,
        'runtime_under_test': 'CoreAI',
        'reference_execution': 'PyTorch CPU',
        'coreai_device_inference_verified': False,
        'artifact_sha256': {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files},
    }
    (args.out / 'affine-native-reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(f'Exported native asset: {path}')
    print(f'PyTorch reference: {expected}')
    print('CoreAI iPhone model loading/inference: NOT YET VERIFIED')


if __name__ == '__main__':
    main()
