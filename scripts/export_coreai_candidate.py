#!/usr/bin/env python3
"""Export a tiny legacy-format candidate; CoreAI acceptance must be tested on device.

Developer-only dependencies: coremltools==9.0 and numpy. The iPhone application
does not import CoreML or invoke Core ML prediction as a fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'prototypes/ios27-coreai/ModelCandidates')
    args = parser.parse_args()
    package = args.out / 'affine.mlpackage'
    if package.exists():
        raise FileExistsError(f'Refusing to overwrite existing candidate: {package}')

    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    if ct.__version__ != '9.0':
        raise RuntimeError('This candidate was defined for coremltools==9.0')

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4))], opset_version=ct.target.iOS18)
    def program(x):
        return mb.add(x=mb.mul(x=x, y=np.float32(2.0)), y=np.float32(0.25), name='y')

    model = ct.convert(program, convert_to='mlprogram',
                       minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT32)
    model.short_description = 'Numerical compatibility candidate: y = 2*x + 0.25'
    spec = model.get_spec()
    assert spec.WhichOneof('Type') == 'mlProgram'
    assert len(spec.description.input) == len(spec.description.output) == 1
    assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 4]
    assert list(spec.description.output[0].type.multiArrayType.shape) == [1, 4]
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(package))
    # The package manifest references a weights directory even though this tiny
    # graph inlines its constants. Git must preserve that directory after pull.
    weights = package / 'Data/com.apple.CoreML/weights'
    weights.mkdir(parents=True, exist_ok=True)
    (weights / '.gitkeep').touch()

    reference = {
        'asset_file': package.name,
        'function_name': 'main',
        'input_name': spec.description.input[0].name,
        'output_name': spec.description.output[0].name,
        'input_shape': [1, 4], 'output_shape': [1, 4],
        'input_values': [0.0, 0.25, -0.5, 1.0],
        'expected_values': [0.25, 0.75, -0.75, 2.25],
        'absolute_tolerance': 1e-6,
        'exporter': f'coremltools {ct.__version__}',
        'candidate_format': 'Core ML ML Program package; CoreAI compatibility unverified',
        'requested_runtime': 'CoreAI',
        'coreai_model_loaded': False,
        'coreai_prediction_executed': False,
        'artifact_sha256': {
            str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*')) if p.is_file()
        },
    }
    (args.out / 'affine-reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(f'Exported candidate and reference: {args.out}')
    print('CoreAI model acceptance and inference: NOT TESTED; run the device probe.')


if __name__ == '__main__':
    main()
