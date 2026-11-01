"""Check the saved candidate and reference; this is not Apple model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'prototypes/ios27-coreai/ModelCandidates'


class CoreAICandidateTests(unittest.TestCase):
    def test_package_manifest_and_reference_cover_all_saved_model_files(self):
        reference = json.loads((ASSETS / 'affine-reference.json').read_text())
        files = {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ASSETS / reference['asset_file']).rglob('*') if p.is_file()}
        self.assertEqual(reference['artifact_sha256'], files)
        manifest = json.loads((ASSETS / reference['asset_file'] / 'Manifest.json').read_text())
        root = manifest['itemInfoEntries'][manifest['rootModelIdentifier']]
        model = ASSETS / reference['asset_file'] / 'Data' / root['path']
        self.assertTrue(model.is_file())
        for item in manifest['itemInfoEntries'].values():
            self.assertTrue((ASSETS / reference['asset_file'] / 'Data' / item['path']).exists())
        self.assertLess(sum((ASSETS / path).stat().st_size for path in files), 65536)

    def test_export_metadata_does_not_claim_coreai_model_execution(self):
        reference = json.loads((ASSETS / 'affine-reference.json').read_text())
        self.assertEqual(reference['requested_runtime'], 'CoreAI')
        self.assertFalse(reference['coreai_model_loaded'])
        self.assertFalse(reference['coreai_prediction_executed'])
        self.assertIn('compatibility unverified', reference['candidate_format'])

    @unittest.skipUnless(importlib.util.find_spec('coremltools'), 'optional export-tool validation')
    def test_serialized_graph_wiring_types_and_constants_match_reference(self):
        # Inspect protobuf directly. Do not call predict() or use a fake CoreAI.
        import coremltools as ct
        reference = json.loads((ASSETS / 'affine-reference.json').read_text())
        model = ASSETS / reference['asset_file'] / 'Data/com.apple.CoreML/model.mlmodel'
        spec = ct.proto.Model_pb2.Model()
        spec.ParseFromString(model.read_bytes())
        reopened = ct.models.MLModel(str(ASSETS / reference['asset_file']), skip_model_load=True)
        self.assertEqual(reopened.get_spec().WhichOneof('Type'), 'mlProgram')
        self.assertEqual(spec.WhichOneof('Type'), 'mlProgram')
        self.assertEqual(list(spec.mlProgram.functions), [reference['function_name']])
        for feature, name, shape in [
            (spec.description.input[0], reference['input_name'], reference['input_shape']),
            (spec.description.output[0], reference['output_name'], reference['output_shape']),
        ]:
            self.assertEqual(feature.name, name)
            self.assertEqual(list(feature.type.multiArrayType.shape), shape)
            self.assertEqual(feature.type.multiArrayType.dataType,
                             ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32)
        function = spec.mlProgram.functions[reference['function_name']]
        self.assertEqual(len(function.block_specializations), 1)
        block = next(iter(function.block_specializations.values()))
        self.assertEqual(list(block.outputs), [reference['output_name']])
        self.assertEqual([op.type for op in block.operations], ['const', 'mul', 'const', 'add'])
        values = {reference['input_name']: reference['input_values']}
        for op in block.operations:
            self.assertEqual(len(op.outputs), 1)
            if op.type == 'const':
                values[op.outputs[0].name] = list(op.attributes['val'].immediateValue.tensor.floats.values)
            else:
                left = values[op.inputs['x'].arguments[0].name]
                right = values[op.inputs['y'].arguments[0].name]
                self.assertEqual(len(right), 1)
                values[op.outputs[0].name] = [x * right[0] if op.type == 'mul' else x + right[0]
                                            for x in left]
        self.assertEqual(values[reference['output_name']], reference['expected_values'])


if __name__ == '__main__':
    unittest.main()
