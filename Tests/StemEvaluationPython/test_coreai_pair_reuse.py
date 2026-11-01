"""Real retained-pair fixture integrity; no emulation of native function reuse."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_coreai_pair_reuse as MODULE
from export_coreai_decoder import sha256


class PairReuseFixtureTests(unittest.TestCase):
    folder=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-pair-reuse'

    def test_real_pair_dual_reference_meets_original_bounds(self):
        import numpy as np
        import torch
        from export_coreai_dit_block import precision_metrics
        ref=MODULE.verify_fixture(self.folder)
        expected=torch.from_numpy(np.fromfile(self.folder/ref['expected_file'],dtype='<f4').copy())
        original=torch.from_numpy(np.fromfile(self.folder/ref['original_fp32_reference_file'],dtype='<f4').copy())
        metrics=precision_metrics(expected,original)
        self.assertTrue(metrics['parity_passed'])
        self.assertEqual(metrics,ref['host_storage16_vs_original_fp32'])

    def test_packaged_inputs_are_complete_finite_and_match_loop_provenance(self):
        import numpy as np
        ref=MODULE.verify_fixture(self.folder)
        self.assertEqual(ref['loop_manifest_sha256'],sha256(ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-loop-4/reference.json'))
        self.assertEqual(len(ref['input_shapes']),8)
        for name,shape in ref['input_shapes'].items():
            values=np.fromfile(self.folder/f'{name}.f32le',dtype='<f4')
            self.assertEqual(values.size,int(np.prod(shape)))
            self.assertTrue(np.isfinite(values).all())

    def test_incompatible_shapes_weights_and_corrupt_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fixture';shutil.copytree(self.folder,folder)
            original=json.loads((folder/'reference.json').read_text())
            for field,value in [('asset_file','blocks-02-03.aimodel'),('run_count',3),('checkpoint_digest','wrong'),('absolute_tolerance',.1),('input_shapes',{'hidden':[1,160,2048]})]:
                changed=copy.deepcopy(original);changed[field]=value
                (folder/'reference.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError,msg=field):MODULE.verify_fixture(folder)
            (folder/'reference.json').write_text(json.dumps(original))
            p=folder/'hidden.f32le';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'integrity mismatch'):MODULE.verify_fixture(folder)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('original')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_coreai_pair_reuse.py'),'--out',tmp],text=True,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'original')


if __name__=='__main__':unittest.main()
