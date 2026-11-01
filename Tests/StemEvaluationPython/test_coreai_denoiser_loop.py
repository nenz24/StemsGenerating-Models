"""Numerical feedback/update and committed-reference integrity; no CoreAI emulation."""
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
import prepare_coreai_denoiser_loop as MODULE


class DenoiserLoopTests(unittest.TestCase):
    folder=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-loop-4'

    def test_feedback_matches_analytic_discrete_decay(self):
        import torch
        initial=torch.tensor([[[2.,-4.]]]);seen=[]
        def velocity(x,t,r):
            seen.append((x.clone(),t.item(),r.item()))
            return x
        result=MODULE.trajectory(velocity,{'noise':initial})
        torch.testing.assert_close(result[-1][1],initial*(0.75**4),rtol=0,atol=0)
        self.assertEqual([(t,r) for _,t,r in seen],[(1.,1.),(.75,.75),(.5,.5),(.25,.25)])
        self.assertTrue(torch.equal(initial,torch.tensor([[[2.,-4.]]])))
        for index,(x,_,_) in enumerate(seen):
            torch.testing.assert_close(x,initial*(0.75**index),rtol=0,atol=0)

    def test_pinned_publisher_ode_executes_same_signed_update(self):
        import torch
        # The actual pinned source excerpt is retained for independent testing
        # without needing a network or a 4.79 GB checkpoint.
        ref=json.loads((self.folder/'reference.json').read_text())
        statements=ref['publisher_sampling_semantics']['ode_statements']
        code=compile('\n'.join(statements),'<pinned publisher ODE>','exec')
        initial=torch.tensor([[[8.,-3.]]]);v=torch.tensor([[[2.,-4.]]])
        env={'torch':torch,'xt':initial.clone(),'vt':v,'t_curr':torch.tensor(.75),
             't_prev':torch.tensor(.5),'bsz':1,'device':'cpu','dtype':torch.float32}
        exec(code,env)
        torch.testing.assert_close(env['xt'],torch.tensor([[[7.5,-2.]]]),rtol=0,atol=0)

    def test_sampling_source_contract_rejects_wrong_sign_and_timestep_r(self):
        ref=json.loads((self.folder/'reference.json').read_text())
        statements='\n'.join('        '+line for line in ref['publisher_sampling_semantics']['ode_statements'])
        source="""def generate_audio(self, infer_steps=4):
    t = torch.linspace(1.0, 0.0, infer_steps + 1, device=device, dtype=dtype)
    self.decoder(timestep_r=t_curr_tensor)
    if infer_method == 'ode':
"""+statements+'\n'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'publisher.py';path.write_text(source)
            MODULE.verify_sampling_source(path)
            for invalid in [source.replace('xt - vt','xt + vt'),source.replace('timestep_r=t_curr_tensor','timestep_r=t_prev')]:
                path.write_text(invalid)
                with self.assertRaises(ValueError):MODULE.verify_sampling_source(path)

    def test_real_references_and_feedback_consistency(self):
        import numpy as np
        ref=MODULE.verify_fixture(self.folder)
        # Independently check ALL updates for both trajectories, including the
        # initial input that must travel with these exact references.
        for velocity_key,latent_key in [('velocity_file','latent_file'),('original_velocity_file','original_latent_file')]:
            previous=np.fromfile(self.folder/ref['input_files']['noise'],dtype='<f4')
            for step in ref['steps']:
                velocity=np.fromfile(self.folder/step[velocity_key],dtype='<f4')
                actual=np.fromfile(self.folder/step[latent_key],dtype='<f4')
                expected=previous-velocity*np.float32(step['timestep']-step['next_timestep'])
                np.testing.assert_array_equal(actual,expected)
                previous=actual

    def test_fixture_is_self_contained_and_rejects_corrupt_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fixture';shutil.copytree(self.folder,folder)
            # No separately generated model bundle or random inputs accompany
            # this copy; integrity verification must still be possible.
            ref=MODULE.verify_fixture(folder)
            self.assertEqual(set(ref['input_files']),{'noise','condition','context'})
            original=(folder/'reference.json').read_text()
            ref['shared_input_sha256']['condition']='0'*64
            (folder/'reference.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'input identity mismatch'):
                MODULE.verify_fixture(folder)
            (folder/'reference.json').write_text(original)
            p=folder/ref['input_files']['condition'];raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'integrity mismatch'):
                MODULE.verify_fixture(folder)

    def test_corrupt_references_and_schedule_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'fixture';shutil.copytree(self.folder,folder)
            ref=json.loads((folder/'reference.json').read_text())
            for change in ['wrong timestep_r','missing step','weaker tolerance']:
                altered=copy.deepcopy(ref)
                if change=='wrong timestep_r':altered['steps'][1]['timestep_r']=.25
                elif change=='missing step':altered['steps'].pop()
                else:altered['absolute_tolerance']=.01
                (folder/'reference.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError,msg=change):MODULE.verify_fixture(folder)
            (folder/'reference.json').write_text(json.dumps(ref))
            p=folder/ref['steps'][-1]['latent_file'];raw=bytearray(p.read_bytes());raw[0]^=1;p.write_bytes(raw)
            with self.assertRaises(ValueError):MODULE.verify_fixture(folder)

    def test_existing_output_preserved_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'keep';marker.write_text('preserve')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_coreai_denoiser_loop.py'),'--out',tmp],
                                  capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Output already exists',result.stderr)
            self.assertEqual(marker.read_text(),'preserve')


if __name__=='__main__':unittest.main()
