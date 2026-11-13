"""Archive identity, strict regional gates and read-only defaults; no learned inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_coreai_instrument_transition_decode as A
import package_coreai_instrument_transition_review as P
import test_coreai_instrument_transition_review as transport


class PlanTests(unittest.TestCase):
    def test_default_needs_no_site_packages_input_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'no-output'
            result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/audit_coreai_instrument_transition_decode.py'),
                                     '--archive',str(out/'missing.zip'),'--out',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('PLAN ONLY',result.stdout)
            self.assertFalse(out.exists())

    def test_listening_page_shows_original_song_clock_and_uses_local_audio_only(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            A.write_listening_page(out,False)
            html = (out/'listen.html').read_text()
            self.assertEqual(html.count('<audio '),4)
            self.assertIn('22+p.currentTime',html)
            self.assertNotIn('http',html)
            A.write_listening_page(out,True)
            self.assertEqual((out/'listen.html').read_text().count('<audio '),6)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Optional NumPy archive reader')
class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = transport.TransitionReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.manifest = P.package(self.fixture.container,self.fixture.out,transport.RUNS)

    def rewrite(self, change):
        path = self.fixture.root/'changed.zip'
        with zipfile.ZipFile(self.fixture.out) as source:
            files = {n:source.read(n) for n in source.namelist()}
        change(files)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as target:
            for name,raw in files.items(): target.writestr(name,raw)
        return path

    def test_reader_retains_exact_latent_raw_audio_and_report_identity(self):
        values,manifest = A.read_review(self.fixture.out)
        self.assertEqual(manifest,self.manifest)
        self.assertEqual(set(values),{'bass','guitar'})
        for role in P.ROLES:
            self.assertEqual(list(values[role]['latent'].shape),[1,750,64])
            self.assertEqual(list(values[role]['audio'].shape),[1,2,P.FRAMES])
            self.assertEqual(values[role]['metadata']['run_id'],transport.RUNS[role])
            self.assertEqual(values[role]['metadata']['numerical_parity'],'NOT EVALUATED')

    def test_changed_raw_bytes_or_missing_array_cannot_be_read(self):
        def alter(files): files['bass/raw-bass.f32le'] = b'\x00\x00\x80\x3f'+files['bass/raw-bass.f32le'][4:]
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): A.read_review(self.rewrite(alter))
        def remove(files): del files['guitar/final-momentum.f32le']
        with self.assertRaisesRegex(ValueError,'entries'): A.read_review(self.rewrite(remove))

    def test_bounds_or_report_identity_changes_are_rejected(self):
        def bounds(files):
            r = json.loads(files['review-manifest.json'])
            r['absolute_tolerance'] = .001
            files['review-manifest.json'] = json.dumps(r).encode()
        with self.assertRaisesRegex(ValueError,'contract'): A.read_review(self.rewrite(bounds))
        def report(files): files['bass/report.txt'] += b'altered report'
        with self.assertRaisesRegex(ValueError,'report/manifest'): A.read_review(self.rewrite(report))

    def test_valid_hash_cannot_admit_nonfinite_payload(self):
        def mutate(files):
            name = 'bass/final-momentum.f32le'
            files[name] = b'\x00\x00\xc0\x7f'+files[name][4:]
            r = json.loads(files['review-manifest.json'])
            r['artifacts'][name]['sha256'] = hashlib.sha256(files[name]).hexdigest()
            files['review-manifest.json'] = json.dumps(r).encode()
        with self.assertRaisesRegex(ValueError,'Nonfinite'): A.read_review(self.rewrite(mutate))


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional Torch numerical gates')
class RegionalChecks(unittest.TestCase):
    def test_small_global_rmse_does_not_hide_one_point_at_final_seam(self):
        import torch
        reference = torch.ones(1,2,P.FRAMES)
        actual = reference.clone()
        actual[0,0,640*1920] += .003
        result = A.waveform_checks(actual,reference)
        self.assertEqual(result['comparisons_target'],13)
        self.assertEqual(result['strict_verdict'],'NOT PASSED')
        self.assertLess(result['checks']['overall']['normalized_rmse'],1e-3)
        self.assertEqual(result['checks']['overall']['pointwise_violations'],1)
        self.assertFalse(result['checks']['seam-5']['parity_passed'])
        self.assertTrue(result['checks']['core-1']['parity_passed'])

    def test_nonfinite_reference_and_wrong_shape_rejected(self):
        import torch
        reference = torch.ones(1,2,P.FRAMES)
        actual = reference.clone()
        reference[0,0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'): A.waveform_checks(actual,reference)
        with self.assertRaisesRegex(ValueError,'full30s'): A.waveform_checks(actual[...,:10],reference[...,:10])


if __name__=='__main__': unittest.main()
