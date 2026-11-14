"""Guidance4 math and explicit quality-candidate scope; no musical evidence."""
import importlib.util
import array
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import audit_coreai_drum_repeat_review as A
import test_coreai_drum_repeat_review as repeat_tests


class CandidateArchiveTests(unittest.TestCase):
    def setUp(self):
        self.f=repeat_tests.RepeatReviewTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        m=self.f.m;m.update(profile='full-song-guidance-candidate-v1',guidance_scale=4,baseline_guidance_scale=7,
            candidate_euler_control_checks=50,candidate_update='Float32 c+0.5*(native_guided7-c); Float32 Euler; native momentum/cap/projection unchanged',
            candidate_vs_baseline_within_bounds=True,baseline_listening_gain=.3)
        del m['repeatability_passed']
        for name in list(self.f.files):
            if name.startswith('repeat/'):
                new='candidate/'+name[len('repeat/'):];self.f.files[new]=self.f.files.pop(name);m['artifacts'][new]=m['artifacts'].pop(name)
        name='candidate/report.txt';self.f.add(name,self.f.files[name]+b'Candidate FP32 Euler control: PASS50/50\n')
        self.f.add('input/piano-focus.f32le',bytes(96000*2*4),[1,2,96000])
        for label in ('captured','candidate'):self.f.add(label+'/piano-with-drums-focus.wav',self.f.files[label+'/focus.wav'])

    def test_candidate_requires_explicit_mode_and_never_claims_repeatability_or_music(self):
        self.f.write()
        with self.assertRaisesRegex(ValueError,'scope'):A.audit(self.f.path)
        r=A.audit(self.f.path,candidate=True)
        self.assertTrue(r['candidate_vs_baseline_within_bounds']);self.assertTrue(r['focus_wav_bindings_passed'])
        self.assertNotIn('repeatability_passed',r);self.assertEqual(r['music_acceptance'],'NOT EVALUATED BY AUDIT')

    def test_expected_output_difference_is_not_candidate_music_failure_or_success(self):
        name='candidate/raw-drums.f32le';self.f.add(name,struct.pack('<f',1.)+self.f.files[name][4:],[1,2,1440000])
        self.f.m['candidate_vs_baseline_within_bounds']=False;self.f.write();r=A.audit(self.f.path,candidate=True)
        self.assertFalse(r['candidate_vs_baseline_within_bounds']);self.assertTrue(r['focus_wav_bindings_passed'])
        self.assertEqual(r['music_acceptance'],'NOT EVALUATED BY AUDIT')

    def test_missing_runtime_control_and_relaxed_bounds_are_rejected(self):
        name='candidate/report.txt';self.f.add(name,self.f.files[name].replace(b'Candidate FP32 Euler control: PASS50/50\n',b''));self.f.write()
        with self.assertRaisesRegex(ValueError,'arithmetic'):A.audit(self.f.path,candidate=True)
        self.f.m['absolute_tolerance']=.1;self.f.write()
        with self.assertRaisesRegex(ValueError,'bounds'):A.audit(self.f.path,candidate=True)

    def test_nonzero_shared_gain_and_piano_mix_are_bound_exactly(self):
        n=96000;start=48000;f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        piano=f(.12345);drum=f(.2);gain=f(.3)
        self.f.add('input/piano-focus.f32le',array.array('f',[piano]*(n*2)).tobytes(),[1,2,n])
        name='candidate/raw-drums.f32le';raw=bytearray(self.f.files[name]);part=array.array('f',[drum]*n).tobytes()
        for c in range(2):raw[(c*1440000+start)*4:(c*1440000+start+n)*4]=part
        self.f.add(name,bytes(raw),[1,2,1440000]);self.f.m['candidate_vs_baseline_within_bounds']=False
        header=self.f.files['candidate/focus.wav'][:58]
        self.f.add('candidate/focus.wav',header+array.array('f',[f(drum*gain)]*(n*2)).tobytes())
        for label,d in [('captured',0.),('candidate',drum)]:
            mixed=f(f(f(piano+d)*.5)*gain)
            self.f.add(label+'/piano-with-drums-focus.wav',header+array.array('f',[mixed]*(n*2)).tobytes())
        self.f.write();r=A.audit(self.f.path,candidate=True)
        self.assertTrue(r['focus_wav_bindings_passed']);self.assertFalse(r['candidate_vs_baseline_within_bounds'])
        self.assertTrue(r['checks']['candidate_piano_mix_binding']['passed'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional CPU APG formula control')
class GuidanceMathTests(unittest.TestCase):
    def test_affine_native_apg_matches_independent_float64_projection_guidance4(self):
        import torch
        from export_coreai_bass_preview import apg_euler
        from export_coreai_dit_block import precision_metrics
        generator=torch.Generator().manual_seed(123)
        def values(scale):return torch.randn((1,750,64),generator=generator)*scale
        for scale in (1e-4,1.,100.):
            cond,uncond,x,momentum=values(scale),values(scale),values(1.),values(scale)
            dt=torch.tensor([.02],dtype=torch.float32)
            native_x,native_m,guided7=apg_euler(torch.cat([cond,uncond]),x,momentum,dt)
            # Independently project the native updated momentum in Float64,
            # following publisher project(), without deriving from guided7.
            norm=native_m.norm(p=2,dim=1,keepdim=True)
            diff=native_m*torch.minimum(torch.ones_like(native_m),2.5/norm)
            direction=torch.nn.functional.normalize(cond.double(),dim=1)
            parallel=((diff.double()*direction).sum(dim=1,keepdim=True)*direction)
            orthogonal=(diff.double()-parallel).float()
            reference=cond+3.*orthogonal
            candidate=cond+(guided7-cond)*.5
            self.assertTrue(precision_metrics(candidate,reference)['parity_passed'])
            self.assertTrue(precision_metrics(x-candidate*dt,x-reference*dt)['parity_passed'])
            self.assertTrue(precision_metrics(x-guided7*dt,native_x)['parity_passed'])

    def test_zero_guidance_correction_does_not_create_an_update_difference(self):
        import torch
        from export_coreai_bass_preview import apg_euler
        c=torch.ones(1,750,64);x=torch.ones_like(c);dt=torch.tensor([.02])
        native,_,g=apg_euler(torch.cat([c,c]),x,torch.zeros_like(c),dt)
        self.assertTrue(torch.equal(g,c));self.assertTrue(torch.equal(x-(c+(g-c)*.5)*dt,native))


if __name__=='__main__':unittest.main()
