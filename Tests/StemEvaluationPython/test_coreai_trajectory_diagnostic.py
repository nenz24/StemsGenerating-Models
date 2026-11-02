"""Explicit diagnostic eligibility, NOT simulated inference or numerical PASS."""
import json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
import export_coreai_bass_preview as M
from coreai_preview_window import LONG,SHORT

PASS={'finite':True,'max_absolute_error':0.,'rmse':0.,'normalized_rmse':0.,'elementwise_fraction_over_bound':0.,'parity_passed':True}
MAC_FINAL={'finite':True,'max_absolute_error':0.0138322114944458,'rmse':0.0014775267470531978,'normalized_rmse':0.0014409045433095061,'elementwise_fraction_over_bound':0.5289306640625,'parity_passed':False}

class TrajectoryDiagnosticTests(unittest.TestCase):
    def comparisons(self):
        rows=[{'step':k,'latent':dict(PASS)} for k in range(1,51)]
        rows[-1]['latent']=dict(MAC_FINAL)
        apg=[dict(PASS) for _ in range(50)]
        checks=[{'step':k,'comparison':dict(PASS)} for k in M.CHECKPOINTS]
        return rows,apg,checks,dict(PASS)

    def test_mac_drift_remains_failed_legacy_gate_and_explicitly_failed_diagnostic(self):
        rows,apg,checks,audio=self.comparisons()
        with self.assertRaises(ValueError):M.classify_host_references(rows,apg,audio)
        status=M.classify_trajectory_diagnostic(rows,apg,checks,audio)
        self.assertEqual(status['host_export_status'],'trajectory_drift_diagnostic')
        self.assertFalse(status['host_numerical_gate_passed'])
        self.assertFalse(status['host_dual_trajectory_parity_passed'])
        self.assertFalse(rows[-1]['latent']['parity_passed'])
        self.assertEqual(rows[-1]['latent']['normalized_rmse'],MAC_FINAL['normalized_rmse'])

    def test_component_failures_incomplete_nonfinite_or_forged_flags_still_block_export(self):
        for field in ['denoiser','apg','nonfinite','forged','missing']:
            rows,apg,checks,audio=self.comparisons()
            bad={**PASS,'elementwise_fraction_over_bound':.01,'parity_passed':False}
            if field=='denoiser':checks[0]['comparison']=bad
            elif field=='apg':apg[0]=bad
            elif field=='nonfinite':rows[0]['latent']['normalized_rmse']=float('nan')
            elif field=='forged':rows[-1]['latent']['parity_passed']=True
            elif field=='missing':checks.pop()
            with self.assertRaises(ValueError,msg=field):M.classify_trajectory_diagnostic(rows,apg,checks,audio)

    def test_explicit_mode_is_rejected_for_short_window_before_work(self):
        for script in ['export_coreai_bass_preview.py','prepare_coreai_preview_window.py']:
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--seconds','5.12','--diagnostic-trajectory'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('10.24',result.stderr)

    def fixture(self,root):
        from test_coreai_bass_preview import BassPreviewTests
        # NONEXECUTABLE files verify contracts/hashes only, never neural behavior.
        folder,source,condition=root/'preview',root/'source',root/'condition'
        ref=BassPreviewTests().fixture(folder,source,condition)
        for name in ref['artifact_sha256']:
            if name=='initial-noise.f32le' or name.startswith('step-'):(folder/name).write_bytes(bytes(4*LONG.frames*64))
            elif name.startswith('audio-'):(folder/name).write_bytes(bytes(8*LONG.samples))
        ref['artifact_sha256']={n:M.sha256(folder/n) for n in ref['artifact_sha256']}
        rows,apg,checks,audio=self.comparisons()
        times=[v[0] for v in struct.iter_unpack('<f',(folder/'timesteps.f32le').read_bytes())]
        for c in checks:
            name=f'step-{c["step"]}-latent-fp32-reference.f32le'
            c.update(phase='after_gold_step',timestep=times[c['step']],input_latent_file=name,
                     input_latent_sha256=ref['artifact_sha256'][name],conditioning_input='original_fp32_condition_reference')
        ref.update(profile=LONG.preview_bundle,stages=M.stage_plan(LONG),latent_frames=LONG.frames,condition_tokens=LONG.condition_tokens,
                   audio_seconds=LONG.seconds,reference_conditioning_scope='original_fp32_vs_storage16_conditioning',weight_storage_dtype=LONG.weight_storage_dtype,
                   host_trajectory_diagnostic_requested=True,host_denoiser_same_input=checks,host_comparisons=rows,
                   apg_same_input_vs_publisher=apg,host_audio_vs_fp32=audio)
        ref.update(M.classify_trajectory_diagnostic(rows,apg,checks,audio))
        (folder/'reference.json').write_text(json.dumps(ref))
        return folder,source,condition,ref

    def test_bundle_requires_explicit_permission_and_original_bounds_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            folder,source,condition,ref=self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError,'explicit'):M.verify_bundle(folder,source,condition,LONG)
            verified=M.verify_bundle(folder,source,condition,LONG,True)
            self.assertFalse(verified['host_numerical_gate_passed'])
            self.assertEqual(verified['relative_rmse_tolerance'],.001)
            with self.assertRaises(ValueError):M.verify_bundle(folder,source,condition,SHORT,True)

    def test_bundle_rejects_changed_diagnostic_input_time_or_false_status(self):
        for field in ['hash','time','status','tolerance']:
            with tempfile.TemporaryDirectory() as temp:
                folder,source,condition,ref=self.fixture(Path(temp))
                if field=='hash':ref['host_denoiser_same_input'][0]['input_latent_sha256']='0'*64
                elif field=='time':ref['host_denoiser_same_input'][0]['timestep']=.5
                elif field=='status':ref['host_export_status']='bounded_diagnostic'
                elif field=='tolerance':ref['relative_rmse_tolerance']=.01
                (folder/'reference.json').write_text(json.dumps(ref))
                with self.assertRaises(ValueError,msg=field):M.verify_bundle(folder,source,condition,LONG,True)

    def test_staging_requires_explicit_selector_and_retains_one_active_profile(self):
        from test_coreai_preview_window import PreviewWindowTests
        import stage_coreai_app_models as staging
        fixture=PreviewWindowTests()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'ModelCandidates';destination=root/'build/Probe.app/ModelCandidates'
            for window in [SHORT,LONG]:fixture.pipeline(source,window)
            staging.stage_resources(source,destination)
            saved=(destination/SHORT.preview_bundle/'payload').read_bytes()
            manifest=source/LONG.preview_bundle/'reference.json';ref=json.loads(manifest.read_text())
            ref['host_trajectory_diagnostic_requested']=True;manifest.write_text(json.dumps(ref))
            fixture.select(source,LONG)
            with self.assertRaisesRegex(ValueError,'explicit'):staging.stage_resources(source,destination)
            self.assertEqual((destination/SHORT.preview_bundle/'payload').read_bytes(),saved)
            selector=source/'active-preview.json';selection=json.loads(selector.read_text());selection['diagnostic_trajectory']=True
            selector.write_text(json.dumps(selection))
            ref['host_trajectory_diagnostic_requested']=False;manifest.write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'component-validated'):staging.plan_resources(source)
            ref['host_trajectory_diagnostic_requested']=True;manifest.write_text(json.dumps(ref))
            plan=staging.stage_resources(source,destination)
            self.assertTrue(plan['diagnostic_trajectory'])
            self.assertEqual(plan['selected_profile'],LONG.resource_profile)
            self.assertFalse((destination/SHORT.preview_bundle).exists())
            self.assertTrue((source/SHORT.preview_bundle).is_dir())
            with self.assertRaises(ValueError):staging.plan_resources(source,LONG.resource_profile)
            fixture.select(source,SHORT);selection=json.loads(selector.read_text());selection['diagnostic_trajectory']=True
            selector.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):staging.plan_resources(source)

if __name__=='__main__':unittest.main()
