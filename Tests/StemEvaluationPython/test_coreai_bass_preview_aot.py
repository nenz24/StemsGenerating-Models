"""AOT argument/packaging integrity tests; no native compilation or inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_coreai_bass_preview_aot as M


class BassPreviewAOTTests(unittest.TestCase):
    def test_command_matches_supplied_metal_27b47_help_and_one_architecture(self):
        cmd=M.compile_command(Path('/source/front.aimodel'),Path('/work/output'),'h18p')
        self.assertEqual(cmd,['xcrun','coreai-build','compile','/source/front.aimodel','--output','/work/output',
                             '--platform','iOS','--min-deployment-version','27.0','--preferred-compute','none','--architecture','h18p'])
        self.assertNotIn('cpuOnly',cmd);self.assertNotIn('cpu',cmd)
        self.assertEqual(cmd.count('--architecture'),1)
        self.assertEqual(len(M.expected_assets('h18p')),16)
        with self.assertRaises(ValueError):M.expected_assets('../../other')

    def fixture(self,root):
        source=root/'source';source.mkdir();(source/'reference.json').write_text('source identity fixture; no executable model')
        compiled={}
        for i,(name,variant) in enumerate(M.expected_assets('h18p').items()):
            path=root/variant
            if i%2:
                path.mkdir();(path/'payload.ir').write_bytes(b'NOT an executable native model; integrity-only fixture')
            else:path.write_bytes(b'NOT an executable native model; integrity-only fixture')
            compiled[name]=path
        return source,compiled

    def test_package_copies_and_hashes_all_variants_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,compiled=self.fixture(root);out=root/'published'
            ref=M.publish_bundle(compiled,source,out,'h18p',{'native_compilation_invoked':False,'scope':'integrity fixture'})
            self.assertEqual(M.verify_aot_bundle(out,source),ref)
            self.assertEqual(ref['runtime_policy'],'default');self.assertEqual(ref['preferred_compute'],'none')
            self.assertFalse(ref['compiler']['native_compilation_invoked'])
            first=out/ref['assets']['front.aimodel'];first.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):M.verify_aot_bundle(out,source)

    def test_wrong_arch_policy_stale_source_and_partial_asset_set_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,compiled=self.fixture(root);out=root/'published'
            ref=M.publish_bundle(compiled,source,out,'h18p',{'native_compilation_invoked':False})
            for key,value in [('architecture','other'),('runtime_policy','cpuOnly'),('preferred_compute','gpu'),('compiled_asset_count',15)]:
                altered={**ref,key:value};(out/'reference.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):M.verify_aot_bundle(out,source)
            (out/'reference.json').write_text(json.dumps(ref))
            (source/'reference.json').write_text('different source bundle')
            with self.assertRaisesRegex(ValueError,'source mismatch'):M.verify_aot_bundle(out,source)
            del compiled['decoder.aimodel']
            with self.assertRaisesRegex(ValueError,'Incomplete'):M.publish_bundle(compiled,source,root/'partial','h18p',{})
            self.assertFalse((root/'partial').exists())

    def test_verified_work_receipt_resumes_without_running_a_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';work.mkdir()
            self.assertIsNone(M.verified_work_result(work,'identity'))
            path=work/'output.aimodelc';path.write_bytes(b'NONEXECUTABLE integrity fixture')
            receipt={'identity':'identity','relative_path':path.name,'payload_sha256':M.payload_hashes(path)}
            (work/'receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(M.verified_work_result(work,'identity'),path.resolve())
            with self.assertRaises(ValueError):M.verified_work_result(work,'wrong')
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):M.verified_work_result(work,'identity')
            receipt['relative_path']='../outside.aimodelc';(work/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):M.verified_work_result(work,'identity')

    def test_existing_output_preserved_and_shell_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);marker=path/'keep';marker.write_text('original')
            r=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_coreai_bass_preview_aot.py'),'--out',tmp],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(r.returncode,0);self.assertIn('preserved',r.stderr)
            self.assertEqual(marker.read_text(),'original')
        r=subprocess.run(['bash','-n',str(ROOT/'scripts/prepare_coreai_bass_preview_aot.sh')],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_compiler_linked_payload_is_materialized_before_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';work.mkdir();artifact=work/'front.h18p.aimodelc';artifact.mkdir()
            weights=root/'source-weights';weights.mkdir();(weights/'data').write_bytes(b'NONEXECUTABLE fixture')
            (artifact/'weights').symlink_to(weights,target_is_directory=True)
            result=M.standalone_result(artifact,work)
            self.assertFalse((result/'weights').is_symlink())
            self.assertEqual((result/'weights/data').read_bytes(),b'NONEXECUTABLE fixture')
            self.assertIn('weights/data',M.payload_hashes(result))


if __name__=='__main__':unittest.main()
