"""Private export integrity with synthetic audio; no musical/model evidence."""
import hashlib
import json
import math
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import package_coreai_instrument_transition_review as P

RUNS = {'bass': '00000000-0000-4000-8000-000000000001', 'guitar': '00000000-0000-4000-8000-000000000002'}


def wav_bytes(planar, gain):
    # Independent transcription of the device's fixed58-byte Float32 WAV writer.
    frames = len(planar)//8
    result = bytearray(b'RIFF' + struct.pack('<I', 50+frames*8) + b'WAVEfmt ' +
                       struct.pack('<IHHIIHHH',18,3,2,48000,384000,8,32,0) +
                       b'fact' + struct.pack('<II',4,frames) + b'data' + struct.pack('<I',frames*8))
    result.extend(bytes(frames*8))
    # Fixtures contain exactly these two impulses; interleave explicitly.
    values = memoryview(planar).cast('f')
    for channel, sample in ((0, frames-10), (1, frames-5)):
        struct.pack_into('<f',result,58+8*sample+4*channel,values[channel*frames+sample]*gain)
    return result


class TransitionReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.container = self.root/'saved.xcappdata'
        self.documents = self.container/'AppData/Documents'
        self.folders = {}
        self.gain = struct.unpack('<f',struct.pack('<f',.36943486))[0]
        for role, run_id in RUNS.items():
            folder = self.documents/'InstrumentReference30'/role/run_id
            folder.mkdir(parents=True)
            self.folders[role] = folder
            for name, shape in P.shapes(role).items():
                with (folder/name).open('wb') as stream: stream.truncate(math.prod(shape)*4)
            raw = bytearray(P.FRAMES*8)
            struct.pack_into('<f',raw,4*(P.FRAMES-10),.25)
            struct.pack_into('<f',raw,4*(2*P.FRAMES-5),-.5)
            (folder/('raw-'+role+'.f32le')).write_bytes(raw)
            (folder/(role+'.wav')).write_bytes(wav_bytes(raw,self.gain))
            self.write_report(role)
            (self.documents/f'last-instrument-reference30-{role}.json').write_text(json.dumps({'run_id':run_id}))
        self.out = self.root/'review.zip'

    def write_report(self, role):
        folder = self.folders[role]
        report = (f'Runtime: CoreAI instrument30 preview; instrument {role}; run {RUNS[role]}\n'
                  'Revision: instrument30-shared-sampler-v1; synthetic transport fixture\n'
                  'BF16 learned DiT storage, FP32 compute/APG/I/O, FP32 VAE; original abs1e-4/rel1e-3/nRMSE1e-3\n'
                  'Reference30 execution: COMPLETED; steps50/50; sampling calls750/750; decoder calls6/6\n'
                  f'{role} numerical parity: NOT EVALUATED;\n'
                  f'Condition preset SHA256: {P.PRESETS[role]}; condition SHA256: '+ '1'*64 + '; all127 mask values valid\n'
                  'LIVE final latent SHA256: '+P.sha256(folder/'final-latent.f32le')+'\n'
                  f'Artifact SHA256: {role} '+P.sha256(folder/(role+'.wav'))+'; mix '+'2'*64+'; piano '+'3'*64+'\n'
                  f'Listening copies: shared gain0.36943486, diagnostic0.5piano+0.5{role};\n'
                  'WAV generated on device: YES\n')
        (folder/'report.txt').write_text(report)

    def test_package_preserves_run_identity_scope_and_binds_audio_to_device_wav(self):
        before = {p:P.sha256(p) for p in self.container.rglob('*') if p.is_file()}
        result = P.package(self.container,self.out,RUNS)
        self.assertEqual(len(result['artifacts']),16)
        self.assertEqual(result['new_inference_calls'],0)
        self.assertFalse(result['baseline_modified'])
        self.assertEqual({p:P.sha256(p) for p in before},before)
        with zipfile.ZipFile(self.out) as archive:
            self.assertEqual(len(archive.namelist()),21)
            self.assertEqual(json.loads(archive.read('review-manifest.json')),result)
            for role in P.ROLES:
                self.assertEqual(result['runs'][role]['numerical_parity'],'NOT EVALUATED')
                self.assertEqual(archive.read(role+'/report.txt'),(self.folders[role]/'report.txt').read_bytes())
                self.assertEqual(hashlib.sha256(archive.read(role+'/last8s-listening.wav')).hexdigest(),result['runs'][role]['last8s_wav_sha256'])
                self.assertEqual(len(archive.read(role+'/last8s-listening.wav')),58+8*48000*8)
            for name,entry in result['artifacts'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),entry['sha256'])
        self.assertLess(self.out.stat().st_size,P.MAX_ZIP)

    def test_latest_and_explicit_ids_resolve_same_folders(self):
        for role in P.ROLES:
            self.assertEqual(P.run_folder(self.container,role,'latest'),(self.folders[role],RUNS[role]))
            self.assertEqual(P.run_folder(self.documents,role,RUNS[role]),(self.folders[role],RUNS[role]))
        with self.assertRaisesRegex(ValueError,'Download the current'):
            P.run_folder(self.container,'bass',RUNS['guitar'])

    def test_nested_download_container_and_namespace_or_role_selection(self):
        downloads = self.root/'Downloads'
        target = downloads/'saved exports/com.nenz24.CoreAIProbe/latest.xcappdata'
        target.parent.mkdir(parents=True)
        self.container.rename(target)
        documents = target/'AppData/Documents'
        for role in P.ROLES:
            expected = documents/'InstrumentReference30'/role/RUNS[role]
            for source in (downloads, target.parent, documents/'InstrumentReference30', expected.parent):
                self.assertEqual(P.run_folder(source,role,RUNS[role]),(expected,RUNS[role]))
        # A search root may contain the output; actual container roots stay protected.
        result = P.package(downloads,downloads/'review.zip',RUNS)
        self.assertEqual(result['runs']['bass']['run_id'],RUNS['bass'])
        with self.assertRaisesRegex(ValueError,'outside'):
            P.package(downloads,target/'review.zip',RUNS)

    def test_identical_duplicate_downloads_resolve_but_changed_copies_are_ambiguous(self):
        copy = self.root/'second-copy'
        shutil.copytree(self.container,copy)
        expected = sorted([self.folders['bass'],copy/'AppData/Documents/InstrumentReference30/bass'/RUNS['bass']])[0]
        self.assertEqual(P.run_folder(self.root,'bass',RUNS['bass']),(expected,RUNS['bass']))
        changed = copy/'AppData/Documents/InstrumentReference30/bass'/RUNS['bass']/'final-momentum.f32le'
        with changed.open('r+b') as stream: stream.write(struct.pack('<f',1))
        with self.assertRaisesRegex(ValueError,'Different/incomplete copies'):
            P.run_folder(self.root,'bass',RUNS['bass'])

    def test_missing_run_lists_search_folder_and_available_ids_without_substitution(self):
        with self.assertRaises(ValueError) as failure:
            P.run_folder(self.container,'bass',RUNS['guitar'])
        message = str(failure.exception)
        self.assertIn('Searched folder: '+str(self.container),message)
        self.assertIn(RUNS['bass'],message)
        self.assertIn(RUNS['guitar'],message)
        self.assertIn('files present',message)
        self.assertFalse(self.out.exists())

    def test_distinct_latest_pointers_require_explicit_uuid(self):
        copy = self.root/'second-copy'
        shutil.copytree(self.container,copy)
        (copy/'AppData/Documents/last-instrument-reference30-bass.json').write_text(json.dumps({'run_id':RUNS['guitar']}))
        with self.assertRaisesRegex(ValueError,'No unique latest bass'):
            P.run_folder(self.root,'bass','latest')

    def test_list_runs_cli_is_read_only_and_skips_linked_containers(self):
        chosen = self.root/'chosen'
        chosen.mkdir()
        (chosen/'external-container').symlink_to(self.container,target_is_directory=True)
        self.assertEqual(P.discover_namespaces(chosen),[])
        before = {p:P.sha256(p) for p in self.container.rglob('*') if p.is_file()}
        result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/package_coreai_instrument_transition_review.py'),
                                 '--source',str(self.container),'--list-runs'],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn(RUNS['bass'],result.stdout)
        self.assertIn(RUNS['guitar'],result.stdout)
        self.assertFalse(self.out.exists())
        self.assertEqual({p:P.sha256(p) for p in before},before)

    def test_cli_missing_run_reports_selected_folder_without_traceback_or_output(self):
        chosen = self.root/'empty-downloads'
        chosen.mkdir()
        result = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/package_coreai_instrument_transition_review.py'),
                                 '--source',str(chosen),'--bass-run',RUNS['bass'],'--guitar-run',RUNS['guitar'],
                                 '--out',str(self.out)],text=True,capture_output=True)
        self.assertEqual(result.returncode,1)
        self.assertIn('Selected search folder: '+str(chosen),result.stdout)
        self.assertIn('Review not created:',result.stderr)
        self.assertIn('InstrumentReference30 folders found: 0',result.stderr)
        self.assertNotIn('Traceback',result.stderr)
        self.assertFalse(self.out.exists())

    def test_final_hash_mismatch_or_incomplete_run_does_not_publish(self):
        checkpoint = self.folders['bass']/'native-step-50-latent.f32le'
        with checkpoint.open('r+b') as stream: stream.write(struct.pack('<f',1))
        with self.assertRaisesRegex(ValueError,'checkpoint50'):
            P.package(self.container,self.out,RUNS)
        self.assertFalse(self.out.exists())
        with checkpoint.open('r+b') as stream: stream.write(struct.pack('<f',0))
        report = self.folders['guitar']/'report.txt'
        report.write_text(report.read_text().replace('execution: COMPLETED','execution: NOT COMPLETED'))
        with self.assertRaises(ValueError): P.package(self.container,self.out,RUNS)
        self.assertFalse(self.out.exists())

    def test_changed_raw_audio_is_rejected_despite_finite_correct_size(self):
        path = self.folders['bass']/'raw-bass.f32le'
        with path.open('r+b') as stream: stream.write(struct.pack('<f',.5))
        with self.assertRaisesRegex(ValueError,'Raw audio does not reproduce'):
            P.package(self.container,self.out,RUNS)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.instrument-transition-*')),[])

    def test_existing_output_and_container_are_preserved(self):
        self.out.write_bytes(b'existing')
        with self.assertRaises(FileExistsError): P.package(self.container,self.out,RUNS)
        self.assertEqual(self.out.read_bytes(),b'existing')
        with self.assertRaisesRegex(ValueError,'outside'):
            P.package(self.container,self.container/'review.zip',RUNS)

    def test_nonfinite_linked_and_truncated_tensor_do_not_publish(self):
        path = self.folders['bass']/'final-momentum.f32le'
        with path.open('r+b') as stream: stream.write(struct.pack('<f',float('nan')))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            P.package(self.container,self.out,RUNS)
        self.assertFalse(self.out.exists())
        path.unlink()
        path.symlink_to(self.folders['bass']/'final-latent.f32le')
        with self.assertRaisesRegex(ValueError,'linked'): P.package(self.container,self.out,RUNS)
        path.unlink()
        path.write_bytes(b'1234')
        with self.assertRaisesRegex(ValueError,'Wrong-size'): P.package(self.container,self.out,RUNS)

    def test_help_is_stdlib_only_and_does_not_create_output(self):
        r = subprocess.run([sys.executable,'-S',str(ROOT/'scripts/package_coreai_instrument_transition_review.py'),'--help'],
                           cwd=self.root,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--choose-container',r.stdout)
        self.assertFalse(self.out.exists())


if __name__ == '__main__': unittest.main()
