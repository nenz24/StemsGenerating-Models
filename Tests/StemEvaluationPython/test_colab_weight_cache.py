"""Portable CPU-only weight backup, trusted restoration, and fresh Colab modes."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools/stem_eval')]
import colab_weight_cache as cache


class ColabWeightCacheTests(unittest.TestCase):
    def fixture(self, base):
        root=base/'ace_context_source_old';payloads={
            'checkpoints/acestep-v15-base/model.safetensors':b'verified fixture weights'*2048,
            'checkpoints/acestep-v15-base/config.json':b'{"fixture":true}\n'}
        records=[]
        for index,(name,data) in enumerate(payloads.items()):
            path=root/'vendor/ACE-Step-1.5'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            algorithm='sha256' if index==0 else 'git-blob-sha1'
            digest=hashlib.sha256(data).hexdigest() if index==0 else hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            records.append({'destination':name,'size':len(data),'algorithm':algorithm,'digest':digest})
        manifest=base/'assets.json';manifest.write_text(json.dumps({'model':'acestep-v15-base','files':records}))
        (root/'.env').write_text('fixture excluded file')
        return root,manifest,payloads

    def test_export_restore_cpu_only_preserves_source_and_reuses_verified_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old,manifest,payloads=self.fixture(base);archive=base/'cache.tar'
            self.assertFalse((old/'vendor/ACE-Step-1.5/.venv').exists())
            self.assertEqual(cache.discover(base,manifest),old)
            with patch('subprocess.run',side_effect=AssertionError('CPU transfer must not launch CUDA/installer')):
                report=cache.export(old,archive,manifest)
                restored=cache.restore(base/'new',archive,manifest)
                repeated=cache.restore(base/'new',archive,manifest)
            self.assertFalse(report['requires_gpu']);self.assertEqual(restored['restored_files'],2)
            self.assertEqual(repeated['restored_files'],0);self.assertEqual(repeated['reused_files'],2)
            with tarfile.open(archive) as tar:
                self.assertEqual(set(tar.getnames()),{cache.HEADER}|{'weights/'+name for name in payloads})
            for name,data in payloads.items():
                self.assertEqual((old/'vendor/ACE-Step-1.5'/name).read_bytes(),data)
                self.assertEqual((base/'new/vendor/ACE-Step-1.5'/name).read_bytes(),data)
            before=archive.read_bytes()
            with self.assertRaises(FileExistsError):cache.export(old,archive,manifest)
            self.assertEqual(archive.read_bytes(),before)

    def write_archive(self,path,manifest,payloads,*,malicious=None,corrupt=False):
        data,records,digest=cache.manifest_data(manifest)
        with tarfile.open(path,'w') as tar:
            header=json.dumps({'schema_version':1,'model':data['model'],'manifest_sha256':digest}).encode()
            info=tarfile.TarInfo(cache.HEADER);info.size=len(header);tar.addfile(info,io.BytesIO(header))
            if malicious:
                info=tarfile.TarInfo(malicious)
                if malicious.startswith('weights/checkpoints/'):
                    info.type=tarfile.SYMTYPE;info.linkname='../../outside'
                    tar.addfile(info)
                else:
                    info.size=1;tar.addfile(info,io.BytesIO(b'x'))
                return
            for index,(name,payload) in enumerate(payloads.items()):
                if corrupt and index==0:payload=b'x'+payload[1:]
                info=tarfile.TarInfo('weights/'+name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))

    def test_bad_hash_never_becomes_a_model_file_and_good_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old,manifest,payloads=self.fixture(base)
            bad=base/'bad.tar';self.write_archive(bad,manifest,payloads,corrupt=True)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):cache.restore(base/'new',bad,manifest)
            target=base/'new/vendor/ACE-Step-1.5'/next(iter(payloads))
            self.assertFalse(target.exists())
            good=base/'good.tar';cache.export(old,good,manifest);cache.restore(base/'new',good,manifest)
            self.assertEqual(target.read_bytes(),next(iter(payloads.values())))
            target.write_bytes(b'preserve this existing changed file')
            with self.assertRaises(ValueError):cache.restore(base/'new',good,manifest)
            self.assertEqual(target.read_bytes(),b'preserve this existing changed file')

    def test_paths_links_and_wrong_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old,manifest,payloads=self.fixture(base)
            for index,name in enumerate(('../outside','weights/../../outside','weights/'+next(iter(payloads)))):
                archive=base/f'bad_{index}.tar';self.write_archive(archive,manifest,payloads,malicious=name)
                with self.assertRaises(ValueError):cache.restore(base/f'target{index}',archive,manifest)
            self.assertFalse((base/'outside').exists())
            good=base/'good.tar';cache.export(old,good,manifest)
            altered=base/'changed.json';altered.write_text(manifest.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'trusted current manifest'):
                cache.restore(base/'changed',good,altered)

    def test_incomplete_source_cannot_create_a_completed_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old,manifest,payloads=self.fixture(base)
            source=old/'vendor/ACE-Step-1.5'/next(iter(payloads));original=source.read_bytes()
            source.write_bytes(b'x'+original[1:])
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):cache.export(old,base/'bad.tar',manifest)
            self.assertFalse((base/'bad.tar').exists());self.assertFalse((base/'bad.tar.partial').exists())
            partial=base/'existing.tar.partial';partial.write_bytes(b'pre-existing transfer')
            with self.assertRaises(FileExistsError):cache.export(old,base/'existing.tar',manifest)
            self.assertEqual(partial.read_bytes(),b'pre-existing transfer')
            source.unlink()
            self.assertFalse(cache.inventory(old,manifest)['complete_by_size'])
            with self.assertRaisesRegex(ValueError,'asset missing/wrong size'):
                cache.export(old,base/'missing.tar',manifest)
            self.assertFalse((base/'missing.tar').exists())

    def test_notebook_cpu_backup_cell_runs_without_gpu_or_installer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old,manifest,payloads=self.fixture(base)
            book=json.loads((ROOT/'notebooks/22_colab_cache_transfer.ipynb').read_text())
            scope={'ACTION':'backup','DRY_RUN':False,'USE_DRIVE':False,'ARCHIVE_PATH':base/'backup.tar',
                   'BASE':base,'cache_info':cache.inventory(old,manifest),'ROOT':old,'ASSETS':manifest,
                   'cache':cache,'json':json,'Path':Path}
            with patch('subprocess.run',side_effect=AssertionError('no CUDA/setup in CPU backup')):
                exec(compile(''.join(book['cells'][3]['source']),'CPU-backup-cell','exec'),scope)
            self.assertTrue((base/'backup.tar').is_file())

    def test_new_vm_notebook_dry_run_restores_source_and_layer_plan_without_downloads(self):
        script=r'''
import contextlib,io,json,subprocess,sys,types,wave
from pathlib import Path
base=Path(sys.argv[1]);book=json.loads(Path(sys.argv[2]).read_text())
source=base/'piano.wav'
with wave.open(str(source),'wb') as audio:
 audio.setparams((2,2,32000,0,'NONE','not compressed'));audio.writeframes(b'\0'*(91*32000*4))
def forbidden(*args,**kwargs):raise AssertionError('no Drive/upload/download in dry run')
module=types.ModuleType('google.colab');module.files=types.SimpleNamespace(upload=forbidden,download=forbidden);module.drive=types.SimpleNamespace(mount=forbidden)
sys.modules['google.colab']=module
original_run=subprocess.run
def run(command,*args,**kwargs):
 if Path(command[0]).name not in ('ffmpeg','ffprobe'):raise AssertionError('unexpected setup/CUDA command: '+str(command))
 return original_run(command,*args,**kwargs)
subprocess.run=run
scope={'WORKSPACE_BASE':base/'new_vm','SOURCE_PATH':source}
with contextlib.redirect_stdout(io.StringIO()):
 for cell in book['cells']:
  if cell['cell_type']=='code':
   code=''.join(cell['source']).replace('DRY_RUN = False','DRY_RUN = True').replace('ACTION = "backup"','ACTION = "prepare"')
   exec(compile(code,'cache-notebook-prepare','exec'),scope)
assert scope['OLD_ROOT'] is None
assert scope['report']['status']=='planned' and scope['report']['model_calls_requested']==2
assert not (scope['ROOT']/'vendor').exists()
assert (scope['CODE_ROOT']/'scripts/guitar_run.py').is_file()
assert (scope['CODE_ROOT']/'references/approved_guitar_style.mp3').is_file()
assert not list(scope['LAYERS_OUT'].rglob('guitar.wav'))
# CPU backup dry-run must also skip every CUDA/setup/generation cell.
scope={'WORKSPACE_BASE':base/'cpu_vm'}
with contextlib.redirect_stdout(io.StringIO()):
 for cell in book['cells']:
  if cell['cell_type']=='code':
   code=''.join(cell['source']).replace('DRY_RUN = False','DRY_RUN = True')
   exec(compile(code,'cache-notebook-backup','exec'),scope)
assert scope['ACTION']=='backup' and 'LAYERS_OUT' not in scope
assert not (scope['ROOT']/'vendor').exists()
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-c',script,tmp,str(ROOT/'notebooks/22_colab_cache_transfer.ipynb')],cwd=tmp,text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
