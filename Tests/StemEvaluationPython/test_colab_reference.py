"""Colab bootstrap handles /content-style CWD, upload packages and safe reuse without GPU."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[2]
BOOK=ROOT/'notebooks/09_colab_reference.ipynb'


class ColabReferenceTests(unittest.TestCase):
    def make_package(self,path,unsafe=False):
        data={'source/scripts/cuda_reference.py':b'# committed fixture\n',
              'reference/prepared/source.wav':b'fixture audio bytes, no inference'}
        if unsafe:data['../outside.txt']=b'not allowed'
        manifest={'schema_version':1,'bridge_commit':'fixture-commit','source_sha256':hashlib.sha256(data['reference/prepared/source.wav']).hexdigest(),
                  'files':[{'path':name,'sha256':hashlib.sha256(value).hexdigest()} for name,value in data.items()]}
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in data.items():archive.writestr(name,value)
            archive.writestr('bundle.json',json.dumps(manifest))

    def bootstrap(self,archive,cwd):
        book=json.loads(BOOK.read_text())
        source=''.join(book['cells'][1]['source']).replace('DRY_RUN = False','DRY_RUN = True',1)
        source=source.replace('BUNDLE_ZIP = None',f'BUNDLE_ZIP = Path({str(archive)!r})',1)
        namespace={}
        with patch.object(Path,'cwd',return_value=cwd):exec(compile(source,'colab-bootstrap','exec'),namespace)
        return namespace['COLAB_BUNDLE']

    def test_upload_package_extracts_under_unrelated_cwd_and_reuses_installed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cwd=root/'content';cwd.mkdir()
            archive=root/'input.zip';self.make_package(archive)
            unpacked=self.bootstrap(archive,cwd)
            self.assertTrue(unpacked.is_relative_to(cwd))
            self.assertTrue((unpacked/'source/scripts/cuda_reference.py').is_file())
            cache=unpacked/'source/.runtime-cache/existing.bin';cache.parent.mkdir();cache.write_bytes(b'keep cache')
            self.assertEqual(self.bootstrap(archive,cwd),unpacked)
            self.assertEqual(cache.read_bytes(),b'keep cache')

    def test_archive_traversal_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cwd=root/'content';cwd.mkdir()
            archive=root/'input.zip';self.make_package(archive,unsafe=True)
            with self.assertRaises(AssertionError):self.bootstrap(archive,cwd)
            self.assertEqual(list(cwd.iterdir()),[])
            self.assertFalse((root/'outside.txt').exists())

    def test_changed_existing_snapshot_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'input.zip';self.make_package(archive)
            unpacked=self.bootstrap(archive,root)
            script=unpacked/'source/scripts/cuda_reference.py';script.write_bytes(b'local change')
            with self.assertRaisesRegex(AssertionError,'Snapshot lokal berubah'):self.bootstrap(archive,root)
            self.assertEqual(script.read_bytes(),b'local change')

    def test_old_feedback_zip_gets_a_clear_message_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'feedback.zip'
            with zipfile.ZipFile(archive,'w') as bundle:bundle.writestr('manifest.json','{}')
            with self.assertRaisesRegex(AssertionError,'bukan feedback.zip'):self.bootstrap(archive,root)
            self.assertFalse(list(root.glob('songmap_cuda_*')))

    def test_colab_upload_relative_filename_uses_the_runtime_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd=Path(directory);archive=cwd/'cuda_reference.zip';self.make_package(archive)
            files=types.SimpleNamespace(upload=lambda:{archive.name:archive.read_bytes()})
            modules={'google':types.ModuleType('google'),'google.colab':types.SimpleNamespace(files=files)}
            book=json.loads(BOOK.read_text())
            source=''.join(book['cells'][1]['source']).replace('DRY_RUN = False','DRY_RUN = True',1)
            namespace={};previous=Path.cwd()
            try:
                os.chdir(cwd)
                with patch.dict(sys.modules,modules):exec(compile(source,'colab-upload','exec'),namespace)
            finally:os.chdir(previous)
            self.assertTrue((namespace['COLAB_BUNDLE']/'reference/prepared/source.wav').is_file())

    def test_notebook_upload_root_and_download_are_explicit_with_no_saved_outputs(self):
        book=json.loads(BOOK.read_text())
        codes=[''.join(cell['source']) for cell in book['cells'] if cell['cell_type']=='code']
        self.assertIn('files.upload()',codes[0])
        self.assertIn('ROOT = COLAB_BUNDLE / "source"',codes[1])
        self.assertIn('files.download(str(result_zip))',codes[-1])
        self.assertEqual(sum('DRY_RUN = False' in code.split('\n') for code in codes),1)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                compile(''.join(cell['source']),'colab-cell','exec')


if __name__=='__main__':unittest.main()
