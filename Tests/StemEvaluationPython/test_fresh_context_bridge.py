"""Exercise the fresh-VM notebook without an old bridge checkout or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FreshContextBridgeTests(unittest.TestCase):
    def test_empty_vm_restore_plan_rerun_and_tamper_detection(self):
        notebook=ROOT/'notebooks/12_colab_fresh_context_90.ipynb'
        # An isolated process prevents imports from the current checkout from
        # accidentally masking files missing in the embedded source snapshot.
        script='''
import contextlib,io,json,sys,types,wave
from pathlib import Path
base=Path(sys.argv[1]);book=json.loads(Path(sys.argv[2]).read_text())
audio=base/'piano.wav'
with wave.open(str(audio),'wb') as w:
 w.setparams((2,2,32000,0,'NONE','not compressed'));w.writeframes(b'\\0'*(91*32000*4))
uploads=[]
def upload():
 uploads.append(1);return {'piano.wav':audio.read_bytes()}
m=types.ModuleType('google.colab');m.files=types.SimpleNamespace(upload=upload,download=lambda *a:(_ for _ in ()).throw(AssertionError('unexpected download')))
sys.modules['google.colab']=m
scope={'WORKSPACE_BASE':base}
def execute():
 with contextlib.redirect_stdout(io.StringIO()):
  for cell in book['cells']:
   if cell['cell_type']=='code':
    source=''.join(cell['source']).replace('DRY_RUN = False','DRY_RUN = True')
    exec(compile(source,'fresh-notebook-cell','exec'),scope)
execute()
root=scope['ROOT'];out=scope['CONTEXT_OUT']
assert root.is_relative_to(base) and root!=Path.cwd()
assert not (root/'.git').exists() and not (root/'vendor').exists()
assert scope['plan']['context_cap_seconds']==90 and len(scope['plan']['chunks'])==2
assert not list(out.rglob('electric_bass.wav'))
assert len(uploads)==1
execute()
assert scope['ROOT']==root and scope['CONTEXT_OUT']==out and len(uploads)==1
damaged=root/'scripts/electric_full_song.py';damaged.write_text('user change')
try:execute()
except AssertionError as e:assert 'Source lokal berubah' in str(e)
else:raise AssertionError('must reject changed source')
assert damaged.read_text()=='user change'
print('PASS: fresh source restore, actual ffmpeg plan, rerun, no old checkout, no inference, preserve changed files')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-c',script,tmp,str(notebook)],cwd=tmp,
                                  text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_default_requires_gpu_before_install_or_upload(self):
        from unittest.mock import patch
        book=json.loads((ROOT/'notebooks/12_colab_fresh_context_90.ipynb').read_text())
        first=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
        with tempfile.TemporaryDirectory() as tmp,patch('shutil.which',return_value=None):
            with self.assertRaisesRegex(AssertionError,'Aktifkan GPU'):
                exec(compile(first,'fresh-startup','exec'),{'WORKSPACE_BASE':tmp})


if __name__=='__main__':unittest.main()
