"""Refresh embedded notebook helpers and a small, reproducible GitHub bridge ZIP."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/11_colab_context_chunks.ipynb'
FRESH_NOTEBOOK = ROOT / 'notebooks/12_colab_fresh_context_90.ipynb'
DRUM_NOTEBOOK = ROOT / 'notebooks/13_colab_drum_probe.ipynb'
DRUM_FULL_NOTEBOOK = ROOT / 'notebooks/14_colab_drum_full_song.ipynb'
DRUM_CHUNK_NOTEBOOK = ROOT / 'notebooks/15_colab_drum_context_90.ipynb'
DRUM_KIT_NOTEBOOK = ROOT / 'notebooks/16_colab_drum_kit_reference.ipynb'
DRUM_KIT_SONG_NOTEBOOK = ROOT / 'notebooks/17_colab_drum_shared_kit_song.ipynb'
GUITAR_NOTEBOOK = ROOT / 'notebooks/18_colab_clean_electric_guitar.ipynb'
ARCHIVE = ROOT / 'bridges/ace_context_tools.zip'
BEGIN = '# BEGIN EMBEDDED CONTEXT HELPERS\n'
END = '# END EMBEDDED CONTEXT HELPERS\n'
FILES = {
    'ace_context_chunks.py': 'scripts/ace_context_chunks.py',
    'electric_full_song.py': 'scripts/electric_full_song.py',
    'ace-context-chunks.md': 'docs/ace-context-chunks.md',
    'electric-context.json': 'configs/electric-context.json',
    'requirements-context-chunks.txt': 'requirements-context-chunks.txt',
    '11_colab_context_chunks.ipynb': 'notebooks/11_colab_context_chunks.ipynb',
    '12_colab_fresh_context_90.ipynb': 'notebooks/12_colab_fresh_context_90.ipynb',
    '13_colab_drum_probe.ipynb': 'notebooks/13_colab_drum_probe.ipynb',
    '14_colab_drum_full_song.ipynb': 'notebooks/14_colab_drum_full_song.ipynb',
    '15_colab_drum_context_90.ipynb': 'notebooks/15_colab_drum_context_90.ipynb',
    '16_colab_drum_kit_reference.ipynb': 'notebooks/16_colab_drum_kit_reference.ipynb',
    '17_colab_drum_shared_kit_song.ipynb': 'notebooks/17_colab_drum_shared_kit_song.ipynb',
    '18_colab_clean_electric_guitar.ipynb': 'notebooks/18_colab_clean_electric_guitar.ipynb',
}


def encode(data):
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', filename='', mtime=0) as stream:
        stream.write(data)
    return {'sha256': hashlib.sha256(data).hexdigest(),
            'gzip_base64': base64.b64encode(compressed.getvalue()).decode()}


def fresh_notebook():
    paths = list((ROOT/'tools').rglob('*.py')) + list((ROOT/'configs').glob('*.json'))
    paths += [p for p in (ROOT/'scripts').glob('*.py')
              if p.name not in {'build_context_bridge.py', 'piano_bass.py', 'bass_performance.py'}]
    paths.append(ROOT/'requirements-context-chunks.txt')
    payload = {str(p.relative_to(ROOT)): encode(p.read_bytes()) for p in sorted(paths)}
    book = json.loads(FRESH_NOTEBOOK.read_text())
    begin, end = '# BEGIN EMBEDDED BRIDGE SOURCE\n', '# END EMBEDDED BRIDGE SOURCE\n'
    matches = [c for c in book['cells'] if c['cell_type']=='code' and begin in ''.join(c['source'])]
    if len(matches)!=1: raise ValueError('Expected one source snapshot cell')
    source=''.join(matches[0]['source']);before,remaining=source.split(begin,1);_,after=remaining.split(end,1)
    matches[0]['source']=(before+begin+'BRIDGE_SOURCE = '+json.dumps(payload,indent=2)+'\n'+end+after).splitlines(keepends=True)
    for c in book['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),str(FRESH_NOTEBOOK),'exec')
            if c['outputs'] or c['execution_count'] is not None: raise ValueError('Clear notebook outputs before publishing')
    return (json.dumps(book,indent=1,ensure_ascii=False)+'\n').encode()


def drum_notebook(notebook=DRUM_NOTEBOOK, helpers=('drum_probe.py','electric_full_song.py')):
    paths = helpers if isinstance(helpers, dict) else {name: 'scripts/' + name for name in helpers}
    payload={name:encode((ROOT/path).read_bytes()) for name, path in paths.items()}
    book=json.loads(notebook.read_text())
    cells=[c for c in book['cells'] if c['cell_type']=='code' and BEGIN in ''.join(c['source'])]
    if len(cells)!=1:raise ValueError('Expected one drum helper cell')
    source=''.join(cells[0]['source']);before,remainder=source.split(BEGIN,1);_,after=remainder.split(END,1)
    cells[0]['source']=(before+BEGIN+'EMBEDDED_HELPERS = '+json.dumps(payload,indent=2)+'\n'+END+after).splitlines(keepends=True)
    for c in book['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),str(notebook),'exec')
            if c['outputs'] or c['execution_count'] is not None:raise ValueError('Clear drum notebook outputs')
    return (json.dumps(book,indent=1,ensure_ascii=False)+'\n').encode()


def build(check=False):
    helpers = {}
    for name in ('ace_context_chunks.py', 'electric_full_song.py'):
        data = (ROOT / FILES[name]).read_bytes()
        helpers[name] = encode(data)
    book = json.loads(NOTEBOOK.read_text())
    cells = [c for c in book['cells'] if c['cell_type'] == 'code' and BEGIN in ''.join(c['source'])]
    if len(cells) != 1:
        raise ValueError('Expected exactly one embedded-helper cell')
    source = ''.join(cells[0]['source'])
    before, remainder = source.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    cells[0]['source'] = (before + BEGIN + 'EMBEDDED_HELPERS = ' + json.dumps(helpers, indent=2)
                          + '\n' + END + after).splitlines(keepends=True)
    notebook_bytes = (json.dumps(book, indent=1, ensure_ascii=False) + '\n').encode()
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Publish notebooks without execution outputs')
    fresh_bytes = fresh_notebook()
    drum_bytes = drum_notebook()
    drum_full_bytes = drum_notebook(DRUM_FULL_NOTEBOOK)
    drum_chunk_bytes = drum_notebook(DRUM_CHUNK_NOTEBOOK,('electric_full_song.py','drum_probe.py','ace_context_chunks.py'))
    drum_kit_bytes = drum_notebook(DRUM_KIT_NOTEBOOK,('electric_full_song.py','drum_probe.py','ace_context_chunks.py','drum_kit_reference.py'))
    drum_kit_song_bytes = drum_notebook(DRUM_KIT_SONG_NOTEBOOK,('electric_full_song.py','drum_probe.py','ace_context_chunks.py','drum_kit_song.py'))
    guitar_bytes = drum_notebook(GUITAR_NOTEBOOK, {
        'electric_full_song.py': 'scripts/electric_full_song.py',
        'guitar_probe.py': 'scripts/guitar_probe.py',
        'guitar_run.py': 'tools/stem_eval/run.py',
    })
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source_path in sorted(FILES.items()):
            data = (notebook_bytes if name=='11_colab_context_chunks.ipynb' else
                    fresh_bytes if name=='12_colab_fresh_context_90.ipynb' else
                    drum_bytes if name=='13_colab_drum_probe.ipynb' else
                    drum_full_bytes if name=='14_colab_drum_full_song.ipynb' else
                    drum_chunk_bytes if name=='15_colab_drum_context_90.ipynb' else
                    drum_kit_bytes if name=='16_colab_drum_kit_reference.ipynb' else
                    drum_kit_song_bytes if name=='17_colab_drum_shared_kit_song.ipynb' else
                    guitar_bytes if name=='18_colab_clean_electric_guitar.ipynb' else (ROOT/source_path).read_bytes())
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    archive_bytes = buffer.getvalue()
    if check:
        if NOTEBOOK.read_bytes() != notebook_bytes or FRESH_NOTEBOOK.read_bytes()!=fresh_bytes or DRUM_NOTEBOOK.read_bytes()!=drum_bytes or DRUM_FULL_NOTEBOOK.read_bytes()!=drum_full_bytes or DRUM_CHUNK_NOTEBOOK.read_bytes()!=drum_chunk_bytes or DRUM_KIT_NOTEBOOK.read_bytes()!=drum_kit_bytes or DRUM_KIT_SONG_NOTEBOOK.read_bytes()!=drum_kit_song_bytes or GUITAR_NOTEBOOK.read_bytes()!=guitar_bytes or not ARCHIVE.is_file() or ARCHIVE.read_bytes() != archive_bytes:
            raise ValueError('Bridge artifacts are stale; run scripts/build_context_bridge.py')
    else:
        NOTEBOOK.write_bytes(notebook_bytes)
        FRESH_NOTEBOOK.write_bytes(fresh_bytes)
        DRUM_NOTEBOOK.write_bytes(drum_bytes)
        DRUM_FULL_NOTEBOOK.write_bytes(drum_full_bytes)
        DRUM_CHUNK_NOTEBOOK.write_bytes(drum_chunk_bytes)
        DRUM_KIT_NOTEBOOK.write_bytes(drum_kit_bytes)
        DRUM_KIT_SONG_NOTEBOOK.write_bytes(drum_kit_song_bytes)
        GUITAR_NOTEBOOK.write_bytes(guitar_bytes)
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(archive_bytes)
    print(('Verified' if check else 'Built'), ARCHIVE.relative_to(ROOT), len(archive_bytes), 'bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
