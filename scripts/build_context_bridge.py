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
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source_path in sorted(FILES.items()):
            data = (notebook_bytes if name=='11_colab_context_chunks.ipynb' else
                    fresh_bytes if name=='12_colab_fresh_context_90.ipynb' else (ROOT/source_path).read_bytes())
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    archive_bytes = buffer.getvalue()
    if check:
        if NOTEBOOK.read_bytes() != notebook_bytes or FRESH_NOTEBOOK.read_bytes()!=fresh_bytes or not ARCHIVE.is_file() or ARCHIVE.read_bytes() != archive_bytes:
            raise ValueError('Bridge artifacts are stale; run scripts/build_context_bridge.py')
    else:
        NOTEBOOK.write_bytes(notebook_bytes)
        FRESH_NOTEBOOK.write_bytes(fresh_bytes)
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(archive_bytes)
    print(('Verified' if check else 'Built'), ARCHIVE.relative_to(ROOT), len(archive_bytes), 'bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
