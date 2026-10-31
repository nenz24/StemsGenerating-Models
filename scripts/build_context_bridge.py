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
}


def build(check=False):
    helpers = {}
    for name in ('ace_context_chunks.py', 'electric_full_song.py'):
        data = (ROOT / FILES[name]).read_bytes()
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode='wb', filename='', mtime=0) as stream:
            stream.write(data)
        helpers[name] = {'sha256': hashlib.sha256(data).hexdigest(),
                         'gzip_base64': base64.b64encode(compressed.getvalue()).decode()}
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
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source_path in sorted(FILES.items()):
            data = notebook_bytes if name.endswith('.ipynb') else (ROOT / source_path).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    archive_bytes = buffer.getvalue()
    if check:
        if NOTEBOOK.read_bytes() != notebook_bytes or not ARCHIVE.is_file() or ARCHIVE.read_bytes() != archive_bytes:
            raise ValueError('Bridge artifacts are stale; run scripts/build_context_bridge.py')
    else:
        NOTEBOOK.write_bytes(notebook_bytes)
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE.write_bytes(archive_bytes)
    print(('Verified' if check else 'Built'), ARCHIVE.relative_to(ROOT), len(archive_bytes), 'bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    build(parser.parse_args().check)
