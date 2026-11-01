"""Shell setup environment regression; tool fixtures do not simulate CoreAI inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DecoderPreparePathTests(unittest.TestCase):
    def run_fixture(self, reuse):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);repo = base / 'repo with spaces';(repo / 'scripts').mkdir(parents=True)
            shutil.copyfile(ROOT / 'scripts/prepare_coreai_decoder.sh', repo / 'scripts/prepare_coreai_decoder.sh')
            tools = base / 'bin';tools.mkdir();log = base / 'calls.jsonl'
            (tools / 'uname').write_text('#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
            (tools / 'uname').chmod(0o755)
            python_fixture = '''import json,os,sys
from pathlib import Path
root=Path(os.environ['PREP_FIXTURE_ROOT'])
for name,suffix in [('UV_PYTHON_INSTALL_DIR','python'),('UV_CACHE_DIR','cache'),('UV_PYTHON_BIN_DIR','bin')]:
 assert os.environ[name]==str(root/'.runtime-cache/uv'/suffix)
 assert Path(os.environ[name]).is_dir()
assert os.environ['HOME']==os.environ['PREP_ORIGINAL_HOME']
with open(os.environ['PREP_FIXTURE_LOG'],'a') as f:f.write(json.dumps(['python',sys.argv[1:]])+'\\n')
'''
            uv_fixture = '''import json,os,sys
from pathlib import Path
root=Path(os.environ['PREP_FIXTURE_ROOT'])
for name,suffix in [('UV_PYTHON_INSTALL_DIR','python'),('UV_CACHE_DIR','cache'),('UV_PYTHON_BIN_DIR','bin')]:
 assert os.environ[name]==str(root/'.runtime-cache/uv'/suffix)
 assert Path(os.environ[name]).is_dir()
assert os.environ['HOME']==os.environ['PREP_ORIGINAL_HOME']
with open(os.environ['PREP_FIXTURE_LOG'],'a') as f:f.write(json.dumps(['uv',sys.argv[1:]])+'\\n')
if sys.argv[1]=='venv':
 target=Path(sys.argv[-1])/'bin/python';target.parent.mkdir(parents=True)
 target.write_text(os.environ['PREP_PYTHON_FIXTURE']);target.chmod(0o755)
'''
            prefix = '#!' + sys.executable + '\n'
            (tools / 'uv').write_text(prefix + uv_fixture);(tools / 'uv').chmod(0o755)
            if reuse:
                target = repo / '.runtime-cache/coreai-decoder-venv/bin/python';target.parent.mkdir(parents=True)
                target.write_text(prefix + python_fixture);target.chmod(0o755)
                (repo / 'prototypes/ios27-coreai/ModelCandidates/decoder-pretrained').mkdir(parents=True)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       UV_PYTHON_INSTALL_DIR='/unwritable/managed-python', UV_CACHE_DIR='/unwritable/cache',
                       UV_PYTHON_BIN_DIR='/unwritable/bin', PREP_FIXTURE_ROOT=str(repo),
                       PREP_ORIGINAL_HOME=os.environ['HOME'], PREP_FIXTURE_LOG=str(log),
                       PREP_PYTHON_FIXTURE=prefix + python_fixture)
            result = subprocess.run(['bash', str(repo / 'scripts/prepare_coreai_decoder.sh')],
                                    cwd=base, env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(sum(a[0]=='uv' and a[1][0]=='venv' for a in calls), 0 if reuse else 1)
            self.assertEqual(sum(a[0]=='uv' and a[1][:2]==['pip','install'] for a in calls), 1)
            python_args = next(a[1] for a in calls if a[0]=='python')
            self.assertEqual('--verify-only' in python_args, reuse)

    def test_fresh_managed_python_and_package_cache_stay_in_checkout(self):
        self.run_fixture(False)

    def test_existing_environment_and_bundle_are_reused_with_local_uv_paths(self):
        self.run_fixture(True)


if __name__ == '__main__':
    unittest.main()
