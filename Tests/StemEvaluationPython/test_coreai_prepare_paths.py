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
    def run_fixture(self, reuse, frames=5, component='decoder', dit=False):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp);repo = base / 'repo with spaces';(repo / 'scripts').mkdir(parents=True)
            shutil.copyfile(ROOT / 'scripts/prepare_coreai_decoder.sh', repo / 'scripts/prepare_coreai_decoder.sh')
            if dit:shutil.copyfile(ROOT / 'scripts/prepare_coreai_dit_block.sh', repo / 'scripts/prepare_coreai_dit_block.sh')
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
            candidates = repo / 'prototypes/ios27-coreai/ModelCandidates'
            baseline = candidates / 'decoder-pretrained/baseline-marker'
            if frames == 25:
                baseline.parent.mkdir(parents=True)
                baseline.write_text('preserve 0.2 s baseline')
            if component == 'encoder':
                decoder_one_second = candidates / 'decoder-pretrained-1s/baseline-marker'
                decoder_one_second.parent.mkdir(parents=True)
                decoder_one_second.write_text('preserve 1 s decoder')
            bundle_name = 'encoder-pretrained-1s' if component == 'encoder' else ('decoder-pretrained-1s' if frames == 25 else 'decoder-pretrained')
            if dit:
                encoder_marker = candidates / 'encoder-pretrained-1s/baseline-marker'
                encoder_marker.parent.mkdir(parents=True)
                encoder_marker.write_text('preserve encoder')
                bundle_name = 'dit-block-pretrained'
            if reuse:
                target = repo / '.runtime-cache/coreai-decoder-venv/bin/python';target.parent.mkdir(parents=True)
                target.write_text(prefix + python_fixture);target.chmod(0o755)
                (candidates / bundle_name).mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       UV_PYTHON_INSTALL_DIR='/unwritable/managed-python', UV_CACHE_DIR='/unwritable/cache',
                       UV_PYTHON_BIN_DIR='/unwritable/bin', PREP_FIXTURE_ROOT=str(repo),
                       PREP_ORIGINAL_HOME=os.environ['HOME'], PREP_FIXTURE_LOG=str(log),
                       PREP_PYTHON_FIXTURE=prefix + python_fixture)
            command = ['bash', str(repo / 'scripts/prepare_coreai_decoder.sh')]
            if component == 'encoder':command += ['--component', 'encoder']
            if frames == 25:command += ['--frames', '25']
            if dit:command = ['bash', str(repo / 'scripts/prepare_coreai_dit_block.sh')]
            result = subprocess.run(command,
                                    cwd=base, env=env, text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(sum(a[0]=='uv' and a[1][0]=='venv' for a in calls), 0 if reuse else 1)
            self.assertEqual(sum(a[0]=='uv' and a[1][:2]==['pip','install'] for a in calls), 2 if dit else 1)
            python_args = next(a[1] for a in calls if a[0]=='python')
            self.assertEqual(sum(a[0]=='python' for a in calls), 1)
            self.assertEqual('--verify-only' in python_args, reuse)
            if not dit:self.assertEqual(python_args[python_args.index('--frames') + 1], str(frames))
            self.assertEqual(Path(python_args[python_args.index('--out') + 1]),
                             candidates / bundle_name)
            if not dit:self.assertEqual(python_args[python_args.index('--component') + 1], component)
            if frames == 25:
                self.assertEqual(baseline.read_text(), 'preserve 0.2 s baseline')
                self.assertIn('Uji blok generator' if dit else f'Uji {component} 1 detik', result.stdout)
            if component == 'encoder':
                self.assertEqual(decoder_one_second.read_text(), 'preserve 1 s decoder')
            if dit:
                self.assertEqual(encoder_marker.read_text(), 'preserve encoder')
                self.assertIn('scripts/export_coreai_dit_block.py', python_args)

    def test_fresh_managed_python_and_package_cache_stay_in_checkout(self):
        self.run_fixture(False)

    def test_existing_environment_and_bundle_are_reused_with_local_uv_paths(self):
        self.run_fixture(True)

    def test_one_second_export_and_reuse_select_separate_bundle_preserving_baseline(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):self.run_fixture(reuse, frames=25)

    def test_invalid_profiles_stop_before_setup(self):
        for args in (['--frames'], ['--frames', '90'], ['--other', '25'], ['--frames', '25', 'extra'],
                     ['--component'], ['--component', 'other'], ['--component', 'encoder', '--frames', '5']):
            with self.subTest(args=args):
                result = subprocess.run(['bash', str(ROOT / 'scripts/prepare_coreai_decoder.sh'), *args],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Usage:', result.stderr)

    def test_encoder_export_and_reuse_preserve_both_decoder_bundles(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):self.run_fixture(reuse, frames=25, component='encoder')

    def test_dit_preparation_only_prepares_environment_and_preserves_all_codec_bundles(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):self.run_fixture(reuse, frames=25, component='encoder', dit=True)


if __name__ == '__main__':
    unittest.main()
