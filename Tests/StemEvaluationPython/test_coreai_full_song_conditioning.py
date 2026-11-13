"""Fixed geometry, provenance and local-only conditioning preparation contracts."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_coreai_full_song_conditioning as conditioning


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.drums = {'instruction': 'Generate drums:', 'caption': 'Drums only.',
                      'text_prompt': 'Generate drums:\nDrums only.\n30 seconds<|endoftext|>\n',
                      'metadata': '30 seconds', 'audio_seconds': 30., 'lyrics': '[Instrumental]',
                      'lyric_prompt': '[Instrumental]<|endoftext|>', 'vocal_language': 'unknown',
                      'preset_sha256': 'd' * 64}
        self.originals = {role: {'instruction': 'Generate ' + role + ':',
                               'caption': role + ' only.', 'preset_sha256': letter * 64}
                          for role, letter in [('bass', 'b'), ('guitar', 'c')]}

    def test_frozen_drums_unchanged_and_new_presets_have_separate_identity(self):
        before = copy.deepcopy((self.drums, self.originals))
        result = conditioning.role_prompts(self.drums, self.originals)
        self.assertEqual(result['drums'], self.drums)
        self.assertEqual((self.drums, self.originals), before)
        for role in ('bass', 'guitar'):
            self.assertEqual(result[role]['source_preset_sha256'], self.originals[role]['preset_sha256'])
            self.assertNotEqual(result[role]['preset_sha256'], self.originals[role]['preset_sha256'])
            self.assertEqual(result[role]['lyric_prompt'], self.drums['lyric_prompt'])
            self.assertTrue(result[role]['text_prompt'].endswith('30 seconds<|endoftext|>\n'))
            self.assertIn('No percussion or vocals.', result[role]['caption'])
        self.assertNotEqual(result['bass']['preset_sha256'], result['guitar']['preset_sha256'])

    def test_ambiguous_substitution_and_wrong_duration_rejected(self):
        for changed in [dict(self.drums, text_prompt=self.drums['text_prompt'] * 2),
                        dict(self.drums, audio_seconds=243.), dict(self.drums, lyrics='')]:
            with self.assertRaises(ValueError):
                conditioning.role_prompts(changed, self.originals)

    def test_tokens_must_fit_without_padding_or_truncation(self):
        prompts = conditioning.role_prompts(self.drums, self.originals)

        class Tokens:
            def __init__(self, text_length): self.text_length = text_length
            def encode(self, value):
                class Encoded: pass
                encoded = Encoded()
                encoded.ids = list(range(15 if value.startswith('[Instrumental]') else self.text_length))
                return encoded

        self.assertEqual(len(conditioning.tokenize(prompts, Tokens(111))['bass']['text']), 111)
        for count in (106, 110, 112, 256):
            with self.assertRaisesRegex(ValueError, 'no padding/truncation'):
                conditioning.tokenize(prompts, Tokens(count))


class FilesystemTests(unittest.TestCase):
    def test_plan_without_site_packages_does_not_read_weights_or_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'absent'
            result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/prepare_coreai_full_song_conditioning.py'),
                                     '--capture', str(out / 'missing'), '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['neural_calls'], 0)
            self.assertFalse(plan['ready_for_native_execution'])
            self.assertFalse(out.exists())

    def test_missing_cached_source_cannot_fall_back_to_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'configs').mkdir()
            (root / 'configs/coreai-lego-preprocessing.json').write_text(json.dumps(
                {'files': [{'path': 'handler/' + n} for n in ('constants.py', 'prompt_utils.py', 'metadata_utils.py')]}))
            with patch.object(conditioning, 'ROOT', root), patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
                with self.assertRaisesRegex(FileNotFoundError, 'no download attempted'):
                    conditioning.formatted_prompts()
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['configs'])

    def test_local_integrity_sha256_and_git_blob_no_repair_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source'
            raw = b'pinned local source\n'
            path.write_bytes(raw)
            for digest in (hashlib.sha256(raw).hexdigest(), hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()):
                record = {'size': len(raw), 'digest': digest}
                self.assertEqual(conditioning.local_file(record, path), path)
                path.write_bytes(b'X' * len(raw))
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    conditioning.local_file(record, path)
                self.assertEqual(path.read_bytes(), b'X' * len(raw))
                path.write_bytes(raw)
            with self.assertRaises(FileNotFoundError):
                conditioning.local_file(record, path.parent / 'missing')

    def test_output_preserves_existing_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture = root / 'capture'
            capture.mkdir()
            sentinel = capture / 'report.json'
            sentinel.write_bytes(b'accepted')
            with self.assertRaises(FileExistsError): conditioning.check_output(capture, [capture])
            with self.assertRaises(ValueError): conditioning.check_output(capture / 'nested', [capture])
            conditioning.check_output(root / 'fresh', [capture])
            self.assertEqual(sentinel.read_bytes(), b'accepted')


if __name__ == '__main__': unittest.main()
