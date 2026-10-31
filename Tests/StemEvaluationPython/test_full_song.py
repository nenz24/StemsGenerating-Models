"""Full-song timeline, crossfade and real-runner recovery with a fake model, not inference."""
import array
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools/stem_eval"))
from audio import RATE, read_pcm, write_pcm
from provenance import sha256
from song_audio import chunk_starts, prepare_song, stitch_song
from run_song import song_plan, stem_paths
import run as runner


def fixture(path, seconds):
    # Every second has a distinct amplitude; a whole-second timing shift is detectable.
    signal = array.array("h")
    for second in range(seconds):
        signal.extend(array.array("h", [(second + 1)*10, -(second + 1)*10]) * RATE)
    write_pcm(path, signal)


class FullSongTests(unittest.TestCase):
    def test_180_seconds_uses_20_chunks_without_gaps(self):
        starts = chunk_starts(180*RATE)
        self.assertEqual(len(starts), 20)
        self.assertEqual(starts[-1], 170*RATE)
        self.assertTrue(all(a < b <= a + 10*RATE for a, b in zip(starts, starts[1:])))
        self.assertEqual(chunk_starts(RATE), [0])
        self.assertEqual(chunk_starts(10*RATE), [0])
        with self.assertRaises(ValueError):
            chunk_starts(0)

    def test_real_three_minute_audio_round_trip_keeps_every_sample_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.wav"
            fixture(source, 180)
            prepared = root / "prepared"
            manifest = prepare_song(source, prepared)
            self.assertEqual(manifest["duration"], 180)
            self.assertEqual(len(manifest["chunks"]), 20)
            output = root / "stitched.wav"
            # Identity stems exercise timeline/crossfade numerically, not model generation.
            stitch_song(manifest, [prepared/c["file"] for c in manifest["chunks"]], output)
            self.assertTrue(read_pcm(output) == read_pcm(prepared / "source.wav"))
            self.assertEqual(sha256(source), manifest["original_sha256"])
            self.assertEqual(prepare_song(source, prepared), manifest)

    def test_short_tail_padding_is_removed_and_wrong_stem_duration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "short.wav", 2)
            song = prepare_song(root / "short.wav", root / "prepared")
            self.assertEqual(song["chunks"][0]["valid_frames"], 2*RATE)
            stitch_song(song, [root / "prepared/chunk_000.wav"], root / "out.wav")
            self.assertEqual(len(read_pcm(root / "out.wav")), 4*RATE)
            with self.assertRaises(ValueError):
                stitch_song(song, [root / "short.wav"], root / "invalid.wav")

    def test_changed_prepared_audio_and_changed_resume_configuration_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "source.wav", 2)
            profile = json.loads((ROOT / "configs/mac-full-song.json").read_text())
            song_plan(root / "source.wav", root / "run", profile, ROOT / "configs/assets-base.json")
            changed = dict(profile, instrument="drums")
            with self.assertRaises(ValueError):
                song_plan(root / "source.wav", root / "run", changed, ROOT / "configs/assets-base.json")
            prepared = root / "run/prepared"
            with (prepared / "chunk_000.wav").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                prepare_song(root / "source.wav", prepared)

    def test_actual_runner_recovers_second_chunk_without_regenerating_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "source.wav", 12)
            profile = json.loads((ROOT / "configs/mac-full-song.json").read_text())
            output = root / "run"
            experiment, song = song_plan(root / "source.wav", output, profile, ROOT / "configs/assets-base.json")
            settings = types.SimpleNamespace(upstream=root / "upstream", backend="ace", checkpoint=None,
                ace_model="acestep-v15-base", assets_manifest=ROOT / "configs/assets-base.json", device="mps",
                offload=True, out=output, prepared=output / "prepared", resume=True,
                execution_signature=experiment["execution_signature"])
            class Model:
                calls = 0
                fail = True
                def generate(self, source, duration, instrument, seed, target):
                    self.calls += 1
                    if self.calls == 2 and self.fail:
                        raise RuntimeError("fixture interrupted second chunk")
                    path = target / "raw.wav"
                    shutil.copyfile(source, path)
                    return path, {"fixture_only": True}
            model = Model()
            factory = MagicMock(return_value=model)
            torch = types.ModuleType("torch")
            torch.inference_mode = nullcontext
            with patch.dict(os.environ), patch.dict(sys.modules, {"torch": torch}), patch("backends.AceBackend", factory), patch("backends.weight_inventory", return_value=[]), patch("model_assets.process"), patch("model_assets.materialize", return_value=root/"runtime/checkpoints"), patch.object(runner, "check_upstream"), patch.object(runner, "select_device", return_value="mps"), patch.object(runner, "synchronize"), patch.object(runner, "reset_memory_stats"), patch.object(runner, "memory_stats", return_value={}):
                with self.assertRaises(RuntimeError):
                    runner.execute(settings, experiment)
                model.fail = False
                runner.execute(settings, experiment)
                self.assertEqual(model.calls, 3)  # success, failure, retry; first chunk generated once
                self.assertEqual(factory.call_count, 2)
                runner.execute(settings, experiment)
                self.assertEqual(factory.call_count, 2)  # all cached: no model load
            stems = stem_paths(output, experiment)
            self.assertEqual(len(stems), 2)
            stitch_song(song, stems, root / "out.wav")
            self.assertTrue(read_pcm(root/"out.wav") == read_pcm(output/"prepared/source.wav"))
            stems[0].write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                stem_paths(output, experiment)

    def test_notebook_is_unexecuted_and_uses_separate_input_folder(self):
        book = json.loads((ROOT / "notebooks/03_mac_full_song.ipynb").read_text())
        source = "".join("".join(c["source"]) for c in book["cells"])
        self.assertIn('data/full_song', source)
        self.assertIn('DRY_RUN = False', source)
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), "full-song-cell", "exec")


if __name__ == "__main__":
    unittest.main()
