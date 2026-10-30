"""Numerical audio checks and runner contract tests; no model/GPU simulation claims."""
import array
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "stem_eval"))
from audio import RATE, add_clicks, listening_mix, read_pcm, validate_times, write_pcm
from backends import AceBackend, StageBackend
from prepare import prepare
from provenance import COMMITS
from run import plan


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.wav"
        self.samples = array.array("h", (int(2000 * math.sin(i * 0.02)) for i in range(2 * RATE)))
        write_pcm(self.source, self.samples)
        self.conditioning = self.root / "input.json"
        self.metadata = {"schemaVersion": 1, "request": {"duration": 1, "start": 45},
                         "beats": [0.031, 0.487, 0.978], "downbeats": [0.031],
                         "chords": [{"start": 0.123, "duration": 0.178, "symbol": "G/B"}]}
        self.conditioning.write_text(json.dumps(self.metadata))
        self.prepared = self.root / "prepared"

    def test_preparation_keeps_clip_times_length_and_source_gain(self):
        manifest = prepare(self.source, self.conditioning, self.prepared)
        output = read_pcm(self.prepared / "source.wav")
        self.assertEqual(len(output), len(self.samples))
        self.assertLessEqual(max(abs(out - round(original * 0.7)) for out, original in zip(output, self.samples)), 1)
        self.assertEqual(json.loads((self.prepared / "conditioning.json").read_text()), self.metadata)
        self.assertEqual(set(manifest["conditions"]), {"A", "B", "C"})

    def test_click_changes_only_actual_beat_windows_and_accents_downbeat(self):
        write_pcm(self.root / "silence.wav", [0] * (2 * RATE))
        add_clicks(self.root / "silence.wav", self.root / "click.wav", [0.031, 0.487], [0.031], True)
        output = read_pcm(self.root / "click.wav")[::2]
        for index, value in enumerate(output):
            inside = any(round(t * RATE) <= index < round(t * RATE) + 320 for t in [0.031, 0.487])
            if not inside:
                self.assertEqual(value, 0)
        first = max(abs(x) for x in output[992:1312])
        second = max(abs(x) for x in output[15584:15904])
        self.assertGreater(first, second)
        self.assertGreater(second, 0)

    def test_no_beat_analysis_does_not_invent_click_conditions(self):
        self.metadata.update(beats=[], downbeats=[])
        self.conditioning.write_text(json.dumps(self.metadata))
        manifest = prepare(self.source, self.conditioning, self.prepared)
        self.assertEqual(manifest["conditions"], {"A": "source.wav"})

    def test_no_overwrite_and_failed_preparation_removes_partial_files(self):
        self.metadata["request"]["duration"] = 2
        self.conditioning.write_text(json.dumps(self.metadata))
        with self.assertRaises(ValueError):
            prepare(self.source, self.conditioning, self.prepared)
        self.assertFalse(self.prepared.exists())
        self.metadata["request"]["duration"] = 1
        self.conditioning.write_text(json.dumps(self.metadata))
        prepare(self.source, self.conditioning, self.prepared)
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.conditioning, self.prepared)
        self.assertTrue((self.prepared / "source.wav").exists())

    def test_clean_mix_rejects_duration_mismatch(self):
        write_pcm(self.root / "short.wav", [0] * 10)
        with self.assertRaises(ValueError):
            listening_mix(self.source, self.root / "short.wav", self.root / "mix.wav")
        write_pcm(self.root / "zero.wav", [0] * len(self.samples))
        listening_mix(self.source, self.root / "zero.wav", self.root / "mix.wav")
        self.assertEqual(read_pcm(self.root / "mix.wav"), array.array("h", (round(x * 0.5) for x in self.samples)))

    def test_plans_have_same_seeds_and_only_stage_receives_clicks(self):
        prepare(self.source, self.conditioning, self.prepared)
        ace = plan(self.prepared, "ace", "bass", [0, 1, 2])
        stage = plan(self.prepared, "stage", "bass", [0, 1, 2])
        self.assertEqual(len(ace["jobs"]), 3)
        self.assertEqual(len(stage["jobs"]), 9)
        self.assertEqual(ace["status"], "planned")
        self.assertEqual({job["seed"] for job in stage["jobs"]}, {0, 1, 2})
        with (self.prepared / "source.wav").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            plan(self.prepared, "ace", "bass", [0])

    def test_actual_cli_plans_without_torch_or_weights(self):
        prepare(self.source, self.conditioning, self.prepared)
        script = Path(__file__).resolve().parents[2] / "tools/stem_eval/run.py"
        subprocess.run([sys.executable, str(script), "--prepared", str(self.prepared),
                        "--backend", "ace", "--out", str(self.root / "plan")], check=True)
        result = json.loads((self.root / "plan/experiment.json").read_text())
        self.assertEqual(result["status"], "planned")
        self.assertFalse(list((self.root / "plan").glob("**/stem.wav")))



class ContractTests(unittest.TestCase):
    def test_invalid_beats_are_rejected(self):
        for values in ([float("nan")], [0.5, 0.2], [-0.1], [1], [0, 0], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_times(values, 1)

    def test_ace_api_selects_base_lego_and_disables_every_planner_flag(self):
        handler = MagicMock()
        handler.initialize_service.return_value = ("ready", True)
        params, config = MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.wav"
            output.touch()
            generate = MagicMock(return_value=types.SimpleNamespace(success=True, audios=[{"path": str(output)}]))
            modules = {
                "acestep": types.ModuleType("acestep"),
                "acestep.handler": types.SimpleNamespace(AceStepHandler=MagicMock(return_value=handler)),
                "acestep.inference": types.SimpleNamespace(GenerationParams=params, GenerationConfig=config, generate_music=generate),
            }
            with patch.dict(sys.modules, modules):
                adapter = AceBackend(Path(directory))
                adapter.generate(output, 10, "bass", 42, Path(directory))
            self.assertEqual(handler.initialize_service.call_args.kwargs["config_path"], "acestep-v15-base")
            options = params.call_args.kwargs
            self.assertEqual(options["task_type"], "lego")
            self.assertEqual(options["inference_steps"], 50)
            for key in ("thinking", "use_cot_caption", "use_cot_metas", "use_cot_language"):
                self.assertFalse(options[key])
            self.assertIsNone(generate.call_args.args[1])

    def test_stage_api_uses_context_and_not_prompt_or_beat_tensor(self):
        model, loader, load, save = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        loader.return_value = model
        modules = {"stage": types.ModuleType("stage"), "stage.utils": types.ModuleType("stage.utils"),
                   "stage.loader": types.SimpleNamespace(load_model=loader),
                   "stage.utils.audio": types.SimpleNamespace(load_audio=load, save_audio=save),
                   "lightning": MagicMock()}
        with patch.dict(sys.modules, modules):
            adapter = StageBackend(Path("/upstream"), Path("/checkpoint"))
            adapter.generate(Path("/source.wav"), 10, "bass", 42, Path("/out"))
        options = model.generate.call_args.kwargs
        self.assertEqual(options["gen_seconds"], 10)
        self.assertIsNone(options["prompt"])
        self.assertIsNone(options["beat"])
        self.assertIsNotNone(options["context"])
        load.assert_called_once_with(Path("/source.wav"))


if __name__ == "__main__":
    unittest.main()
