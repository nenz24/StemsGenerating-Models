"""Caption-only CUDA experiment contracts; no actual model inference is performed."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools/stem_eval")]
from compare_bass_tone import ELECTRIC_BASS_CAPTION, caption_worker, compare
from compare_guidance import prepare
from cuda_precision import FLOAT32_WORKER
from provenance import COMMITS, sha256
import test_guidance_comparison


class BassToneTests(unittest.TestCase):
    def baseline(self, root):
        run, job, _ = test_guidance_comparison.GuidanceComparisonTests().make_comparison(root)
        reference = root / "reference"
        snapshot = prepare(run, reference)
        baseline = root / "cuda_baseline"
        folder = baseline / "generation/000_A_bass_seed0"
        folder.mkdir(parents=True)
        shutil.copyfile(reference / "baseline_cfg7.wav", folder / "stem.wav")
        settings = snapshot["baseline_settings"]
        settings["params"]["src_audio"] = str(reference / "prepared/source.wav")
        job.update(source=str(reference / "prepared/source.wav"), guidance_scale=7)
        result = dict(job, status="generated", device="cuda", settings=settings, execution_signature=None,
                      stem_file="stem.wav", stem_sha256=sha256(folder / "stem.wav"))
        (folder / "result.json").write_text(json.dumps(result))
        (baseline / "generation/experiment.json").write_text(json.dumps({"status": "generated", "jobs": [job], "offload_to_cpu": True}))
        (baseline / "generation/session.json").write_text(json.dumps({"assets_manifest_sha256": sha256(ROOT / "configs/assets-base.json")}))
        capture = {"status": "captured", "model_device": "cuda", "model_dtype": "torch.float32",
                   "calls": [{"backend": "pytorch_vae", "dtype": "torch.float32", "status": "returned"}]}
        report = {"status": "generated", "upstream_commit": COMMITS["ace"], "decoder_capture": capture,
                  "source_sha256": job["source_sha256"], "settings": settings, "candidate_sha256": result["stem_sha256"],
                  "requested": {"device": "cuda", "model": "acestep-v15-base", "seed": 0, "guidance_scale": 7, "steps": 50, "duration": 30}}
        (baseline / "cuda_reference.json").write_text(json.dumps(report))
        return reference, baseline, result, capture

    def test_plan_preserves_baseline_and_does_not_load_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, baseline, _, _ = self.baseline(root)
            before = sha256(baseline / "cuda_reference.json")
            with patch("cuda_reference.check_cuda", side_effect=AssertionError("must not inspect GPU in plan")):
                report = compare(ROOT, reference, baseline, root / "plan")
            self.assertEqual(report["status"], "planned")
            self.assertEqual(report["requested_caption"], ELECTRIC_BASS_CAPTION)
            self.assertEqual(sha256(baseline / "cuda_reference.json"), before)
            self.assertFalse((root / "plan/electric_listen.wav").exists())
            self.assertFalse((root / "plan/bass_tone_result.zip").exists())

    def test_candidate_verifies_caption_fp32_and_all_other_settings(self):
        for changed in (None, "guidance", "precision", "failure"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reference, baseline, original, capture = self.baseline(root)
                original_hash = sha256(baseline / "generation/000_A_bass_seed0/stem.wav")
                output = root / "tone"
                def generate(command, **kwargs):
                    self.assertEqual(command[1], "-c")
                    self.assertIn(ELECTRIC_BASS_CAPTION, command[2])
                    if changed == "failure":
                        raise RuntimeError("fixture model failure")
                    folder = output / "generation/000_A_bass_seed0"
                    folder.mkdir(parents=True)
                    candidate = copy.deepcopy(original)
                    candidate["settings"]["params"]["caption"] = ELECTRIC_BASS_CAPTION
                    if changed == "guidance": candidate["settings"]["params"]["guidance_scale"] = 3
                    shutil.copyfile(baseline / "generation/000_A_bass_seed0/stem.wav", folder / "stem.wav")
                    (folder / "result.json").write_text(json.dumps(candidate))
                    job = {key: original[key] for key in ("condition", "instrument", "duration", "seed", "source_sha256", "guidance_scale")}
                    (output / "generation/experiment.json").write_text(json.dumps({"status": "generated", "jobs": [job]}))
                    (output / "generation/session.json").write_text("{}")
                    (output / "capture").mkdir()
                    candidate_capture = copy.deepcopy(capture)
                    if changed == "precision": candidate_capture["model_dtype"] = "torch.float16"
                    (output / "capture/capture.json").write_text(json.dumps(candidate_capture))
                with patch("cuda_reference.check_cuda", return_value={"fixture_only": True}), patch("stream_process.run_logged", side_effect=generate) as worker:
                    if changed:
                        with self.assertRaises((ValueError, RuntimeError)):
                            compare(ROOT, reference, baseline, output, execute=True, float32_worker=FLOAT32_WORKER)
                    else:
                        compare(ROOT, reference, baseline, output, execute=True, float32_worker=FLOAT32_WORKER)
                self.assertEqual(worker.call_count, 1)
                report = json.loads((output / "comparison.json").read_text())
                self.assertEqual(report["status"], "failed" if changed else "generated")
                self.assertEqual(sha256(baseline / "generation/000_A_bass_seed0/stem.wav"), original_hash)
                self.assertEqual((output / "bass_tone_result.zip").exists(), changed is None)
                if changed is None:
                    with zipfile.ZipFile(output / "bass_tone_result.zip") as archive:
                        self.assertIn("electric_mix.wav", archive.namelist())
                        self.assertFalse(any("checkpoints/" in name for name in archive.namelist()))

    def test_uploaded_worker_runs_in_fresh_python_and_serializes_only_caption_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/tools/stem_eval"; source.mkdir(parents=True)
            upstream = root / "source/vendor/ACE-Step-1.5"
            package = upstream / "acestep"; package.mkdir(parents=True)
            (package / "__init__.py").touch()
            (package / "handler.py").write_text("class AceStepHandler: pass\n")
            (upstream / "torch.py").write_text('float32="float32"\n')
            (package / "inference.py").write_text(
                "from dataclasses import dataclass,asdict\n@dataclass\nclass GenerationParams:\n"
                "    caption: str\n    seed: int\n    guidance_scale: float\n"
                "    def to_dict(self): return asdict(self)\n")
            shutil.copyfile(ROOT / "tools/stem_eval/audio.py", source / "audio.py")
            shutil.copyfile(ROOT / "tools/stem_eval/system_tools.py", source / "system_tools.py")
            script = source / "run.py"
            script.write_text(
                "import json,audio\nfrom acestep.inference import GenerationParams\n"
                "print(json.dumps(GenerationParams(caption='old',seed=0,guidance_scale=7).to_dict()))\n")
            elsewhere = root / "content"; elsewhere.mkdir()
            result = subprocess.run([sys.executable, "-c", caption_worker(FLOAT32_WORKER), str(script)],
                                    cwd=elsewhere, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"caption": ELECTRIC_BASS_CAPTION, "seed": 0, "guidance_scale": 7})

    def test_changed_input_is_rejected_before_creating_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, baseline, _, _ = self.baseline(root)
            with (reference / "prepared/source.wav").open("ab") as stream: stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "source differs"):
                compare(ROOT, reference, baseline, root / "bad")
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
