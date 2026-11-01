"""Long-context preparation/export and isolated worker contracts; no real GPU inference."""
import array
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
import electric_full_song as single
from audio import RATE, read_pcm, write_pcm
from provenance import sha256
from run import plan
from test_full_song import fixture


class SinglePassTests(unittest.TestCase):
    def test_three_minute_plan_has_one_context_and_no_song_chunks(self):
        self.assertEqual(single.PRESET, json.loads((ROOT / "configs/electric-bass.json").read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "piano.wav"; fixture(source, 180)
            original_hash = sha256(source)
            output = root / "one_pass"
            with patch("stream_process.run_logged", side_effect=AssertionError("plan must not launch runtime")):
                report = single.generate(ROOT, source, output)
            self.assertEqual(report["status"], "planned")
            self.assertEqual(report["input"]["duration"], 180)
            self.assertEqual(report["input"]["model_duration"], 180)
            self.assertEqual(report["model_calls_requested"], 1)
            self.assertEqual(len(plan(output / "prepared", "ace", "bass", [0])["jobs"]), 1)
            self.assertEqual(list(output.rglob("chunk_*.wav")), [])
            self.assertFalse((output / "generation").exists())
            self.assertEqual(sha256(source), original_hash)

    def test_excerpt_is_one_continuous_context_and_cli_accepts_600_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "piano.wav"; fixture(source, 70)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/electric_full_song.py"),
                "--source", str(source), "--out", str(root / "excerpt"), "--start", "5", "--duration", "60"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "excerpt/single_pass.json").read_text())
            self.assertEqual(report["input"]["duration"], 60)
            self.assertEqual(report["input"]["start"], 5)
            # The upper bound is accepted by the validator; rejection here is the
            # source's length, not the old 30s bridge cap.
            with self.assertRaisesRegex(ValueError, "Source is shorter"):
                single.prepare(source, root / "long", duration=600)
            with self.assertRaises(ValueError): single.prepare(source, root / "invalid", duration=601)

    def fractional_source(self, root):
        source = root / "piano.wav"; fixture(source, 60)
        pcm = read_pcm(source)
        pcm.extend(array.array("h", [321, -321]) * (RATE // 8))
        write_pcm(source, pcm)
        return source

    def fake_generation(self, output, changed=None):
        prepared = output / "prepared"
        generation = output / "generation"
        folder = generation / "000_A_bass_seed0"; folder.mkdir(parents=True)
        experiment = plan(prepared, "ace", "bass", [0])
        if changed == "oom":
            experiment.update(status="failed", error="OutOfMemoryError: CUDA out of memory (fixture)")
            (generation / "experiment.json").write_text(json.dumps(experiment))
            raise RuntimeError("CUDA out of memory (fixture)")
        job = experiment["jobs"][0]
        settings = {"params": {"caption": single.PRESET["caption"], "seed": 0, "guidance_scale": 7,
            "inference_steps": 50, "duration": job["duration"], "task_type": "lego", "instrumental": True,
            "enable_normalization": False, "thinking": False, "use_cot_caption": False,
            "use_cot_metas": False, "use_cot_language": False},
            "config": {"seeds": [0], "batch_size": 1, "use_random_seed": False}}
        if changed == "duration": settings["params"]["duration"] = 30
        shutil.copyfile(prepared / "source.wav", folder / "stem.wav")
        shutil.copyfile(folder / "stem.wav", folder / "mix.wav")
        result = dict(job, status="generated", device="cuda", execution_signature=None, settings=settings,
                      stem_file="stem.wav", stem_sha256=sha256(folder / "stem.wav"),
                      peak_cuda_allocated_bytes=123, peak_cuda_reserved_bytes=456, inference_seconds=0.01,
                      fixture_only=True)
        (folder / "result.json").write_text(json.dumps(result))
        experiment["status"] = "generated"
        (generation / "experiment.json").write_text(json.dumps(experiment))
        (generation / "session.json").write_text('{"fixture_only": true}')
        (output / "capture").mkdir()
        capture = {"status": "captured", "model_device": "cuda", "model_dtype": "torch.float32",
                   "calls": [{"backend": "pytorch_vae", "dtype": "torch.float32", "status": "returned"}]}
        if changed == "precision": capture["model_dtype"] = "torch.float16"
        (output / "capture/capture.json").write_text(json.dumps(capture))

    def test_fractional_duration_padding_is_removed_without_losing_any_source_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = self.fractional_source(root); output = root / "song"
            def run(command, **kwargs):
                self.assertIn("--worker", command)
                self.assertEqual(command[command.index("--root") + 1], str(ROOT))
                self.fake_generation(output)
            with patch.object(single, "model_python", return_value=Path(sys.executable)), patch("stream_process.run_logged", side_effect=run) as worker:
                report = single.generate(ROOT, source, output, execute=True)
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(report["input"]["duration"], 60.125)
            self.assertEqual(report["input"]["model_duration"], 61)
            self.assertEqual(report["input"]["end_padding_frames"], int(0.875 * RATE))
            self.assertEqual(single.frames(output / "electric_bass.wav"), int(60.125 * RATE))
            self.assertEqual(read_pcm(output / "electric_bass.wav"), read_pcm(output / "prepared/original.wav"))
            self.assertEqual(single.frames(output / "mix_full.wav"), single.frames(source))
            with zipfile.ZipFile(output / "full_song_feedback.zip") as archive:
                self.assertIn("electric_preview.mp3", archive.namelist())
                self.assertFalse(any(name.endswith(".wav") or "runtime/" in name for name in archive.namelist()))
            self.assertEqual(report["peak_cuda_allocated_bytes"], 123)

    def test_failed_memory_or_unexpected_settings_do_not_become_success_or_chunk_fallback(self):
        for changed in ("oom", "duration", "precision"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = self.fractional_source(root); output = root / "song"
                with patch.object(single, "model_python", return_value=Path(sys.executable)), patch("stream_process.run_logged", side_effect=lambda *a, **kw: self.fake_generation(output, changed)) as worker:
                    with self.assertRaises((ValueError, RuntimeError)):
                        single.generate(ROOT, source, output, execute=True)
                self.assertEqual(worker.call_count, 1)
                report = json.loads((output / "single_pass.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertFalse((output / "electric_bass.wav").exists())
                self.assertEqual(list(output.rglob("chunk_*.wav")), [])
                with zipfile.ZipFile(output / "full_song_feedback.zip") as archive:
                    self.assertEqual(json.loads(archive.read("single_pass.json"))["status"], "failed")

    def test_fresh_python_worker_applies_fp32_caption_and_long_export_without_legacy_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bridge = root / "source"
            tools = bridge / "tools/stem_eval"; scripts = bridge / "scripts"
            upstream = bridge / "vendor/ACE-Step-1.5"; package = upstream / "acestep"
            for path in (tools, scripts, package): path.mkdir(parents=True)
            for name in ("system_tools.py",): shutil.copyfile(ROOT / "tools/stem_eval" / name, tools / name)
            shutil.copyfile(ROOT / "scripts/runtime_environment.py", scripts / "runtime_environment.py")
            (tools / "provenance.py").write_text(
                "import json\ndef check_upstream(*a): return 'fixture pinned'\n"
                "def write_json(path,data): path.write_text(json.dumps(data))\n")
            (upstream / "torch.py").write_text("float32='float32'\nclass cuda:\n    is_available=staticmethod(lambda: True)\n")
            (package / "__init__.py").touch()
            (package / "handler.py").write_text(
                "class AceStepHandler:\n    def _load_main_model_from_checkpoint(self,**kwargs): pass\n")
            (tools / "backends.py").write_text(
                "from types import SimpleNamespace\nclass AceBackend:\n"
                "    def __init__(self):\n        from acestep.handler import AceStepHandler\n"
                "        self.handler=AceStepHandler()\n        self.handler._load_main_model_from_checkpoint(device='cuda')\n"
                "        self.params_type=lambda **kw: SimpleNamespace(**kw)\n")
            (tools / "run.py").write_text(
                "import json\ndef plan(prepared,*args):\n"
                "    return {'jobs':[{'instrument':args[1],'duration':json.loads((prepared/'prepared.json').read_text())['duration']}]}\n"
                "def execute(args,experiment):\n    import backends\n    model=backends.AceBackend()\n"
                "    params=model.params_type(caption='generic')\n"
                "    canonicalize(args.prepared/'source.wav',args.out/'export.wav',duration=61,gain=1,max_duration=args.max_job_seconds)\n"
                "    probe={'reference_audio':getattr(params,'reference_audio',None),'instrument':experiment['jobs'][0]['instrument'],'caption':params.caption,'dtype':model.handler.dtype,'vae_dtype':model.handler._get_vae_dtype('cuda'),'limit':args.max_job_seconds,'calls':1}\n"
                "    (args.out/'probe.json').write_text(json.dumps(probe))\n    experiment['status']='generated'\n")
            output = root / "output"; output.mkdir()
            source = self.fractional_source(root)
            single.prepare(source, output / "prepared")
            elsewhere = root / "content"; elsewhere.mkdir()
            result = subprocess.run([sys.executable, str(ROOT / "scripts/electric_full_song.py"),
                "--worker", "--root", str(bridge), "--out", str(output)], cwd=elsewhere,
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            probe = json.loads((output / "generation/probe.json").read_text())
            self.assertEqual(probe, {"reference_audio":None,"instrument":"bass", "caption": single.PRESET["caption"], "dtype": "float32", "vae_dtype": "float32", "limit": 600, "calls": 1})
            self.assertEqual(single.frames(output / "generation/export.wav"), 61 * RATE)
            import drum_probe
            drums=root/'drums';drums.mkdir()
            shutil.copytree(output/'prepared',drums/'prepared')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/drum_probe.py'),'--worker',
                '--root',str(bridge),'--out',str(drums)],cwd=elsewhere,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            probe=json.loads((drums/'generation/probe.json').read_text())
            self.assertEqual(probe,{'reference_audio':None,'instrument':'drums','caption':drum_probe.PRESET['caption'],
                'dtype':'float32','vae_dtype':'float32','limit':600,'calls':1})
            ref_out=root/'drums_ref';ref_out.mkdir();shutil.copytree(output/'prepared',ref_out/'prepared')
            ref=ref_out/'reference.wav';shutil.copyfile(source,ref)
            (ref_out/'drum_probe.json').write_text(json.dumps({'reference_audio':{'path':str(ref),'sha256':sha256(ref)}}))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/drum_probe.py'),'--worker',
                '--root',str(bridge),'--out',str(ref_out)],cwd=elsewhere,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            probe=json.loads((ref_out/'generation/probe.json').read_text())
            self.assertEqual(probe['reference_audio'],str(ref))
            self.assertEqual(probe['instrument'],'drums')


if __name__ == "__main__": unittest.main()
