# Acoustic drum probe

Bass context 90 is frozen, with its noisy chunks documented in
`baselines/electric-bass-context90/`. Do not overwrite those archives or regenerate
bass while starting a new instrument.

Use notebook 13 on the active Colab VM with the existing ACE-Step runtime/weights.
Its embedded helper discovers the source folder created by notebook 12, including
when notebook globals are missing. If needed, copy its cells into the active
notebook. Existing uploaded piano is reused; otherwise upload the original piano.
No installer/download is called by the drum probe. An empty VM needs notebook 12's
source/runtime/weights cells first; stop before its bass generation cells.

One 30-second Base/LEGO drum generation uses CUDA FP32, seed 0, CFG 7, 50 steps,
offload, and a restrained acoustic-kit caption saved in `configs/acoustic-drums.json`.
It uses the original piano directly with gain .7, no invented BPM/meter, and no
bass mix as the condition. The bass 90-second cap is not assumed suitable for drums.

Listen for kick/snare/hi-hat, groove/timing, fills and noise. Upload
`drum_feedback.zip` for review. Save `drum_result.zip` for the WAVs. Parameter,
input, seed, drum instruction, FP32 capture and output-length checks must pass
before reporting success. Failure ZIPs contain reports, not fabricated audio.

CLI on a complete checkout:

```bash
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drums_plan
python scripts/drum_probe.py --source /path/to/piano.mp3 --out runs/drums_real --execute
```

`--start` selects a different 30-second excerpt. Every output directory must be
new; the previous bass/drum runs are preserved. CPU tests validate plan, targeting,
export and failure handling with mocked inference. Real drum quality/GPU success
has not been tested in the chat environment.
