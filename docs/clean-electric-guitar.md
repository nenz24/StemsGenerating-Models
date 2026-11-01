# Clean electric guitar: first probe

The full-song shared-kit drum output is now frozen as a listener-approved
baseline. The next selected instrument is **clean electric guitar** accompanying
the original piano. Start with 30 seconds; approve the guitar articulation,
chords and timing before trying 90-second contexts or a full-song arrangement.
The bass/drum presets and existing outputs are preserved.

Open [`18_colab_clean_electric_guitar.ipynb`](../notebooks/18_colab_clean_electric_guitar.ipynb)
on the active CUDA Colab VM and run all cells. It reuses the existing runtime
and complete weight cache. Upload the original piano only if `SOURCE_PATH` is
unavailable. For an empty VM, run only notebook 12's source/setup/runtime/weight
cells first. A new empty VM still needs the weights.

The notebook embeds checked helpers and the current evaluation engine. That
engine adds the `guitar` track on older source bridges **within the guitar
process**, without replacing their tracked source or requiring a fresh runtime.
It uses ACE-Step 1.5 Base / LEGO / CUDA FP32 / seed 0 / CFG 7 / 50 steps. The
caption requests warm, undistorted electric guitar, gentle chord picking and
sparse rhythmic strums. This is a first candidate, not an approved guitar sound.

The pinned upstream commit `ca1e85fe9430179831e6bc6be790c332190a3866`
lists `guitar` in `acestep/constants.py:TRACK_NAMES` and LEGO uses
`Generate the {TRACK_NAME} track based on the audio context:`. The instruction
uses `guitar`; the clean electric character is requested in the caption, not a
separate checkpoint. Whether it produces that sound needs a GPU/listening trial.

Listen to both `guitar_preview.mp3` and `mix_preview.mp3`. Keep
`guitar_result.zip` (lossless WAVs and reports); upload `guitar_feedback.zip`
(previews and actual settings) for review. Report solo guitar noise, tone and
playing feel, chord agreement, timing and whether it leaves room for the piano.
This first probe conditions on piano only, rather than an evolving generated
mix. The later combined piano/guitar/bass/drum mix needs its own assessment.

`DRY_RUN=True` prepares and checks one guitar job with no model loading or
downloads. Command-line equivalent from a prepared checkout:

```bash
python scripts/guitar_probe.py --source piano.mp3 --out runs/guitar_clean_probe --execute
```

Use `--start` and `--duration` for later 10–30-second probes in separate output
folders. Source duration is measured from the audio; no guessed BPM or meter
is supplied. GPU model inference and musical quality are untested on the chat machine. Offline
iPhone/iOS 27/CoreAI remains a separate implementation and validation task.
