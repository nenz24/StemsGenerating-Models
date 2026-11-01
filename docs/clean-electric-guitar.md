# Clean electric guitar: accepted probe and 90-second contexts

## Accepted short probe and full-song continuation

The listener found the uploaded 30-second result reasonably good and requested
the existing bass/drum 90-second chunk method. The feedback and metadata are
frozen in [`clean-electric-guitar-30-v1`](../baselines/clean-electric-guitar-30-v1/).
Its actual guitar settings match the preset, and the piano model-input hash was
reproduced exactly. This acceptance covers the excerpt only.

Use [`19_colab_guitar_context_90.ipynb`](../notebooks/19_colab_guitar_context_90.ipynb)
on the active CUDA VM. Run all: it makes guitar contexts capped at 90 seconds
**including** 6-second margins and integer padding, then stitches with a
0.58-second float equal-power fade. The 243.330625-second piano produces three
model contexts (90/89/90 seconds), keeps the exact source length, and exports
guitar solo plus piano/guitar mix. The same accepted caption/preset is retained.
The 30-second output cannot substitute for a longer context with different
input boundaries, so the three new guitar chunks are generated separately.

Resume reuses verified completed chunks. Notebook helpers and a guitar-capable
engine are embedded for old cached bridges, without reinstalling weights or
rewriting runtime source. A complete cache is required; a new empty VM still
needs setup. Bass/drum audio is not generated again or used as guitar input.

Keep `guitar_chunked_result.zip` (full lossless exports and raw chunks); upload
`guitar_chunked_feedback.zip` (full MP3s, previews, seam excerpts and actual
reports). This notebook evaluates guitar with piano. The later combined mix can
reuse the existing bass lossless export and approved shared-kit drum export;
keep those result ZIPs. The bass baseline retains its known noise caveat.

Guitar timbre, voicings and phrasing can change between contexts. The 90-second
choice is inherited from bass/drum experiments, not validated for guitar yet.
Monophonic bass pitch continuity is skipped for polyphonic guitar; numerical
seam diagnostics cannot establish chord agreement or musical quality.

## Original 30-second workflow

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
sparse rhythmic strums. The opening result was accepted for this longer trial.

The pinned upstream commit `ca1e85fe9430179831e6bc6be790c332190a3866`
lists `guitar` in `acestep/constants.py:TRACK_NAMES` and LEGO uses
`Generate the {TRACK_NAME} track based on the audio context:`. The instruction
uses `guitar`; the clean electric character is requested in the caption, not a
separate checkpoint. The listener has assessed the 30-second Colab output;
longer contexts need their own GPU/listening trial.

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

Use `--start` and `--duration` for 10–90-second contexts in separate output
folders. Source duration is measured from the audio; no guessed BPM or meter
is supplied. GPU model inference and musical quality are untested on the chat machine. Offline
iPhone/iOS 27/CoreAI remains a separate implementation and validation task.
