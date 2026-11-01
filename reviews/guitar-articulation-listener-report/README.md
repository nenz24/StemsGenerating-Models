# Guitar articulation: listener rejects both variants

Latest listener report: **"ternyata original guitar juga tidak masuk denga akor
dan nada piano, serta picked jadi glitch semua"**.

The original variant is rejected for harmonic/pitch agreement; the picked
variant is rejected for glitches. This is a listener report only. The notebook-20
feedback/result ZIP has not been uploaded, so the exact run, input position,
settings, audio and cause of glitches cannot yet be independently checked.
No harmonic score, glitch localization or causal diagnosis is claimed.

The earlier favorable opening-30s impression remains historical tone feedback,
not evidence of correct chord following. No guitar arrangement is currently
approved for harmony. Existing audio and the bass/drum baselines are preserved.

## What the current runner controls

`tools/stem_eval/backends.py:AceBackend.generate` passes the source audio,
LEGO guitar instruction, caption and generation settings. The shared worker
overrides the caption. It does not pass a time-aligned chord/note sequence.
`tools/stem_eval/run.py:plan` explicitly labels conditioning as audio-only.

Audio conditioning can convey harmony, but this wrapper provides no hard
guarantee that generated guitar notes agree with it. A global key/BPM setting
would not by itself specify each chord or permitted note at each time. This
does not establish that ACE-Step can never accompany piano correctly, nor
identify the cause of the picked glitches.

## Next useful work

1. Obtain `guitar_articulation_feedback.zip` from the failed paired run and
   verify its actual source, settings and timing. Lossless result WAVs are
   useful if MP3 evidence leaves the glitch analysis ambiguous.
2. Compare piano and guitar over the exact same short interval: estimate
   local pitch-class content/chord candidates, identify disagreements and
   listen at those times. Chroma overlap alone is not a calibrated correctness
   score: guitar voicings, melody notes and passing tones need musical context.
3. Validate the source harmony/timing before using it as a generation guide.
   Existing automatic piano estimates are candidates, not verified transcription.
4. Evaluate a generation path that actually consumes time-aligned harmonic
   guidance and retains natural guitar articulation. Support in a selected
   model/adapter must be verified; no new model or working implementation is
   claimed here. Earlier MIDI/SoundFont bass did not meet the user's playing-feel
   requirement, so it is not adopted as a guitar-quality solution.

Caption/seed/duration sweeps and another full-song render are not the next
recommended action. Notebooks 18–20 are retained for explicit reproduction and
default to plan-only. The guitar source/target conditioning and the picked
signal defects need separate evaluation before another quality claim.
