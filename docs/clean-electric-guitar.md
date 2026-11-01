# Clean electric guitar: harmonic agreement has not passed

The listener reports: **original does not fit the piano chords/pitches; picked
is full of glitches**. Neither articulation variant qualifies. The paired-run
feedback ZIP has not been uploaded, so the latest run's exact settings, interval
and audio defects have not been independently checked here. See the
[current review](../reviews/guitar-articulation-listener-report/).

## What the evidence establishes

The opening 30-second clip previously sounded reasonably good to the listener.
It is retained as [historical tone evidence](../baselines/clean-electric-guitar-30-v1/),
not a harmony-approved guitar baseline. The latest report does not separately
identify that opening clip as the compared input.

The [full-song 90-second run](../reviews/clean-electric-guitar-context90-rejected/)
was rejected for noise and unnatural strumming. All three model-input hashes
were reproduced; actual reports matched v1 and CUDA FP32. The complete MP3 had
the expected frames and finite samples below amplitude 1. Those checks did not
establish clean generation, correct harmony or natural playing. A level change
around the second seam also did not isolate the noise's cause.

The current runner passes source audio, a LEGO guitar instruction and caption.
It does not pass a time-aligned chord/note sequence. Audio conditioning can
convey harmony, but this wrapper provides no hard pitch/chord constraint.
Caption changes, chunk length, global key or BPM alone do not establish a fix.
The picked glitches need their own signal/runtime evaluation; a wrong chord
and an audio glitch are different failures.

## Next useful work

Obtain `guitar_articulation_feedback.zip`. Verify the actual interval, source
hashes, settings and both audio previews before comparing their local harmonic
content and inspecting glitches. Lossless result WAVs may help if MP3 evidence
leaves a specific defect ambiguous. No additional render is needed to collect
these already-produced files.

The prior piano analysis matches the original piano hash. Its 52 automatic
chord candidates overlapping notebook 20's default interval are preserved in
[`source-harmony-candidate.json`](../reviews/guitar-articulation-listener-report/source-harmony-candidate.json).
These are unverified major/minor estimates with uncalibrated score margins;
not a verified transcription or a correctness oracle. The latest trial's actual
interval still needs its report. Local pitch-class overlap is diagnostic only:
voicings, melody and passing tones require musical context.

Validate source harmony/timing, then evaluate a generation path that actually
consumes time-aligned harmonic guidance while preserving natural guitar
articulation. No such new model/adapter has been selected or implemented yet.
The earlier sampled MIDI bass did not meet the user's playing-feel requirement;
it is not adopted as a guitar-quality solution. Another caption/seed/duration
sweep or full-song render is not the next recommended step.

## Reproduction history

Notebooks 18–20 now default to `DRY_RUN=True`. Existing outputs, presets and
bass/drum references are preserved. Explicit reproduction can enable inference
on an existing CUDA runtime with a complete weight cache; an empty VM still
requires notebook 12's source/runtime/weight setup cells.

| Notebook | Experiment | Current qualification |
| --- | --- | --- |
| [18](../notebooks/18_colab_clean_electric_guitar.ipynb) | Opening 30s, original clean-electric caption | Historical favorable tone impression; harmonic agreement not approved |
| [19](../notebooks/19_colab_guitar_context_90.ipynb) | Full song, contexts 90/89/90s with 6s margins and 0.58s crossfades | Rejected: noise and unnatural strumming |
| [20](../notebooks/20_colab_guitar_articulation.ipynb) | Two matched 30s original/picked variants, default start 77.130625s | Both rejected by listener; latest paired archive pending |

Original model settings: ACE-Step 1.5 Base / LEGO / CUDA FP32 / seed 0 / CFG 7 /
50 steps. V1 asks for chord picking plus sparse rhythmic strums; v2 asks for
individually plucked chord tones and arpeggios. Actual guitar instruction:
`Generate the guitar track based on the audio context:`. Pinned upstream commit
`ca1e85fe9430179831e6bc6be790c332190a3866` lists `guitar` in `TRACK_NAMES`.
Track support and passing software tests do not establish musical quality.
