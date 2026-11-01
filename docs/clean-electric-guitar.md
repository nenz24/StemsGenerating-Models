# Clean electric guitar: selected style and simultaneous layers

The user has clarified the target: the original 90s chunks are useful, with
liked strumming in chunk 0 and picking in chunk 2, but sound changes across
chunks. Both techniques should play **simultaneously**. The reuploaded
`guitar_feedback.zip` selects the opening-30s clip as the preferred sound/style;
its bytes match the original archive exactly.

Use [notebook 21 and the layering guide](guitar-layers.md). Two new performances
follow the same piano timeline and share the same frozen reference. Preview30
makes two 30s clips; full90 makes two whole-song roles with 90s context caps and
resumes verified chunks. Both roles and their layered/piano mixes are exported.
The new picked role does not reuse failed picked-v2. Voice/technique/harmonic
quality still needs listening; no new layered result has been approved yet.

Directly shifting chunk 2 onto chunk 0 would combine different piano positions
and chords. The layered workflow keeps shared song positions instead. Existing
90s audio, the selected short reference, bass and drums are preserved.

## Evidence and scope

- [Selected opening-30s sound/style](../baselines/clean-electric-guitar-30-v1/):
  explicitly reselected by the user; no assertion that a style reference locks
  each generated chord/note.
- [Useful 90s candidate](../reviews/clean-electric-guitar-context90/): latest
  clarification supersedes the earlier categorical rejection. The previous
  noise/strumming report, hashes and seam diagnostics remain recorded as history.
- [Original/picked-v2 trial](../reviews/guitar-articulation-listener-report/):
  verified uploaded archive, source start 77.130625s, identical 30s piano input
  and actual parameter differences caption/file path only. The listener rejected
  this original arm's harmonic agreement and picked-v2 glitches. This scoped
  result does not disqualify the separate preferred examples.

The current adapter uses audio and text conditioning, now with optional frozen
style reference. It does not impose a time-aligned hard chord/note constraint.
Matching source bytes, duration and software tests establishes experiment
integrity, not musical correctness or guitar-playing realism.

Notebooks 18–20 remain for reproduction with plan-only defaults. Notebook 21
uses the active CUDA runtime/cache and embeds the approved MP3 style preview;
that preview is not the original lossless guitar WAV. See its guide for output
ZIP names and full-song/preview configuration.
