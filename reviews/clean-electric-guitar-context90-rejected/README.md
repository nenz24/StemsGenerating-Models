# Rejected guitar 90-second contexts

Listener feedback: **"jadi noise dan strummingnya terdengar aneh"**.
This complete 243.330625-second result is rejected as a musical candidate.
It does not replace the accepted opening-30s guitar reference.

The uploaded feedback ZIP is preserved with archive/member hashes in
`review.json`. All three actual input WAV hashes were reproduced from the
original piano. Reports match the original guitar preset, seed 0, 50 steps,
CFG 7 and CUDA FP32. Decoded full-song MP3 has the expected number of frames,
finite samples and no samples reaching absolute amplitude 1. These checks do
not establish clean raw generation or musical quality.

The second seam has a reported level change of about 9.57 dB. This may need
work but does not establish the cause of the reported noise/strumming. The
original caption explicitly asks for sparse rhythmic strums. No unique cause
is inferred from metadata, and no automated noise/articulation verdict is made.

Notebook 19 is retained as opt-in reproduction, defaulting to plan-only.
Notebook 20 compares two matched 30-second runs at source start 77.130625s:
original caption versus a picked-note candidate. It tests shorter contexts
outside the accepted intro, then articulation at the same duration/input.
It generates two clips, not another full song. The guitar full/chunk lossless
WAVs were not uploaded; reported raw WAV hashes remain Colab evidence only.
