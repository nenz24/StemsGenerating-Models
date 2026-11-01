# Approved drum-kit reference comparison

Listener: "oke sudah cukup baik" and requested the whole song in this style.
The uploaded `drum_kit_feedback.zip` preserves the control, reference and candidate
previews plus actual reports. The approved result is **chunk 0**, generated with
the fixed reference excerpt from chunk 1. Approval is local, not a full-song verdict.

`baseline.json` records reported WAV/reference hashes, archive hash, preset and
duration. The raw WAVs were not uploaded here; recover them from the existing
Colab comparison or `drum_kit_result.zip`. Do not reconstruct them from MP3.

Notebook 17 freezes those reference bytes, reuses approved chunk 0 unchanged,
generates chunks 1/2 with the same reference, and stitches a separate result.
Original presets/baselines remain intact. A common reference can still change
groove/timbre between contexts; listen before approving the complete song.
