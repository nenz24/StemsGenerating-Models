# Clean electric guitar: accepted opening probe

The listener described `guitar_feedback.zip` as **"hasil kitar lumayan oke"**
and requested the next stage using the existing 90-second chunk approach.
This preserves the opening **30 seconds**, not a full-song guitar verdict.

The actual report matches clean-electric-guitar-v1: ACE-Step Base / LEGO /
CUDA FP32 / seed 0 / CFG 7 / 50 steps, with guitar instruction and the clean
electric caption. The piano input WAV SHA-256 was reproduced exactly from the
original piano upload. No bass/drum tracks or guitar reference were model inputs.

`baseline.json` records the source/preset, reported WAV hashes and uploaded
archive/member hashes. The ZIP contains MP3 previews and reports; keep
`guitar_result.zip` from Colab for lossless WAVs. Reported raw guitar/mix WAV
hashes cannot be verified here without those WAVs.

Notebook 19 continues with the same preset in contexts capped at 90 seconds,
6-second margins and 0.58-second equal-power crossfades. The longer guitar
contexts, chord changes, playing feel, timbre and seams still
need listening. Existing bass/drum baselines remain available for a subsequent
combined mix, without regenerating them.
