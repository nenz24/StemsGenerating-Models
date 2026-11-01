# Approved full-song drum baseline

The listener approved `drum_shared_kit_feedback.zip`: **"sudah bagus, simpan ini
sebagai basline"**. This freezes the complete 243.330625-second arrangement for
the original piano recording, using notebook 17 and the acoustic-drums-v1 preset.

Three contexts (at most 90 seconds including margins) use the same frozen kit
reference. Approved chunk 0 is reused unchanged; chunks 1/2 were generated with
that reference. Stitching uses 6-second margins and a 0.58-second equal-power
crossfade, with final global headroom gain 0.95. No separate kick replacement
or EQ was applied. Model: ACE-Step Base / LEGO / CUDA FP32 / seed 0 / CFG 7 /
50 steps. See `baseline.json` for parameters and archive/member SHA-256 hashes.

The ZIP preserves full-song drum/mix MP3s, before-stitch previews, seam WAV
excerpts and actual per-chunk reports. It does **not** contain the complete
lossless WAVs or reference WAV. Keep `drum_shared_kit_result.zip` from Colab for
those; MP3 decoding cannot reconstruct the approved WAV bytes. WAV hashes in
the metadata are reported by the Colab verification, not rechecked against raw
WAVs on this machine.

Original seam diagnostic flags are retained alongside the listener's approval.
This baseline covers this recording; it does not establish quality on other
songs or offline iPhone/CoreAI feasibility. Further drum tuning is optional.
The next instrument is clean electric guitar, starting with a 30-second probe
conditioned on the original piano, then longer contexts and the combined mix.
