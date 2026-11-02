# Uploaded iPhone bass WAV audit

The user supplied `bass.wav` and `piano-with-bass.wav` after the
[complete CPU-request run and playback confirmation](../coreai-iphone17pro-bass-preview-cpu-complete/README.md).
The original audio remains in the uploaded attachments; audio/weights are not
added to Git or to app resources. SHA-256 values and measurements are recorded
in [metrics.json](metrics.json). Neither audio file was edited, normalized,
resampled, time-aligned or gain-fitted for the comparisons.

## Valid output format and levels

Both files contain **245760 stereo frames / 5.12 s / 48 kHz / Float32 PCM**.
All 491520 sample values per file are finite; none is at or above unit full
scale. Bass peak is 0.5564696789, RMS 0.1574208451 (-16.0588 dBFS).
Mix peak is 0.6434965134, RMS 0.1005269414 (-19.9544 dBFS). These are digital
level checks, not a score for feel, timbre, musical harmony or perceived noise.

The supplied mix closely matches `0.78*piano + 0.55*supplied_bass` using the
available Linux source PCM, with global gain 1.0, maximum residual
5.960464477539062e-7 and RMSE 3.898773511541241e-8. This supports correct sample
alignment and mix construction. It does not supply the Mac source binary.

## Comparison to available Linux references

The local Linux preview manifest and its source binding were checked; both
waveform binaries and the piano PCM were hash-verified against those manifests.
References are the **existing Linux export**, not the Mac per-run references
that the phone used. Across-host decode/preparation differences can contribute
to the discrepancy; it is not attributed solely to CoreAI lowering.

| Reference | Normalized RMSE | Max absolute difference | Samples above original pointwise bound | Strict waveform parity |
| --- | ---: | ---: | ---: | --- |
| Linux partition/storage16 | 9.12638247e-5 | 0.0001275837421 | 25 / 491520 | NOT PASSED |
| Linux full FP32 | 8.55458967e-5 | 0.0001294910908 | 4 / 491520 | NOT PASSED |

The Pearson waveform correlation exceeds 0.999999995 for both references;
FFT cross-correlation searching +/-100 ms has its maximum at zero lag.
**No realignment was applied to the numerical comparison.** These findings
support close preservation of waveform and timing relative to this reference,
not proof that the original reference has musically correct notes/groove.

The existing strict check is unchanged: every sample must satisfy
`abs(actual-ref) <= 1e-4 + 1e-3*abs(ref)` and normalized RMSE must be <= 1e-3.
Metrics use the existing `precision_metrics` from
`scripts/export_coreai_dit_block.py`, with reference shape `[1,2,245760]`.
Both relative RMS errors are small and under the RMS bound, but a few pointwise
violations still prevent strict PASS. No relaxed tolerance or overall numerical
baseline is claimed. Audio reading/metadata used SciPy/ffprobe; mix checks used
Float32 arithmetic and measurements used Float64 accumulation.

## Generation report and next action

The pasted text is again the **conditioning-only** report: 27/27 stages, six
checks, 7100.412209 ms and sampled app footprint maximum 117.17385864257812 MiB.
Those are not full generation latency or RAM. A subsequent
[per-output report](../coreai-iphone17pro-bass-preview-cpu-complete/generation-report.json)
now supplies the full measurements: 85.95 seconds, sampled process maximum
417.19 MiB, 8/10 output checks PASS and the final latent pair NOT PASSED. Both
Mac-reference waveform checks PASS; host status remains `bounded_diagnostic`.
This does not change the separately measured Linux-reference failures above.

Each successful generation already writes `report.txt` next to its two WAVs.
The app now provides **Lihat laporan generasi** and **Bagikan laporan generasi**
for that per-output file. On reopening/rebuilding it discovers the latest
complete saved output pair/report without running any model. Restored audio is
explicitly labeled as saved; the shared progress/last-error report remains
available independently. A component probe can no longer force the generation
report share button to send its unrelated report. Older files/cache/models are
not deleted. Native Apple build/validation of these new UI/restoration helpers
is pending; they use existing Foundation/SwiftUI APIs and change no model math.

The user subsequently confirmed **“Sudah cocok dan enak didengar”** for notes
and feel with piano. This is saved as the [5.12-second device audition baseline](../../baselines/iphone-coreai-bass-preview-512-v1/README.md),
separate from numerical parity. It does not replace the accepted 90-second CUDA
musical baseline or establish full-song feasibility.
