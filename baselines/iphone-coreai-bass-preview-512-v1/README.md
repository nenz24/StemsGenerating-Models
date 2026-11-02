# iPhone CoreAI bass preview: 5.12 seconds

**User accepted:** “Sudah cocok dan enak didengar.” Approval covers bass notes
and feel alongside piano for the two supplied 5.12-second WAVs. This is the
short **device audition baseline**, not the accepted 90-second CUDA baseline,
a full-song baseline, or a strict numerical PASS.

The [baseline record](baseline.json) pins both WAV hashes, runtime/device context
and sampling configuration. [Execution evidence](../../reviews/coreai-iphone17pro-bass-preview-cpu-complete/README.md)
records all 50 denoiser/APG rounds and one decoder. The
[read-only WAV audit](../../reviews/coreai-iphone17pro-bass-preview-waveforms/README.md)
validates duration/format/levels and close waveform agreement to existing Linux
references. Four of 491520 samples exceed the original pointwise bound against
the Linux FP32 waveform, so that strict comparison remains NOT PASSED.

The subsequently supplied [per-output report](../../reviews/coreai-iphone17pro-bass-preview-cpu-complete/generation-report.json)
records 85.95 seconds for 5.12 seconds of audio and a 417.19 MiB sampled app
process maximum (not exact peak/total RAM). Eight of ten output comparisons
pass; final-step latent comparisons fail, both decoded waveforms against Mac
references pass, and host `bounded_diagnostic` keeps the overall gate failed.
These are separate from the Linux-reference waveform checks above.
[Next measurement](../../docs/coreai-bass-preview-performance.md) uses the same
case in Release; no latency gain or cold-cache repeatability is yet proven.
Longer clips, raw recording import and three generated tracks remain separate work.

Git stores metadata/measurements; the original uploads and iPhone Documents
retain the WAV files. Models/checkpoints and audio are not added to app resources
or this Git folder. The app can recover an existing saved pair/report for
playback and **Bagikan laporan generasi** without generating it again.

## Runtime pendek yang dipilih

[Report Release/reuse](../../reviews/coreai-iphone17pro-bass-preview-reuse/README.md)
mengonfirmasi 59,24 s, sampled app maximum 450,42 MiB, 15 opens/15 binds/735 hits,
dan kedua hash WAV identik. Route ini dipilih untuk prototype pendek, dengan
pool dilepas sebelum decoder. Cold-start/repeatability dan full-song memory
belum diukur; sampled app footprint bukan total RAM atau exact peak.
