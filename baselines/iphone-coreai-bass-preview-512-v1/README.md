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

The per-output Mac/iPhone generation report is still required for its ten
comparison results, total latency and sampled pipeline footprint. Conditioning
measurements alone do not establish these. Cold starts, repeatability, longer
clips, raw recording import and three generated tracks remain separate work.

Git stores metadata/measurements; the original uploads and iPhone Documents
retain the WAV files. Models/checkpoints and audio are not added to app resources
or this Git folder. The app can recover an existing saved pair/report for
playback and **Bagikan laporan generasi** without generating it again.
