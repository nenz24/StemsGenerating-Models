# Complete bass conditioning: iPhone PASS

The user supplied the physical iPhone [report](report.txt). The [structured
record](report.json) includes its SHA-256, all 27 stage profiles and six output
comparisons. This is user-reported execution, not an independent Linux inference.
The report requests `cpuOnly`; accelerator placement is unmeasured.

All 28 Qwen blocks and all 12 ACE lyric/timbre blocks execute, with both token
embedding lookups, projection, normalization and packing. The device completes
**27/27 stages and 6/6 comparisons**. Qwen text embeddings, final condition and
condition mask pass both the storage16 pipeline and original FP32 references.

| Output vs original FP32 | Max absolute error | Normalized RMSE |
| --- | --- | --- |
| Text embeddings | 0.00018310546875 | 3.6495256000142243e-6 |
| Condition | 0.00009918212890625 | 3.6272825649971524e-6 |
| Mask | 0 | 0 |

The absolute-plus-relative bound and normalized RMSE are unchanged. Final
condition is `[1,117,2048]`, mask `[1,117]`. This is the fixed bass prompt,
host-tokenized 105 text/11 lyric tokens, empty lyrics, no padding and publisher
silence as the 750-frame no-timbre reference. Actual neural conditioning runs
on the device; references are used only for comparison.

Total probe time is **9462.808417 ms**. Summed stage phases are load/bind
8239.081001 ms, backend-await 573.068332 ms and readback 205.769415 ms.
The remaining 444.889669 ms is outside those intervals, including validation,
comparisons and reporting. These API wall intervals are not pure kernel time,
guaranteed cold-cache timings or full-generation estimates.

Sampled app-process maximum is **783.9873199462891 MiB**, baseline
24.626731872558594 MiB. Ten-millisecond sampling is not an exact peak or total
model/device RAM. This report does not measure a combined pipeline, backend
cache eviction or sustained generation. Adding its footprint to separate audio
or denoiser peaks would not establish combined memory requirements.

**Decision: stage-2 component gates are complete for the fixed prototype
fixture.** The [piano audio-context gate](../coreai-iphone17pro-source-context/README.md)
also passes separately. No repeat of the same conditioning gate is needed now.
Proceed to stage 3: integrate actual source context, bass condition/mask,
shape-matched denoiser, sampling/guidance and decoder to a short WAV.

This closes a component gate, not the complete on-device recording workflow.
Canonical PCM decoding/resampling and tokenization remain on the Mac; arbitrary
prompt entry and raw recording import on iPhone are untested. The two
conditioning paths have not run jointly with the denoiser/decoder, and no bass
waveform or musical-quality result exists yet. The former synthetic denoiser
contract (25 latent frames/64 condition tokens) must be adapted to the real
128-frame source and 117-token condition before integration.
