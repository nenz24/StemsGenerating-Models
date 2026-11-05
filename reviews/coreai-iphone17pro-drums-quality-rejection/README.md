# Drum groove fits; glitches still block audio acceptance

User-supplied [report](report.txt), run `3A1FD8B9-86EF-4DBA-8625-3980E130BB49`,
started **4 November 2026, 23:32:34 WIB**. Release/-O, DEBUG OFF, iOS 27.0.1
(24A446), h18p. cpuOnly requested; placement unmeasured.

Execution completes **50/50** and writes three stereo Float32/48 kHz, 10.24 s
WAVs. This supersedes the previous initial-bind failure as the latest observed
execution result. No preflight report was supplied, so completion cannot be
attributed to preflight, warmup or a proven cache repair. Initial feedback
reported glitches throughout and insufficiently audible drums. The user then
clarified: **the drum groove fits, but many glitches mask the drum sound**.
Groove is accepted; overall audio quality remains **NOT ACCEPTED** because of
artifacts. Preserve the current preset/groove and generation parameters during
diagnosis. Masking by glitches is not evidence that the stem needs a gain boost.
No timbre, fill or full-song acceptance is inferred from groove feedback alone.
The earlier bass-only waiver does not cover these glitches.

Subsequent user listening check: **drum glitches stay the same with the
headroom playback button**. The attenuation comparison has not resolved the
problem. Mix-headroom listening has not been reported. This does not prove all
playback paths are fault-free or identify BF16/native generation as the cause.
Do not repeat drum headroom or increase gain as a presumed fix. Next compare
already-prepared original FP32 and partition-host references on the Mac,
without model inference/re-export, preserving the accepted groove and gates.

That host comparison is now reported: **both original FP32 and partition-host
headroom WAVs still contain glitches**. This reproduces the complaint outside
the iPhone and without BF16 partitioning; it does not isolate a unique cause.
[The Colab conditioning audit](../coreai-drums-colab-conditioning-audit/README.md)
finds a literal lyric-input mismatch and prepares one host-only controlled
candidate. Do not repeat headroom/host export or run another native generation
as the next diagnostic. Existing raw report and numerical waveform audit remain
unchanged; subsequent listening feedback is recorded separately in the summary.

The [lyric-only Mac candidate](../coreai-drums-instrumental-lyrics-result/README.md)
has now completed with four exact-zero control replay checks, but both audio
variants are reported glitchy. Do not adopt the tag-only change. Current Mac
source PCM differs from this device import; obtain that run's source metadata,
waveforms and saved latents before another model hypothesis is tested.

## Report and waveform checks

[Summary](summary.json) and [audio audit](audio-audit.json) retain exact hashes
and measurements. Uploaded audio stays local, not committed to the repository.
All three uploaded WAV SHA-256 values match the report. All samples are finite.

| Uploaded waveform | Sample peak | RMS | Samples with abs >1 |
| --- | ---: | ---: | ---: |
| Drum | 1.357289 / +2.653 dBFS | 0.195886 | 5,962 |
| Piano | 0.669804 / −3.481 dBFS | 0.083206 | 0 |
| Mix | 0.980000 / −0.175 dBFS | 0.120833 | 0 |

`Float32(0.9601622) * (Float32(0.78)*piano + Float32(0.55)*drums)` reproduces
every mix sample **bit exactly**, max error 0. Imported planar PCM hash also
matches when reconstructed from the piano WAV. No summation/export corruption
was found. Raw drum samples >1 remain preserved in Float32; playback can clip.
An 8x polyphase estimate gives true peak +2.659 dBFS for drum and −0.156 dBFS
for mix (filter-dependent estimate). Clipping alone does not explain the mix
complaint or prove the generated waveform is good.

Whole-clip FFT energy puts **87.54%** of drum energy at 20–150 Hz. That is
consistent with strong low-frequency content but does not identify kick/snare,
prove missing kit components, measure perceived loudness or establish what the
iPhone speaker reproduces. No listening judgment or automatic glitch detection
was performed by this numerical audit. Finite values are not musical acceptance.

Report accounting: correct drum preset hash and 111/123 text/condition tokens;
conditioning **6/6 PASS**; 30 opens, 30 binds, 720 hits, 750 sampling calls,
one decoder call, repair **0/1**. Source/output import parity remains **NOT
EVALUATED (0/0)**. Host provenance remains **bounded_diagnostic**. Total pipeline
85.824 s, sampled process maximum 809.612 MiB; neither exact peak nor total RAM.
First step 12.493 s, repeated-step median 1.007 s. All 50 post-step thermal
samples report 0. Six per-asset phase sums and six first+repeat aggregate sums
match report totals; completed-call counts sum to 751. No controlled comparison
to bass or claim of cold-cache performance is made.

Prepared host PCM hash `9517cc69…10bb945` differs from this imported PCM
`c79163af…d57278`. A direct host-vs-upload waveform error would conflate input
resampling with generation differences. Matching-input native baseline remains
available, but first listen to existing original FP32 and partition-host WAVs.

## Concrete next comparison

[Current instructions](../../docs/coreai-drums-quality.md) collect existing
current Mac waveforms/source metadata/latents, without inference. The lyric-only
candidate did not resolve the complaint. No production/native change is made.
The earlier comparison tools below remain available for reproduction.

The previous instructions use existing output and host
references. The opt-in app buttons create attenuated temporary playback copies
with a −3 dBFS sample-peak target; originals are hashed before/after. The host
export command uses full existing integrity gates, writes raw and common-gain
headroom copies to a new folder, and performs zero model calls/native exports.
It preserves diagnostic host status, bundle contents and active selector.

No production sampler, source import, checkpoint, preset, precision, APG, mix
gain or tolerance changes. No limiter/EQ, cache deletion or generation retries.
No claim that attenuation fixes the generated drums. Guitar/full-song musical
validation stays pending drum diagnosis and acceptance.

Six new host tests cover Float32 stereo/order/over-unity preservation, common
attenuation, invalid PCM/gain, gate failure, destination preservation and
unchanged input/selector hashes. Verifier boundaries in exporter tests are
mocked with explicitly nonexecutable fixtures; no native execution is claimed.
[Validation](validation.json): 141 host tests PASS, two optional native/export
tests skipped, Python syntax and whitespace PASS. Listening-copy smoke checks
on the uploaded drum/mix reach the headroom target and preserve both originals'
hashes. Production sampler/decoder source is byte-identical; source import is
identical after excluding the new playback helper. Apple SDK/build/playback and
actual Mac reference export remain pending.
