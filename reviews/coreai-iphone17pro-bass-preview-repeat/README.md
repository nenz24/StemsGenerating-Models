# Profiled same-output run: high repeated backend cost

The unchanged [supplied report](report.txt) identifies run
`203AF0B8-7FDA-41FC-AB0E-CD542E1EF560`, started 2 November 2026, **18:23:13 WIB**.
[Derived measurements](summary.json) preserve scope and checksums.

| Measurement | Earlier report | This report |
| --- | ---: | ---: |
| Audio | 5.12 s | 5.12 s |
| Total pipeline | 85.9538 s | 755.2927 s |
| Sampled app-process maximum | 417.19 MiB | 436.47 MiB |
| Completed sampling steps | 50/50 | 50/50 |
| Mac-reference waveform checks | PASS 2/2 | PASS 2/2 |
| Overall numerical gate | NOT PASSED | NOT PASSED |

The run is about **8.787 times slower**. It still reports 750 sampling opens,
750 binds, zero retained hits, all 50 steps complete and decoder/WAV success.
Crucially, embedded hashes for **both WAVs exactly match the listener-approved
baseline**. This binds the new report to that same audio content; no new audio
upload/listening review is required merely to establish bit equality. Final
latent comparisons and host `bounded_diagnostic` keep the strict gate failed.

Completed sampling/decoder API intervals total 730.3833 s in backend-await,
1.4468 s in model-open, 0.6393 s in bind, and 6.4525 s in output copy.
**96.7% of entire pipeline wall time is backend-await.** Repeated asset calls
alone account for 714.4935 s of backend-await. Those API intervals may include
initialization/deferred compilation, execution or scheduling; they are not
isolated kernel measurements and do not prove repeated compilation.

## Build metadata failure

Both new build fields say **unrecorded**. The supplied report does not verify
Release or `-O`, so the slowdown must not be attributed to Release as a proven
cause. The custom `INFOPLIST_KEY_*` approach did not reliably put these values
into the generated plist on the user's build. The project now uses a tracked
base `CoreAIProbe-Info.plist`, with explicit build-setting expansion, alongside
the existing generated standard values. A compile-time DEBUG on/off field is
also recorded as an independent fallback. The plist is not added to Resources.
Native build verification of this fix remains pending.

## Next targeted experiment: same-shape retained handles

The new **CoreAIProbe Reuse** scheme runs Release with
`COREAI_PREVIEW_REUSE=1`. Existing Debug/Performance schemes do not set it.
The fixed 5.12-second source CPU case lazily retains both model AND function
for **14 DiT shards + one APG asset**, across the same 50 sampling steps.
Expected sampling counts are **15 opens, 15 binds, 735 reuse hits, 750 calls**.
The decoder still has a separate scope, after the sampling pool is explicitly
released. The pool is also cleared on failure. Source encoder and Qwen handles
never enter this pool; at most fifteen sampling entries are allowed.

This uses the existing retained-stage implementation already validated for a
[smaller synthetic four-step full loop](../coreai-iphone17pro-retained-loop/README.md).
That result does not validate the larger batch-two real-conditioning case.
The existing before-stage and post-bind 2000 MiB app footprint guards remain.
Retaining handles can increase live memory; neither these guards nor Swift
release guarantees compiler peak memory or native cache eviction. No automatic
retry/fallback changes the tested mode after an error. Ordinary one-scope mode
remains available. AOT is rejected for this retained experiment; source assets
and CPU-only requests are required.

Actual inputs, outputs, 50-step schedule, APG, precision, references and strict
bounds stay unchanged. Host tests cannot measure this CoreAI runtime strategy;
Apple/device build, shape-specific memory and timing remain pending. No speedup
is promised from a previous component benchmark or the synthetic loop.
