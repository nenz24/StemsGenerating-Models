# 10.24-second device attempt: first step complete, cached bind fails

Run `EFF5CCF3-8A93-43B3-B0D9-CAB8C74933E7`, started
2026-11-02 16:04:13 UTC (23:04:13 WIB). User-supplied [report](report.txt) and
[console](console.txt), after the first-step scope mitigation in `742388e`.
iPhone 17 Pro, iOS 27.0.1 (24A446), h18p; Release, Swift `-O`, cpuOnly requested.
Actual accelerator placement is unmeasured. User reports 148 GB free storage.

| Observation | Result |
| --- | --- |
| Imported recording | `piano.mp3`, first 10.24 seconds |
| Source encoding | Completed; new-input numerical parity not evaluated |
| Fixed conditioning | 27/27 stages, 6/6 comparisons PASS |
| First actual sampling step | All 14 denoiser assets plus APG completed, 12.379366875 seconds |
| Failure | Step 2, stage 4, `blocks-04-05.aimodel`, initial function bind |
| Error | `BNNSCoreAIDelegate.BNNSDelegate.BNNSError.graphLoadFailed`, domain/code 1 |
| Completed sampling calls | 18; 19 successful model opens; 18 binds; 0 handle hits |
| Completed steps / pipeline time | 1/50; 25.517598708 seconds |
| Sampled app footprint maximum | 761.7211227416992 MiB, includes source component |
| Footprint before failing bind | 93.11605072021484 MiB |
| Bass waveform | Not generated |

The formerly failing `blocks-02-03` completed. The new failure occurs while
binding a cached graph that completed inference in step 1. This establishes a
cache-loading failure, not proven file corruption, storage exhaustion, or a
numerical failure. App footprint measurements do not bound native memory or
transient compiler allocations. Imported-input output-reference checks remain
0/0, NOT EVALUATED; no full generation or numerical PASS is claimed.

The new mitigation allows at most one targeted cache deletion and retry per
10.24-second generation. Eligibility requires the exact observed BNNS
domain/code/graphLoadFailed signature at initial function binding, before
inference; no retained entry for the asset may exist. The failed model scope
has unwound. Only its derived cpuOnly cache entry is deleted using the public
`AIModelCache.default.deleteEntry(for:options:)` API. The same stage then receives
the same input NDArrays, with no step restart or replay. The retry cannot trigger
another repair. Failures elsewhere still stop the pipeline.

Existing first-step sequential ownership and later retained handles remain.
A successful recovery adds one model open, no successful bind or inference for
the failed attempt: expected 31 opens, 30 successful binds, 720 hits and 750
sampling calls. Decoder runs separately afterward. Logs preserve the original
failure and recovery outcome. Default purge policy and 5.12-second behavior are
preserved; no cache-wide deletion, persistent cache policy or asset re-export.

Apple documents [targeted cache removal and deferred deletion while an AIModel
still owns an entry](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching).
The API also appears in the user's uploaded SDK interface. This recovery has
not been executed on Apple hardware in the Linux workspace; deletion does not
prove physical eviction or successful recompilation.

Validation: the seven window/selection/precision contract tests and six host
trajectory-diagnostic gate tests pass. These do not execute the Swift recovery
path or prove device success; native cache recovery remains pending an iPhone
run. The source/weights and numerical gates were not re-exported or changed.
