# First drum device run stops at initial specialization; no waveform

User-supplied [report](report.txt), run `5E09A91D-F4A2-4CB9-8B38-BD530B9ACBCD`,
started **4 November 2026, 23:02:12 WIB**. Release/-O, DEBUG OFF, iOS 27.0.1
(24A446), h18p; cpuOnly requested, actual placement unmeasured.

Correct drum preset SHA-256 and 111 text / 123 condition tokens are reported.
Source encoding completes; all 27 conditioning stages and six matching preset
comparisons PASS. Reported original recording/PCM hashes match the bass run.
Host provenance is `bounded_diagnostic`, not a new full numerical PASS.

Sampling completes front and eight pairs through **14–15**, nine calls total.
At first-step stage 10, **blocks-16-17.aimodel**, initial bind fails with BNNS
domain/code 1 `graphLoadFailed`. The existing single targeted repair removes
only that derived cpuOnly entry, then retries the same stage once. The retry
fails earlier during **model-open/specialization**, CoreAI domain/code 1
`failedToSpecialize`. No successful retry and no inference for that asset.

Final accounting: **0/50 steps, 9 completed calls, 10 successful model opens,
9 binds, 0 reuse hits; repair 1/1, successful repaired executions 0**. Decoder
and WAV generation are not reached. Total 12.928 s, sampled process maximum
762.831 MiB, also reached during source encoding. This is not exact compiler
peak/total RAM. The report does not establish memory exhaustion, corrupt cache,
disk exhaustion or a defective exported graph. It supplies no underlying cause
for `failedToSpecialize`. Strict numerical failure is not the stopping cause.

[Summary](summary.json) records report SHA-256, exact nine-stage order, 54
per-asset and six aggregate phase-sum checks, accounting and shared input-hash
checks. No native replay or new waveform was performed in Linux.

## Next: isolated explicit native preflight

Do not increase retries, purge all caches, change precision/gates or blindly
re-export. The source/conditioning pipeline has already loaded many native
assets before sampling. A fresh-process probe can test the failed asset before
that work, then exercise the other sampler/decoder assets once.

[Preflight instructions](../../docs/coreai-native-preflight.md) use a new
opt-in button, no piano, independent deterministic synthetic inputs and no WAV.
The matching saved failure is ordered first when available. No generation
stage is replayed, no cache deletion requested, no production policy changed.
Completion demonstrates finite standalone execution only; it does not establish
native cache persistence, parity, actual-input completion or a repair.

If the failed asset still fails first in a fresh process, stop and retain the
report plus final BNNS/Xcode console lines. If all sixteen complete, stop the
process and launch the same installed app from its icon before one normal drum
generation. Do not rebuild between preflight and generation. That tests whether
standalone preparation helps; it is not a controlled cold-cache benchmark.

Eight focused instrument-contract regressions pass; static order/API/lifecycle
checks and whitespace checks pass. New Apple build/preflight runtime remain
pending. Accepted bass, checkpoint, source recording and all bundles are retained.
