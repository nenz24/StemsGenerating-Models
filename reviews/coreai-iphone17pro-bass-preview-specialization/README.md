# First integrated device run: sampling specialization blocked

The user supplied the physical iPhone [report](report.txt); its bytes are
preserved, with SHA-256 in [report.json](report.json). The caller successfully
loads the bounded diagnostic contract and verifies component identities,
noise, schedule and ten numerical references.

Actual source encoding passes all four output comparisons in one integrated
call. Full Qwen/ACE conditioning passes all six comparisons, all 40 layers and
27 stages. These components now execute sequentially inside the integrated
caller. Source time is 1480.79525 ms; conditioning time 10593.091375 ms.

The pipeline then reports `CoreAIDelegates.AIModelError.failedToSpecialize`,
**0/50 completed sampling steps, 0/10 sampling/output comparisons**, total
18625.629 ms. It generates no WAV. The original final report drops the active
stage/progress line, so it does **not** identify the failing asset or distinguish
model opening, function binding or deferred backend compilation. Zero completed
steps does not prove the first front asset failed: a later shard or APG asset
within step 1 could also fail. No specific operator, dtype or batch-size cause
has been established.

Whole-pipeline sampled app-process maximum is 790.3313446044922 MiB, baseline
27.111106872558594 MiB. These 10 ms samples do not measure exact peak,
compiler-worker memory or total model/device RAM. The initial app report alone
does not isolate or exclude memory as a cause.

The user's subsequent Xcode console supplies a concrete BNNS compiler error:
`mmap of bnns ir file for writing failed with error: Cannot allocate memory`.
This confirms allocation failure while writing the BNNS IR mapping, without
identifying the exact source asset, allocation size or process limit. PointerUI
and LaunchServices messages in the same console do not establish the model
failure's cause. Asset/phase tracing remains useful alongside this evidence.

The separate Mac terminal excerpt now passes decoding comparison: normalized
RMSE 6.505227779631598e-5, max error 9.968876838684082e-5, strict waveform PASS.
Host preparation remains bounded diagnostic because latent pointwise parity
failed earlier. That excerpt ends at staging; it does not include Ready. The
device report nevertheless shows its prepared diagnostic reference is present.
Host numerical status and device specialization failure are separate blockers;
changing tolerances would not resolve the runtime specialization error.

The trace update preserves the last pipeline phase in final output, records
each first-step completed asset, and wraps retained-stage errors with exact
asset, operation, reused-handle status and bounded NSError details/underlying
chain. APG execution gets its own progress line. Counters show completed calls,
opens, binds and hits even on failure. There is no automatic retry, cache
clearing, reference regeneration or model/math change.

Apple documents [ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)
on the Mac to move expensive compilation out of the device. It still leaves
some specialization on device. The user's `xcrun coreai-build compile --help`
currently fails because the utility is unavailable. Apple's documented
prerequisite is `xcodebuild -downloadComponent MetalToolchain`; installed-tool
help is needed before choosing CPU and architecture arguments. AOT has not
been executed or validated here.

Updated trace build and device execution are pending. The trace-only app retry
uses the existing prepared models, with no reference generation/export needed.
See [device retry instructions](../../docs/coreai-bass-preview.md).
