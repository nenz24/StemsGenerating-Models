# Sampling failure localized to blocks 18–19

The user supplied [report.txt](report.txt), preserved byte-for-byte with its
SHA-256 and structured summary in [report.json](report.json). The new tracing
successfully identifies **stage 11/14, `blocks-18-19.aimodel`, during
function-bind/input-validation**, with `BNNSDelegate.BNNSError.graphLoadFailed`
(domain code 1). It is a new handle, not a reused one.

Source encoding and full bass conditioning still pass. The first ten sampling
assets execute successfully, including front and pairs through blocks 16–17.
There are ten completed calls, eleven model opens, ten binds and zero hits;
step 1 remains incomplete (0/50), with no waveform/output checks or generated
WAV. Total time is 19683.4925 ms. Sampled app-process maximum is
687.7238235473633 MiB, baseline 35.12675476074219 MiB. These samples do not measure
compiler-worker memory or total model/device RAM.

The earlier console showed BNNS failing to allocate a write mapping for its IR.
Ten prior model/function pairs were intentionally retained in this run. Their
retention is a controllable source of resource pressure; this evidence does not
prove it is the sole cause or that the cached IR is corrupt. The latest general
PointerUI/LaunchServices messages do not identify a different model failure.

The preview now uses **one Swift model/function scope per asset**, with the
existing detached CPU output copies. That scope is released before the next
asset starts. There is no cross-step retained pool for this real-shape preview.
Expected sampling counts become 750 opens, 750 binds, 750 completed calls and
zero retained-handle hits, versus the earlier 15/15/750/735 plan. It may be
slower; actual latency and native residency remain unmeasured. The old retained
synthetic probes remain available and unchanged for their tested shapes.

Before opening any pipeline model, the caller checks its saved prior report.
Only if that report names this asset and its BNNS graph-load error, it requests
`AIModelCache.default.deleteEntry(for:options:)` for **this CPU-only compiled
entry**. The API signature is verified in the user's public SDK interface.
The recovery action/result is logged. A deletion failure is reported, then the
single-stage path is still attempted. This does not delete source model assets,
weights, checkpoint downloads, references, other cache entries, or generated
WAVs. There is no hidden retry or global cache reset.

Model math, shapes, prompt, source audio, 50-step APG sampler, tolerances and
diagnostic labels are unchanged. No host reference regeneration, export or
Metal Toolchain download is required for this runtime change. AOT remains an
optional future mitigation once the Mac compiler is available.

Apple compilation and device validation of the new ownership/cache recovery
path are **pending**. Backend cache eviction is not guaranteed by releasing
Swift handles. Pull, Stop/Run once in Xcode, press the same generation button
and share the final report/WAV if produced. [Guide](../../docs/coreai-bass-preview.md).
