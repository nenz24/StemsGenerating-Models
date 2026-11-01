# Monolithic denoiser: reported memory-resource failure

The user supplied [the Xcode exception](error.txt) and
[the last persisted report](last-report.txt). The report identifies iOS 27.0.1
(24A446), h18p and an iOS device build. The previously user-identified device is
iPhone 17 Pro. Source commit and local asset hashes were not emitted.

Native source validation returned true in 1.24475 ms. The last durable phase
was `loading/specializing complete model`. No inference output, completed model
load or numerical parity was reported. The memory exception is not evidence of
a numerical mismatch or an unsupported tensor operator.

The diagnostic says `RESOURCE_TYPE_MEMORY: high watermark memory limit exceeded`
with `limit=3376 MB`. The thread label `llvm-worker-0` does not establish a separate
worker process or mean this is the device's total RAM. The UI remained visible
but unresponsive according to the user. An Xcode pause on the exception can
explain a frozen UI; debugger state/stack trace was not supplied, so it is not
treated as independently confirmed.

The monolithic device route is retired. The default preparation now exports
14 ordered assets: front projections/embeddings, twelve pairs of original
layers, and tail projection. Only one stage is held in the Swift caller at a
time and results are copied into independent CPU NDArrays before its scope ends.
CoreAI-internal cache eviction remains unverified. A sampled process-footprint
check stops before the next stage at 2000 MiB; it does not cap in-stage peaks.

The failed monolithic resource is preserved under ignored cache outside the
app's copied resource directory. The six accepted component bundles remain
available. See [updated device steps](../../docs/coreai-full-denoiser.md).
