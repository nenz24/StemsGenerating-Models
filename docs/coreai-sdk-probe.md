# CoreAI SDK gate before the iPhone audio prototype

Initial device: iPhone 17 Pro / iOS 27.0. User-reported Xcode: 27.0,
build 27A266a, with iphoneos27.0 and iphonesimulator27.0 SDKs. The user then
reported a successful `import CoreAI` type-check for arm64-apple-ios27.0.
Module import is established by that user output; a custom audio model path
and physical-device execution are not established.

Run on the Mac with that Xcode installation:

```bash
bash scripts/probe_coreai_sdk.sh
```

An optional argument selects the output parent directory. The default creates
a fresh report folder and ZIP on the Desktop; existing reports are preserved.
The probe uses only local SDK/compiler tools. No model files, remote downloads,
GPU, Apple account credentials, connected device or signing configuration are
required for this type-check.

The report contains Xcode/SDK versions, the selected device SDK path, one
`import CoreAI` probe, its compiler status/log, and readable public headers,
module maps and Swift interfaces for CoreAI, CoreAIDelegates, CoreAIRuntime,
CoreAICompiler, CoreAICommon and CoreAIAsset in the SDK Frameworks,
SubFrameworks or Swift overlay locations. These names/locations are grounded in
the supplied compiler report, not a guess about their APIs. It records source-file
counts and missing paths per module. It excludes PrivateFrameworks and compiled module caches.
It also captures the compiler's `-scan-dependencies` JSON and diagnostics with
an explicit success/failure status. Compiler resolution may identify dependencies
outside the guessed public SDK locations; the report does not automatically copy
those dependencies or infer that they are public APIs. Import and scan failures
are captured rather than replaced with a stub or an import of another framework.

The next implementation depends on the real interfaces: model loading/import
formats, tensor/graph execution, supported audio operations and available
accelerators must be established before choosing conversion or training work.
An import-only success does not prove any of those. A missing interface path
also does not by itself prove that no module can be imported elsewhere; read
the compiler result together with available public documentation.

## Uploaded interface review

The user supplied `coreai-api.apDMX4.zip` (1,497 bytes, SHA-256
`7c8d1c8cde67d4f062032c0704fc0991d958ceb17e0952defd2d7c4daa5cb597`).
It contains Xcode 27.0 / 27A266a metadata and one public source interface:
`Modules/CoreAI.swiftmodule/arm64e-apple-ios.swiftinterface` (1,169 bytes,
Apple Swift 6.4). Its only non-standard module import is:

```swift
@_exported public import CoreAIDelegates
```

There are no model/tensor/graph declarations in that wrapper interface and no
CoreAIDelegates interface in the supplied ZIP. This is an incomplete API capture,
not evidence that custom model support is present or absent. The arm64e interface
does not replace the separately user-reported arm64 import test or establish
physical-device execution. The next capture needs public CoreAIDelegates source
interfaces or location diagnostics if they are unavailable. Do not infer a model
format, conversion path or runtime API from the wrapper alone.

The subsequent `coreai-delegates.ig8KPy.zip` is 366 bytes, SHA-256
`c23ebc1adec7ff312b0c5e8357a5e0e8cb921df5d16b20812e750816cab70fd5`.
It contains only an empty `module-locations.txt`, with no interface sources.
The user's SDK-tree filename search captured no matching paths. This does not
establish that the compiler cannot resolve the module elsewhere, that public
custom model support exists, or that it is absent. Stop guessing SDK folders:
the next report must capture compiler dependency resolution for the arm64 device
target (and diagnostics if it fails), then inspect actual resolved public APIs.

The next report, `coreai-resolution.LXF4vO.zip` (61,576 bytes, SHA-256
`8826a535d80f6779c4d10a304a367b27aa9f3f8ccbe3187587e5ca9adc18505f`),
passes archive CRC verification and contains `Dependency scan: OK`, an empty
compiler log and a valid 961,552-byte dependency JSON for `CoreAIProbe`.
The JSON uses alternating module-identity / module-details entries. The CoreAI
wrapper resolves from `System/Library/Frameworks/CoreAI.framework`; the five
other CoreAI modules resolve from `System/Library/SubFrameworks`, each with a
readable-source path ending in `Modules/<module>.swiftmodule/arm64e-apple-ios.swiftinterface`.
The compiler command target is arm64-apple-ios27.0 despite that source filename.

Resolved CoreAI-family dependencies:

- CoreAI -> CoreAIDelegates.
- CoreAIDelegates -> CoreAIAsset, CoreAICommon, CoreAICompiler, CoreAIRuntime.
- CoreAIRuntime -> CoreAICommon (and system dependencies including Metal).
- CoreAICompiler -> CoreAICommon.

All six CoreAI module records have `libraryLevel: api`. Neither this metadata nor
the presence of Metal in dependencies establishes model loading, operators,
accelerator selection, audio support or device execution. The ZIP contains paths
and compiler arguments, not the five dependency interface source contents. The
next collection can address those exact known source files in one pass. Why the
earlier filename walk captured no matches is not established; symlink traversal
is one possibility, not a confirmed diagnosis. No more SDK-root guessing is
needed for these six modules.

## Complete public interface capture

`coreai-interfaces.ZrS4iz.zip` is 11,915 bytes, SHA-256
`f2cf8caff9103c089cbdfb10c7d7a9fc4d804eb20a16117bbc8f54c5e4476ff2`.
Archive CRC verification passes. The status lists all six modules as copied;
their readable interface contents are now supplied, not only paths.

| Interface | Bytes | SHA-256 |
| --- | ---: | --- |
| CoreAI | 1169 | c6c64df0f87473f19faba5269e285093cc7295cb35cc1c0db1a5756d75de0ab2 |
| CoreAIDelegates | 7395 | ddfea9c6411ff0d584cb8bb79df33e379d7db2d85e641d6759a6c0e9bceca1b9 |
| CoreAIRuntime | 63785 | d1344503ae8ea4c0f5f7f6536125c4342562fa3623d6f2c91f26aaf8bb8bfbce |
| CoreAICompiler | 1278 | fcbc24af351a517dabc21ef337313516db1feaeb0efe42cb2613aa2683a41a35 |
| CoreAICommon | 1249 | eeb090a6f63326568b917050e1b0d0cffeb53fcf40e284888a36fe55d5cdd0d3 |
| CoreAIAsset | 12099 | ec950d9015c15f5985b72edc19072a68e8ae88f4355adda5dcd2ccc5c6f61da6 |

The public declarations establish the following runtime API surface for iOS 27.
This is source inspection, not successful compilation or device execution of
these individual calls.

| Capability | Declaration evidence | Limit |
| --- | --- | --- |
| Validate/inspect local assets | `AIModelAsset.isValid(at:)`, `init(contentsOf:)`, `summary(includingStatistics:)` | No accepted file format/extension or exporter is specified by these declarations. |
| Load and specialize | `AIModel.init(contentsOf:options:) async throws`, `AIModel.specialize(contentsOf:options:cache:cachePolicy:)` | Compatibility with ACE-Step assets has not been tested. |
| Find model functions and contracts | `functionNames`, `functionDescriptor(for:)`, `loadFunction(named:)` | Actual function names and tensor shapes come from the exported model. |
| Run tensor inputs | `InferenceFunction.run(inputs: [String: NDArray], ...) async throws -> Outputs` | No model has been run. State/output-view overloads exist; the initial caller intentionally handles only a stateless fixed-shape contract. |
| Read outputs | `Outputs.names`, `Outputs.remove(_:)`, consuming `InferenceValue.ndArray` | Outputs and inference values are noncopyable; lifetime/ownership rules matter in real compilation. |
| Create/read tensors | `NDArray.init(scalars:shape:)`, `shape`, `scalarType`, `view(as:)`, `View.withUnsafePointer` | Shape/type/strides/preprocessing remain model-specific. |
| Compute preferences | `ComputeUnitKind.cpu`, `.gpu`, `.neuralEngine`, `availableKinds`; `SpecializationOptions.cpuOnly`, `init(preferredComputeUnitKind:)` | Available kinds and a preference do not prove a particular model/operator will use that accelerator. |
| Model cache | `AIModelCache.default`, cache policies and targeted deletion APIs | Cache presence does not prove numerical correctness or audio quality. |

NDArray scalar types enumerate FP32, FP16, BF16, several integer widths and
other formats. This is not evidence that all of those types/operations execute
on all accelerators, or that ACE-Step can be quantized to them without loss.
There is no audio-specific descriptor in these interfaces; numeric audio inputs
could use NDArrays only with an established model contract. No built-in music
generator, tokenizer, codec, audio resampler or diffusion scheduler is established.

CoreAICompiler and CoreAICommon contain public imports but no callable compiler,
export or graph-construction declarations in this capture. Therefore the next
unresolved gate is the official model export/conversion tool and valid asset
format. Absence of those Swift declarations does not prove that an external
export tool does not exist. Do not invent an extension, Python package or a
direct `.safetensors` loading route; do not substitute another runtime.

## Prototype prepared from the observed API

[Device and model probes](../prototypes/ios27-coreai/README.md) contain a native
`CoreAIProbe.xcodeproj`, shared scheme and Swift source for a physical-device
NDArray round-trip/compute-availability check and a separate local-model caller.
The model caller requires a valid asset and
explicit function/input contracts. No checkpoint or asset was generated.
The Swift source has not been compiled here: no Apple SDK/Swift compiler is
installed in this Linux workspace. The user subsequently reported successful
device-app execution and tensor round-trip; see the device report below.
No mock runtime is used as evidence.

First run the device probe in Xcode on the iPhone. In parallel, establish the
official exporter and create a tiny known numerical model with reference
inputs/outputs. Then load/run that asset and measure memory/latency before a
short one-instrument audio test. The CUDA/GarageBand mix remains the quality
reference. Distribution of the project does not establish successful Apple
compilation, signing, physical-device execution or model conversion.

## User-reported first device result

The [preserved device report](../reviews/coreai-iphone17pro-device-probe/README.md)
records a successful Float32 [1, 4] NDArray read-back on an iPhone with OS
27.0.1 (24A446), CoreAI architecture string `h18p`. Available compute kinds
are reported as CPU/GPU/Neural Engine. This advances the app/tensor gate from
source inspection to user-reported physical-device execution. No model is
loaded or run, and accelerator placement, model export compatibility, audio
quality, memory, inference latency and thermals remain unverified.

## Next numerical compatibility candidate

**Retired after device rejection.** The user reported `Missing hash file` during
specialization of the raw package; no inference ran. They then corrected the
native format to `.aimodels` and supplied https://developer.apple.com/core-ai/ .
The legacy button and runner have been removed; the planned `.mlmodelc` follow-up
was cancelled before publication/execution. The account's proxy denied access to
the official URL (HTTP 403), so its exporter commands and hash schema have not
been read or verified. Use a documented native export; do not rename the legacy
package or invent the missing hash. See [device rejection](../reviews/coreai-iphone17pro-raw-candidate/README.md).
The following describes the archived experiment, not a current recommendation.

`ModelCandidates/affine.mlpackage` and its JSON reference are a tiny ML Program
package produced by Core ML Tools 9.0, not a claimed CoreAI-native export.
The graph is `y = 2*x + 0.25`, Float32 shape [1, 4], function `main`, input `x`,
output `y`. Inline constants mean no trained weights; a tracked empty marker
preserves the weights directory referenced by the package manifest after pull.
The reference records every package file's SHA-256. No Apple prediction was
performed in Linux. Tests reopen the saved package without model loading and
inspect its serialized constants, wiring, types and reference arithmetic.

This is a bounded compatibility experiment: the new button reports source asset
validation and asks CoreAI's loader to accept or reject the source. All model
calls use CoreAI. The SDK umbrella is still the only linked AI framework, and
the folder resource is copied raw to avoid generated Core ML wrappers. No
alternative runtime is called on failure. Generic callers validate assets by
default; the explicit compatibility experiment delegates source acceptance to
the loader even if the asset validator reports false.

CoreAI acceptance, the updated Swift build and numerical inference remain
unverified until the user runs the new button. A tiny numerical parity pass
would establish only that particular model/format path, not ACE-Step operator
coverage, music quality, Neural Engine placement or production performance.
The first run and two warmed runs use `.cpuOnly` and include output read-back;
model load/specialization timing is recorded separately.

## Official documentation now accessible; native export prepared

The published network change enabled HTTP 200 retrieval of the official CoreAI
integration and AOT guides. They specify `.aimodel` source and `.aimodelc` AOT
variants. This supersedes the earlier unverified `.aimodels` requirement and
blocked-page notes. Apple's exporter is `coreai-torch`, using `TorchConverter`
and `AIProgram.save_asset`. The native affine export was actually produced with
coreai-torch 0.4.3/coreai-core 1.0.0b3/PyTorch 2.8.0+cpu, including the native
`main.hash` emitted by the exporter. No hash schema was guessed.

See [native export guide](coreai-native-export.md) for authoritative sources,
reproduction and exact validation scope. Export and PyTorch reference generation
are verified; current Swift compilation and native inference on iPhone are not.
An optional Linux in-package runtime smoke was terminated without a completed
result; it supplies no model-inference or parity evidence.

## Native affine inference confirmed by user device report

The [native model report](../reviews/coreai-iphone17pro-native-affine/README.md)
advances the tiny-model gate to PASS on iPhone/iOS 27.0.1/h18p. Asset validation
is true, model load succeeds, and all three inference runs match exactly.
Load/specialization is 299.960625 ms; first inference 117.172042 ms; two warm
runs 0.817167 and 0.722708 ms. Cache state and actual hardware placement were
not reported. This supersedes pending-device statements for this affine test,
not for audio/ACE-Step. The export-time reference manifest remains an immutable
record of the original export; subsequent device evidence is stored separately.
