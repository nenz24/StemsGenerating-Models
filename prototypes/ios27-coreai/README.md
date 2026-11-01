# CoreAI device and model probes

Initial target: iPhone 17 Pro, iOS 27.0, Xcode 27.0 / 27A266a. These Swift
sources use APIs declared in the six public interfaces supplied by the user.
They have **not been compiled, linked or run on Apple hardware** here: this
workspace is Linux and has no Apple SDK. No stub CoreAI implementation is used.
The folder includes a native Xcode project and shared `CoreAIProbe` scheme.

## Run the device probe

Di Terminal Mac, jalankan dari folder repository:

```bash
git pull --ff-only
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

1. Di Xcode pilih project **CoreAIProbe**, lalu target **CoreAIProbe**.
   Buka **Signing & Capabilities**, biarkan **Automatically manage signing**
   aktif, lalu pilih **Team** akun Apple Anda. Team ID sengaja tidak disimpan
   karena harus sesuai dengan akun signing di Mac Anda.
2. Sambungkan iPhone 17 Pro, buka kunci, dan pilih **Trust/Percayai** jika diminta.
   Pilih scheme **CoreAIProbe** dan iPhone tersebut di toolbar Xcode, lalu tekan
   **Run** atau **Cmd+R**. Deployment target sudah disetel ke iOS 27.0.
   Jika diminta, aktifkan **Settings > Privacy & Security > Developer Mode**
   di iPhone dan ikuti instruksinya.
3. Tap **Jalankan pemeriksaan** and share the resulting text. Report any compile,
   link or launch failure as a failure; successful import alone is insufficient.

This allocates a four-element Float32 NDArray, checks its type/shape and reads
the values back. It lists the compute kinds reported as available and the CoreAI
architecture string. It needs no microphone permission, model, network or GPU
download. A Simulator result is labelled and is not an iPhone measurement.
Reported compute availability does not prove a model ran on that accelerator.
The device probe does not call model inference or measure audio generation.

## Compile the model caller

`CoreAIModelProbe.swift` is already included in the target to type-check the
caller. It does not run automatically. It accepts a valid local asset and an explicitly named
function with fixed-shape NDArray inputs and no state/interleaving. It rejects
unsupported contracts instead of inventing source-to-audio preprocessing.
It can enumerate actual function names through `probe.model.functionNames`.
The caller uses `SpecializationOptions.default` unless a preference is supplied;
`preferredComputeUnitKind` is a preference, not proof of actual placement.

This caller is not an exporter, model or audio generator. An API such as
`AIModel(contentsOf:options:)` does not imply safetensors or PyTorch files are
accepted. No supported asset extension or ACE-Step conversion route has been
established. The supplied CoreAICompiler and CoreAICommon public interfaces
contain imports but no compiler/graph-construction declarations.

Before audio tests, establish the official export tool and create a tiny known
model asset with reference inputs/outputs. Load it on device, run the function,
compare values to the reference, and measure load/specialization and warmed
inference separately. Then attempt one audio component and a short one-instrument
trial. End-to-end ACE-Step compatibility, quantization, operator support, peak
RAM, generation latency and thermal behaviour remain unverified.

API evidence and limitations: [SDK review](../../docs/coreai-sdk-probe.md).

## Optional command-line build on the Mac

From the repository root, this compiles and links without device signing or
installing an app:

```bash
xcodebuild \
  -project prototypes/ios27-coreai/CoreAIProbe.xcodeproj \
  -scheme CoreAIProbe \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/CoreAIProbeBuild \
  CODE_SIGNING_ALLOWED=NO build
```

The project links the SDK's public CoreAI umbrella framework. SwiftUI/UIKit
use normal SDK autolinking. No CocoaPods, package fetch, project generator,
model download, audio assets or custom build script is required. Xcode build
products and user-specific workspace files are excluded from Git.
