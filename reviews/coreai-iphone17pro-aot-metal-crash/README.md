# h18p AOT/default native compiler crash

The [unmodified supplied saved report](report.txt) verifies all sixteen AOT
artifacts and their source identities. Actual piano source encoding passes all
four output checks; the complete 27-stage Qwen/ConditionEncoder passes all six
checks. The last saved phase is sampling step 1/50, stage 1/14, `front.aimodel`.
Its resolved AOT resource is `front.h18p.aimodelc`, loaded with `.default` to
match compiler preference `none`. No completed sampling step or bass WAV is
reported. Host status remains **bounded_diagnostic**, not strict numerical PASS.

The subsequent [user-supplied LLDB backtrace](backtrace.txt) shows thread 14,
`llvm-worker-0`, EXC_BAD_ACCESS code 1 at address `0x3f5`. The top frames are
MetalPerformanceShadersGraph's
`MPSGraphAICodeCompilerDelegate.getInitializedAICodeBytecodeWithPayloadPrefix:delegateId:`
and `Compiler_coreAI.compile`, followed by CoreAICompiler/ODIE serialization.
This locates a native Metal/MPS compilation failure. It does **not** prove GPU
inference completed, that RAM was exhausted, an exact null-pointer cause, or a
particular toolchain/OS incompatibility. The prior PointerUI messages do not
explain this compiler stack. The report did not yet distinguish model-open,
function-bind, backend-await, readback or destruction; no such interval is
retroactively claimed.

## Mitigation prepared

`COREAI_MODEL_PROFILE=auto` now selects source models and the existing
`.cpuOnly` loader, even if AOT files are present. The build stage prunes only
unused AOT copies from the app product; original compiled variants, weights,
references, checkpoints and working cache remain on the Mac. `bass-aot` is still
an explicit diagnostic profile, retaining its matching `.default` runtime
policy. The CLI has no supported CPU-only compile option; compiler `none` is
not relabeled CPU, and a CPU-only AOT load with mismatched options is not assumed
valid. Default `.cpuOnly` requests use the source assets instead.

The pipeline now persists a run UUID/time and operation markers immediately
before model-open (including asset validation), bind/validation, memory guard,
backend-await, readback and Swift scope release. Markers include the resolved
filename and requested CPU-only policy. They are written synchronously before
the native call and printed to the Xcode console; high-level UI progress still
uses its existing MainActor callback. A native access violation cannot be
caught as a Swift error. API timing intervals exclude marker writes where
possible; total pipeline/other intervals include diagnostic overhead.

This is a targeted route around the observed Metal compiler crash, not proof of
successful bass generation. Previous source CPU/BNNS compilation memory failure
remains a device risk. Single-live-stage ownership, existing targeted raw cache
repair and the 2000 MiB process guard remain in place; none guarantees compiler
peak memory or native eviction. Build/device validation of these changes is
pending; Linux tests validate packaging, reference math and integrity only.
