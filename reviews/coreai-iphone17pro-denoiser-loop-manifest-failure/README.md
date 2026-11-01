# Initial four-step device attempt: manifest validation failed

The [user-supplied report](report.txt) shows `ShardError.invalidManifest`, zero
completed steps/stages and a total of 23.277916 ms. It contains no stage-loading
phase or model execution. This is a validation failure before inference, not a
measured numerical mismatch, BNNS failure or memory exhaustion. The compiled app
ran on the device; four-step model execution and parity remain unverified.

The original caller emitted the same error for many guards. Consequently this
report cannot identify the exact failed field. The separately prepared Mac
manifest/input files were not supplied. Do not label a particular field or a
cross-platform random-number difference as the confirmed cause.

Inspection nevertheless identified an unnecessary portability constraint: the
v1 caller required Mac-prepared random input hashes to match the Linux inputs
used for the committed loop references. Same-seed random tensors do not promise
identical bytes across platforms/package builds. Shape-compatible native assets
can accept the committed test inputs without requiring this hash equality.

Fixture schema v2 includes the **exact original three input tensors**, verified
against the existing fixture input hashes. All sixteen reference binaries are
unchanged. The caller now reads and hashes these inputs from the same fixture
as the references, while retaining model shape/order/checkpoint validation,
reference integrity, precision and numerical bounds. No native model export,
cache deletion, checkpoint download or new random case is needed.

Every sharded/loop manifest value check now names its field and expected/actual
values. This preserves rejection of incompatible/corrupt inputs and makes any
remaining validation failure actionable. The v2 caller still requires an Apple
SDK build and a new device run. See [updated instructions](../../docs/coreai-denoiser-loop.md).
