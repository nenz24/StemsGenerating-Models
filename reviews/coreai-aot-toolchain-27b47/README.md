# AOT toolchain available: Metal 27B47

The user reported successful Metal Toolchain installation and supplied
`xcrun coreai-build compile --help`. The [transcribed help](compile-help.txt)
records supported arguments, not a compiler invocation or model result.

The CLI accepts `--architecture h18p`, `--platform iOS`, deployment 27.0,
`--output`, and `--preferred-compute` values gpu/neural-engine/none. There is no
CPU-only option. Preparation uses **none** and runtime AOT loading uses
**default**, following Apple's guide to match compile/load compute policy.
Source/Qwen and non-AOT sampling stay CPU-only. Actual placement is unmeasured.

The implementation preserves original bundles and compiles only the sixteen
preview assets for one architecture. Per-asset receipts bind results to source,
compiler executable/help identity and architecture; corrupt/stale results stop
instead of being silently reused or overwritten. The complete package has
file hashes and source-reference identity; it is published only after validation.

Six new tests check exact CLI arguments, mixed file/directory packaging,
corruption/architecture/policy/source rejection, resume receipt integrity,
preservation, standalone copies of linked compiler payloads and shell syntax. Fixtures are labeled non-executable; no fake
native compiler or inference is used. Native Mac AOT compilation and the new
Apple/device loader are **not yet validated** in this Linux workspace.

The loader chooses matching compiled variants and verifies their hashes before
inference. It keeps one live Swift stage scope and preserves existing numerical
references/tolerances/diagnostic status. No implicit CPU source fallback is
performed if an AOT bundle exists but fails validation.

See [instructions and remaining specialization limits](../../docs/coreai-bass-preview-aot.md).
