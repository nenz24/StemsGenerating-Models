# Mac native30s export completed; component-only device test prepared

The uploaded [original manifest](reference.json), SHA256 `f2f84e6c…24d53ed`,
reports all 16 assets exported successfully: 14 partitioned DiT assets plus APG
and full-window decoder. Its complete contract matches the reviewed 750-frame,
127-condition-token preparation plan. All six host gates pass at unchanged
abs 1e-4 / rel 1e-3 / nRMSE 1e-3. [Manifest audit](manifest-audit.json).

FP32 and BF16-storage partitioned step 1 checks each report nRMSE
`7.8567614e-7`, max abs `1.9073486e-5` and zero pointwise violations against
the accepted original FP32 velocities. APG momentum and folded decoder checks
are exact; APG guided/latent checks also pass. Equal aggregate metrics are not
by themselves a learned-parameter identity proof. There was no new 50-step host
generation or change to the source, checkpoints, BF16 baseline, sampler or mix.

All 43 copied capture hashes match the arrays independently verified in the
earlier uploaded [capture ZIP](../coreai-drums-reference30-native-preparation/README.md).
Native model bytes were **not uploaded** here. Their 48 file digests are reported
by the Mac manifest, not independently computed in cloud. Full local bundle
integrity is rechecked before build selection; the iPhone rehashes every native
payload and all 11 selected captured tensors before its first model open.
Native/full-trajectory parity remains `NOT EVALUATED` until device evidence.

## Current device action

[`activate_coreai_drum_reference30.py`](../../scripts/activate_coreai_drum_reference30.py)
uses the existing reviewed export only. It verifies every source artifact,
copies to a separate ModelCandidates namespace, verifies the copy, backs up the
old build selector and atomically selects `drums-reference30-components`.
Existing matching copies are verified/reused. Source exports, baseline assets,
old selectors and failed/interrupted staging directories are preserved. No
checkpoint download, model export or trajectory generation occurs.

The [stdlib contract](../../scripts/coreai_drum_reference30_contract.py) pins the
exact reviewed manifest. The Xcode stager includes 61 files: that manifest,
48 native payload files and 12 captured/reference files. Unused source48,
posterior/text vectors, mid-trajectory captures, source/Qwen encoders and other
baseline/probe bundles stay on Mac. Build pruning is confined to generated .app
resources. Normal 128/256 preview contracts and generation profiles are unchanged.

Use the [current Mac command and button instructions](../../docs/coreai-drums-quality.md).
The new button **Uji komponen drum 30 detik** performs 16 serial model calls and
six original-bound comparisons:

1. One 14-shard prediction using saved noise/context/conditioning/timestep.
   Compare conditional and unconditional velocities to captured step 1 gold.
2. One independent APG call using **captured gold velocities**, saved initial
   latent, zero prior momentum and actual saved dt. Compare next latent,
   momentum and guidance. This isolates APG from DiT rounding differences;
   it is not a native feedback trajectory.
3. One full 30s decoder using saved final latent, transposed explicitly to
   `[1,64,750]`. Compare planar Float32 audio to the saved accepted raw audio.
   No WAV is generated or played and no clipping/level processing is applied.

Finite numerical failures remain **NOT PASSED**, while other independent
components are collected. Invalid hashes/types/shapes, nonfinite values,
native errors or process-memory guard failures stop. Every model/function
Swift scope is released before the next; backend eviction is not guaranteed.
Existing CPU-only requests, pre-call/post-bind 2000 MiB process guards, 10 ms
sampling and the existing single targeted initial-bind BNNS repair policy are
reused. There is no new broad cache deletion, inference retry or precision
exception. Component PASS does not establish full 50-step sampling, pool retention
behavior or native music acceptance.

Reports use separate per-run files and breadcrumbs; baseline generation reports
are not overwritten. A restarted app restores the latest component report and
its matching interrupted native phase. UI progress is bounded; full details
are saved at component boundaries and on success/failure. The button runs once
per process and provides the existing **Bagikan laporan** action.

## Validation and limits

[Validation](validation.json): 207 Python tests, 205 PASS,two optional skips;
seven new packaging/selection tests include failed-copy preservation,
idempotency, missing/truncated inputs, exact payload selection and a build CLI
run with site packages disabled. Asset fixtures are explicitly nonexecutable;
they test transport, not neural inference. [Swift grammar parsing](swift-syntax-audit.json)
passes for all three edited Swift files. This is not Apple SDK typechecking,
Xcode linking, native execution, resource-footprint measurement or device PASS.

No private audio, Float32 vectors, weights, ZIP or native binaries are committed.
Only source/tests/docs and metadata are published. The next evidence is the
device component report; a real independent 50-step trajectory, listening and
full-song bass/drums/guitar remain subsequent stages.
