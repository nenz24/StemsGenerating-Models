# Instrument preview host-contract audit (not device acceptance)

The pinned publisher formatter and 11,423,705-byte Qwen tokenizer were actually
loaded, checksum verified and exercised using Transformers 4.57.3. No learned
checkpoint or full native model export was performed in this Linux checkout.

[Tokenization](tokenization.json) records exact prompts, preset/tokenizer hashes
and token-ID digests: bass/strumming 106 text tokens, drums/picking 111, empty
lyrics 11; condition sizes 118/123. These are all-valid single-example sequences,
without fabricated padding or caption edits. The Swift allowlist pins the same
preset hashes, instruction and geometry independently.

[Reuse plan](reuse-plan.json) compares actual stage contracts. Drum from bass
requires 17 new conditioning and 13 new sampler assets, reusing 10+3. Strumming
from bass and picking from a ready drum bundle can reuse all 27+16 native assets.
Each target still computes its own original-FP32 conditioning and trajectory
references; copied native assets are not a new numerical/device PASS.

135 host tests pass, two optional tool-dependent tests skip (CoreML and native
export-tool fixtures). Eight new tests cover profile isolation, exact native
reuse rejection, atomic selector preservation on failure, diagnostic-flag
forwarding, activation-only and pinned presets. Existing BF16/APG/schedule,
corruption and trajectory admission tests remain unchanged. These tests use
nonexecutable integrity fixtures where native payloads are required.

Source generation bundles, checkpoint, caches and accepted bass WAVs are not
overwritten. Only one selected profile is staged; new output names/restoration
are instrument-specific. Apple compilation, full exports and iPhone results
for the new presets are pending. Guitar roles currently use silence timbre;
the frozen Colab style reference and on-device layer mixing are not ported.
See the [run guide](../../docs/coreai-instrument-previews.md).
