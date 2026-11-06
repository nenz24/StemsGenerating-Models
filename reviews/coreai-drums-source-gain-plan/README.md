# One source-gain-only generation candidate after clean codec controls

**Subsequent result:** the [Mac report](../coreai-drums-source-gain-result/README.md)
completes50/50 with nine zero-error control checks, but the user still hears
hit-transition glitches. This gain-only experiment has not fixed the complaint;
do not repeat it, sweep gains or adopt the candidate. The [captured30s host
reference](../coreai-drums-reference30-plan/README.md) is now the active task.

The latest user asks what to do next. Piano and existing approved-percussion
roundtrip controls have been heard clean. Existing generated drums remain
unaccepted: scaled APG made them clearer but still had hit-transition glitches.
The accepted Colab preparation applies fixed source gain0.7 before source
encoding; the current preview preserves unnormalized PCM. This is a real input
difference, not yet a proven glitch cause.

[`compare_coreai_drum_source_gain.py`](../../scripts/compare_coreai_drum_source_gain.py)
isolates **PCM multiplication by0.7 before VAE source encoding**. It reuses the
verified stored10.24s PCM, exact saved source epsilon, full-range lego mask,
text/lyric/timbre conditioning, diffusion noise/schedule, original APG cap2.5,
guidance7,50 Euler steps, weights and decoder. It computes one independent
original-FP32 candidate trajectory against cached original-FP32 control audio.
This is not the cap1.46059 candidate, not playback attenuation, not a combination
of lyric/cap edits, and not a reproduction of Colab's32kHz PCM16/30s pipeline.
Longer-context reference reproduction stays a separate plan: isolate a concrete
known input difference first rather than simultaneously change preparation,
metadata/lyrics and duration. No new seed/cap/duration sweep is added.

## Controls and isolation

Source VAE executes twice: first on unchanged PCM, checking its mean/std/source
latent/context against all four existing gold tensors at the original abs1e-4,
rel1e-3,normalizedRMSE1e-3 limits; then on the scaled PCM with the **same saved
epsilon and mask**. Candidate source PCM, posterior mean/std, latent and context
are saved independently. The original mask must remain identical.

The existing diagnostic host executor accepts an optional candidate context.
Its default path stays unchanged for lyric/APG experiments. A source-bound CFG
predictor keeps the original control context for the first-step replay and
selects candidate context only for the candidate's live50-step trajectory.
Original conditioning/mask, unchanged candidate conditioning, first gold step
and cached-control decoder remain gated. Candidate conditioning must be
byte-identical to the control. Thus nine replay checks precede/complement
execution; no failed check is reclassified or used to replace a reference.

Existing `bounded_diagnostic` provenance remains. A successful replay does not
establish full independent-trajectory or native parity. Weights resolve from
local caches only with full pinned checksums; no download fallback. A new output
directory must be outside ModelCandidates. Missing weights/gate failures retain
a failure report. Prepared source/model files, selector, BF16 baseline, caches,
production sampler and all numerical tolerances are preserved.

## Verification actually performed here

[Encoder smoke report](encoder-audit.json) records two real CPU FP32 calls on
the uploaded piano PCM with the pinned VAE and the same explicit CPU epsilon
fixture. Candidate input peak becomes0.46843329, source context changes, mask
stays identical and output is finite. **This is not a Mac source-gold replay**:
fresh CPU seed31 epsilon hash `b2b3a04c…b8d1c` differs from the Mac's stored
`98f0be58…40d2e2`. Hash verification caught that mismatch before any initial
parity claim. Original epsilon/mean/std arrays were not uploaded. The smoke
therefore labels comparison metrics as descriptive and unmatched, even if a
numerical metric happens to fall within bounds. Do not assume a common seed
guarantees identical bytes or infer why the hashes differ.

The executable Mac experiment reads actual saved epsilon and gold arrays from
the verified existing bundle, never regenerating them from the seed. This
avoids substituting a convenient new reference. Base/Qwen weights are not in
this cloud workspace; no full50-step candidate was run here, no new music was
heard, and no repair or musical acceptance is claimed. [encoder_audit.py](encoder_audit.py)
is a reproduction source for the explicitly bounded cloud smoke only.

[Validation](validation.json):169 host tests run,167 pass,two optional skips.
Six new tests cover read-only planning and provenance, admission/output guards,
source-gate failure retention, candidate-context-only routing/incomplete-loop
failure, independent CFG context binding and gain-before-encoder with shared
saved epsilon/mask. The analytic encoder fixture checks control flow and
closed-form posterior math, not musical model validity. Syntax/JSON/links and
the explicit diagnostics-only change list are checked. No Apple SDK/device
execution is claimed. Audio, tensors, checkpoint and user ZIPs are not committed.

## User action

Use the [Mac command](../../docs/coreai-drums-quality.md). This is one CPU
generation with existing weights, not another codec-only listening request.
Listen to the two headroom drum WAVs at the same volume/output, and return the
report plus both artifact/glitch and groove feedback. Do not rebuild the iPhone
or replace the BF16 baseline yet. Source gain is fixed at0.7 with no CLI sweep.
