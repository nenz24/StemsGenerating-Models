# Window decoder passes on iPhone; frozen30s live generation prepared

[Actual device report](window-device-report.txt), run `54FC8924`, completes 6/6
window decoder calls with 1 open / 1 bind / 5 reuse hits. All 12 comparisons pass original
abs 1e-4 / rel 1e-3 / nRMSE 1e-3: six retained cores, five seam neighborhoods and full 30s
assembled audio. Full waveform max abs `5.453825e-5`, nRMSE `2.1635352e-6`, zero
pointwise violations among 2,880,000 values. [Structured audit](window-device-audit.json).

Sampled process maximum 775.253 MiB is below the existing 2000 MiB budget. Total
test 15,370.754 ms; thermal nominal at start/end, low-power off, no cache deletion
or retry. Earlier standalone 750 peak 2696.770 MiB came from a separate run with
different thermal/cache conditions. These measurements are not a controlled
timing comparison or a combined sampler/decoder residency proof. Matching
aggregate error statistics do not establish byte-identical native output.

This accepts the standalone fixed-input bounded decoder's numerical/resource
gate. Native full 50-step feedback, combined lifecycle, device listening and
full-song bass/drums/guitar remain **NOT EVALUATED**.

## Next action: one independent 50-step native trajectory

[`activate_coreai_drum_reference30_generation.py`](../../scripts/activate_coreai_drum_reference30_generation.py)
verifies the existing canonical sampler export and prepared bounded decoder,
binds their exact manifest identities in a new configuration and backs up the
previous selector before selecting `drums-reference30-generation`. There is no
checkpoint download, host inference, new export or baseline replacement.
Different existing configurations are preserved and rejected rather than
overwritten. [Fixed contract](../../scripts/coreai_drum_reference30_generation_contract.py).

The build includes 65 files: 15 BF16-storage sampler/APG assets, the one FP32
bounded decoder, exact frozen source/context/condition/noise/timesteps, five
gold checkpoint latents, original raw waveform and configuration/provenance
JSON. The original full 750 decoder and source/Qwen/guitar/bass model bundles
are excluded from generated .app only. Existing 128/256 generation contracts and
all original source/model folders remain unchanged on Mac.

The [current Mac command](../../docs/coreai-drums-quality.md) enables
**Generate drum 30 detik (acuan)**. The caller:

1. Rehashes all staged model/input/checkpoint data and validates exact source,
   configuration and prepared decoder identities before opening a model.
2. Runs the actual first step with serial model/function scopes, then retains
   at most 15 sampling handles from step 2. Fifty APG/Euler updates use live
   native velocity, latent and momentum. Saved gold is read only for comparison
   at 1/5/10/25/50, never fed back into the trajectory. No source RNG or condition
   is regenerated; frozen original FP32 input tensors are used directly.
3. Releases the sampler handle pool before loading the bounded decoder. Six
   decoder inputs are sliced from the **live native final latent**, not the
   component test's captured gold final latent or gold tile hashes. Recording
   boundaries/retained-core geometry remain the verified window256/core128/
   halo64 plan, without fade or output filtering.
4. Compares five checkpoint latents and final raw waveform to the host reference
   at unchanged original gates. This is six comparisons on an independent live
   trajectory, not every unsaved intermediate state. Finite strict failures
   remain NOT PASSED while later checkpoints are collected. Invalid/native/
   nonfinite/current-memory guard errors stop the run; no state is reset.

Expected normal accounting: 750 sampling calls, 30 opens / 30 binds / 720 hits, then six
decoder calls with 1 open / 1 bind / 5 hits (756 calls total). Existing narrowly targeted
initial-bind BNNS repair is limited to one successful sampling-stage repair;
there is no inference retry, broad cache deletion or decoder repair. Repair
events remain explicit. Pre-call/post-bind 2000 MiB guards and sampled combined
peak use the existing budget; sampler/window decoder component peaks do not
substitute for this combined measurement.

## Output and acceptance

The caller saves native checkpoint/final latent/momentum and raw drum audio in
a new private per-run Documents folder. It produces diagnostic Float32 stereo
WAVs for drums, accepted prepared piano and 0.5 piano + 0.5 drums. All three share the
reference listening attenuation, lowered together only if needed for headroom.
Raw numerical checks precede attenuation. Source gain 0.7 is not applied again;
production mix 0.78/0.55 is unchanged.

WAVs remain diagnostic until numerical/resource checks and listening feedback
are reviewed. A completed finite trajectory may produce listening files while
strict failures remain explicitly NOT PASSED. Playback/share controls do not
mark such output accepted. Resource readiness separately requires sampled peak
below 2000 MiB; sampling is not exact transient/native memory measurement.

Live UI progress updates once per sampler step and decoder window. Small native
breadcrumbs persist before API calls; full reports are saved at checkpoints,
every five steps and on completion/failure. Each run keeps its own report and
previous baseline/component reports are untouched. A restarted app restores
the latest matching run, breadcrumbs and fully completed audio artifacts.

[Validation](validation.json): 227 Python tests, 225 PASS, two optional skips; seven
new configuration/activation/65-file packaging tests. Transport files are
explicitly NONEXECUTABLE and synthetic; no new learned-model output is claimed
by them. [Swift grammar audit](swift-syntax-audit.json) covers the three edited
Swift files. No Apple SDK typecheck/build/device generation was run here; the
next evidence is the user's actual 50-step iPhone report and listening verdict.
No private audio, tensor, checkpoint, ZIP or native binary is committed.
