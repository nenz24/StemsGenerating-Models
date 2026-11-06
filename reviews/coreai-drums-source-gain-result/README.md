# Source gain0.7 did not resolve hit-transition glitches

User [report](report.json) is preserved byte-for-byte,8357bytes. Listening:
**"glicthnya ada lagi dipergantian pukulan"**. Per-file judgments were not
separately specified; do not fabricate a clean candidate or a new groove verdict.
The source-gain-only candidate is not adopted. Do not repeat or sweep gain/cap.

[Validation summary](summary.json):50/50 candidate steps, two source encoder
calls, nine replay checks with all reported errors exactly0. Conditioning is
byte-identical; prompts, source PCM and three component reference identities
match the previous experiments. Original control raw-WAV reported hash matches
the previous control. Candidate source PCM reported SHA `316ee2f7…d10d9e75`
is independently reproduced by Float32 multiplication of the earlier verified
uploaded PCM by0.7. Thus the recorded input change is internally consistent;
this does not independently verify new output waveform/latent bytes, which
were not uploaded. Source peaks0.6691904→0.46843329 match that change.

Original source mean/std/latent/context, conditioning/mask/unchanged-condition,
first control Euler step and control decoder replay all pass original bounds.
The existing `bounded_diagnostic` provenance remains; these checks do not prove
all50 independent-trajectory outputs or any native execution match a reference.
Reported raw peaks: control1.32284009,candidate1.36214399; shared gain0.51972904
matches−3dBFS policy. Headroom/control lineage is metadata validation, not proof
of the audible artifact's source. No waveform clipping/splicing diagnosis is
made from this report alone.

The earlier **clean percussion VAE control** still applies to that existing
Colab drum reconstruction. It does not certify arbitrary generated latents.
Gain-only and lyric-only changes have not resolved the complaint; scaled APG
was partial improvement. Keep FP32 VAE/BF16 production/cap2.5/groove requirement,
all checkpoint/cache/original audio and abs1e-4/rel1e-3/nRMSE1e-3 limits.

Next, one [captured30s reference reproduction](../coreai-drums-reference30-plan/README.md)
matches the accepted input preparation/metadata/lyrics/context settings together.
It is not a duration-only causal test and is not an arbitrary duration sweep.
Actual inputs and intermediate outputs are captured instead of inferring
CPU/CUDA RNG identity from seed numbers. No repaired audio, root cause or new
iPhone/full-song acceptance is established by the current report.
