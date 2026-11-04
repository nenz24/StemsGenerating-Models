# Full generation completes faster; minor-glitch investigation waived by user

User-supplied report and three Float32 WAVs, run
`972785C4-C9AC-4662-8565-219374CA2AB6`, started **4 November 2026, 22:15:58 WIB**.
`generation-bounded-v1`, Release/-O, iOS 27.0.1 (24A446), h18p; cpuOnly requested,
actual placement unmeasured. User hears glitches in **both bass solo and mix
inside the app**, then states the glitches are minor and explicitly asks to
ignore them. Investigation is stopped at the user’s request; it does not block
continued work toward drum/guitar previews. No artifact-free claim is made.

## Audio evidence

All three uploaded WAVs match both this report and the recorded hashes of the
previously listening-approved [10.24-second baseline](../coreai-iphone17pro-audio-import-1024/README.md):

| File | SHA-256 |
| --- | --- |
| Bass | `8f423449cb287daeaa053e77272184fcf501b61b53e633321545d97b108f2315` |
| Piano | `cc19086fe37b2c0dd27a85a8111daf70e2841ebe215ef8e9b8b722d5ccffecdc` |
| Mix | `cb6209981c6c02521bf9ae22e9b3071ad8dfb10a0956e77f039553d93a8bb271` |

The faster run did not produce changed audio bytes relative to those baseline
identities. This does **not** dismiss the listening complaint: artifacts could
already be present in that waveform, or playback/output could expose them.
Historical approval is not proof of glitch-free audio and does not replace
the current user feedback and subsequent choice to continue.

All WAVs are finite stereo Float32, 48 kHz, 491,520 frames / 10.24 s. Bass peak
1.195408 (+1.550 dBFS), 1,307 channel samples above unity; the Float32 file
preserves them without integer clipping. Such values can clip on playback, but
sample peaks do not prove audible clipping. Mix peak is 0.972835 (−0.239 dBFS),
with zero above-unity samples. An 8x resampling estimate is recorded separately;
it is not a playback measurement or a glitch detector. Raw-bass headroom alone
does not explain a complaint affecting the mix as well.

For the first time the supplied piano WAV permits independent verification of
the imported planar PCM hash and the listening mix. Its planar Float32 SHA-256
matches the reported input; `Float32(0.78)*piano + Float32(0.55)*bass`, gain 1,
reconstructs **every mix sample bit-exactly**. No unexplained mixing/file
corruption is found by these checks. Finite/hash-consistent audio can still
contain audible musical artifacts. No listening claim is made from signal
metrics or the largest adjacent-sample delta.

## Runtime and numerical status

50/50 steps, 750 sampling calls plus one decoder; sampling 30 opens/30 binds/
720 hits, repair 0/1. Pipeline **96.438 s** versus prior 772.415 s, an observed
8.009x ratio between runs, not a controlled causal benchmark or full-song
throughput claim. Steps 3–50 average **1.058 s**; all 50 after-step thermal
states nominal. Sampled maximum process 791.784 MiB, not exact peak/total RAM.

Conditioning six checks PASS. Imported-output parity remains **NOT EVALUATED
(0/0)**; host full numerical gate remains NOT PASSED. No tolerances/references,
weights or baseline audio were modified. Performance improvement and audio
quality are separate outcomes.

## User decision and next scope

The external-player/timestamp comparison was proposed but not completed.
The user instead says the glitches are minor and asks to ignore them. Stop
this investigation; do not keep requesting listening tests or insert a
limiter/filter/sampler change for it. Preserve the raw WAVs and existing bass
baseline. Continue toward instrument previews using the measured runtime,
without extrapolating to full-song latency or granting numerical PASS.

The [report](report.txt), [summary](summary.json) and [audio audit](audio-audit.json)
retain the evidence. Exact step ordering, asset call counts, phase totals and
per-step deltas were checked; uploaded files were independently hashed and
decoded. WAV binaries are retained locally, not committed. No new generation
or native replay occurred in Linux; original uploads remain unchanged.
