# Confirmed Release source CPU retained preview

The unchanged [device report](report.txt), summarized in [summary.json](summary.json),
identifies run `41DEC6A2-87C5-4E53-BD5D-59044308A627`, started **2 November 2026,
19:03:45 WIB**. The explicit plist now records **Release**, **`-O`**, and the
independent DEBUG compilation condition is **OFF**. Reuse is requested and the
counter contract is observed: **15 opens, 15 binds, 735 Swift handle hits,
750 completed sampling calls**, plus the decoder. Sampling pool release is
reported before decoder execution.

| Observation | First single-scope report | Profiled single-scope report | Confirmed Release/reuse |
| --- | ---: | ---: | ---: |
| Pipeline time | 85.95 s | 755.29 s | **59.24 s** |
| Audio duration | 5.12 s | 5.12 s | 5.12 s |
| Sampled app-process maximum | 417.19 MiB | 436.47 MiB | **450.42 MiB** |
| Recorded build flags | unavailable | unrecorded | Release / -O / DEBUG OFF |

Both new WAV hashes **exactly match the listener-approved baseline**. No new
listening verdict is fabricated or required to establish that bit equality.
Eight of ten output comparisons pass; final-step latent checks still fail and
host status remains `bounded_diagnostic`. Both decoded waveform comparisons to
Mac references pass. Gate values/tolerances and audio have not changed.

## Performance interpretation

The observed wall-time ratio is 12.7507 versus the slower single-scope run,
and 1.4511 versus the original 85.95-second run. These are separate observations,
not a controlled cache/thermal experiment or proof of an isolated Release-only
speedup. Native residency, cold start and repeatability are unmeasured.

Total backend-await falls from 730.38 s in the previous report to **37.14 s**;
output copy is **7.71 s**. The initial observed backend calls total about
16.56 s; repeated calls total about 20.58 s. Step 1 takes 15.07 s and steps 2–50
average **651.59 ms**. These API intervals may contain deferred work and are not
isolated kernel time. Reusing both model and function clearly takes the observed
software path; the report does not identify the exact internal compiler/kernel
cause of the previous regression.

The app-process maximum reported by the 10 ms sampler is **450.4243392944336 MiB**,
not total RAM or an exact peak. This is one fixed-shape batch-two case; it does
not prove a cold-start memory envelope or safe scaling to 90 seconds/full song.
The 2000 MiB before-stage/post-bind guards remain; native transient peaks and
native eviction remain outside their guarantee.

## Prototype route selection

Retained source CPU generation is selected as the **short prototype runtime
baseline**. Standalone launches without an override now default to reuse for
source assets, so reopening the installed Release app need not return to the
slow one-scope lifecycle. Explicit AOT resources still default to one-scope
AOT behavior; this result does not validate AOT.

Xcode schemes remain explicit: **CoreAIProbe Reuse** is Release/reuse=1;
**CoreAIProbe Performance** is Release/reuse=0; **CoreAIProbe** is Debug/reuse=0.
The controls remain available, and overrides last for the launched process.
The new default-selection branch needs Apple build verification, while the
actual retained model/function execution and plist metadata are now verified
by the supplied device report. Model math/assets/precision/inputs/strict bounds
are unchanged. No additional generation is needed merely to accept/store this
existing run. Raw recording import, longer context and other instrument roles
remain separate work; this is not real-time or production feasibility evidence.
