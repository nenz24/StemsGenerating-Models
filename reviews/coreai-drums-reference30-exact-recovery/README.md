# v2 has no generated drum; restore the original input byte-for-byte

The user says **"tidak ada file audionya"** and uploads [report](failed-report.json)
plus preparation ZIP. [Archive review](archive-review.json) verifies all four
members and the separate report byte identity. The4280-byte report is preserved
unchanged:0steps,0source encoder calls,0decoder calls. Execution is still at
source preparation, so no generated drum/mix02/03 exists. ZIP `source32.wav`
is the piano input. This is not a playback, iPhone model or cache failure.

## Remaining difference is measured, not waived

[Aggregate comparison](comparison.json): explicit nearest-even reduced PCM
differences from939512(v1) to1785(v2) out of1920000 interleaved samples. Each
remaining difference is1LSB, balanced883/902 by sign. The captured Mac Float32
stream reproduces its uploaded PCM exactly under the new rounding function;
the Float32 stream itself differs from the original CPU stream (maximum
7.45058e-7, relative RMS6.37529e-7). There is no further evidence to uniquely
attribute this to decoder version, resampler implementation or architecture.

The original frozen source checksum remains. As the preceding sensitivity
probe showed, numerically close input does not guarantee matching VAE output.
No numeric-source allowlist, accepted-hash replacement, gate override, EQ,
limiter or new model configuration is introduced.

## Deterministic reconstruction of the existing reference fixture

The original accepted input is available in this workspace and the actual v2
input has now been uploaded. A measured sparse integer delta can reconstruct
the former from the latter without platform decoding/Float32 rounding. It
contains **only increasing sample-index gaps and a±1 sign**,1785entries,3234
compressed bytes. It is not a general audio correction or a production
preprocessor for arbitrary recordings. Output is the original reference data,
not a new reference chosen to pass. The private edit map is supplied **only in
the user chat**, not in Git; [metadata](correction-metadata.json) records hashes,
counts and encoding but no edit positions, amplitude values or payload.

[`recover_coreai_reference_source.py`](../../scripts/recover_coreai_reference_source.py)
accepts the private JSON via stdin/file, verifies the measured local PCM digest,
validates bounded zlib/base64 and strictly increasing unit edits, then requires
both the original PCM SHA `1d57ce14…7baaf23` and full WAV SHA `1f68460c…d66b54`
before creating a new file. Incorrect inputs/payload/targets, overflow, indices,
counts, existing files and baseline/ModelCandidates output paths are rejected.
Neither hashes nor thresholds are configurable overrides. Data is never
executed. It preserves the local input and writes a separate reconstruction
receipt with hashes/counts; only after success can its optional child generation
start. No source audio, correction payload, tensors or weights are committed.

[Actual reconstruction audit](reconstruction-audit.json): the produced WAV is
byte-identical to the original archived reference. The unmodified generator's
strict prepared-source gate passes and cloud source48bytes match the existing
original encoder input exactly. The original uploaded source is unchanged.
Actual CLI runs with both private-file and stdin delivery also pass. This is
integer/file reconstruction, with **zero neural calls**; no Mac/CUDA/full30s
generation or musical acceptance is claimed here. Source32byte identity is
portable because it uses integer edits/hash checks, though downstream numerical
parity still requires matching tensors and measurements as before.

## One executable continuation for the user

The complete private here-document command is returned in chat. After updating
main it reads the existing `drums-reference30-v2/source32.wav`, creates a new
`drums-reference30-source-exact-v1.wav`, and launches the unchanged30s/50-step
host reference with `--prepared-source` into `drums-reference30-v3`. It therefore
does not select/decode MP3 again or apply a second source gain. The Mac opens
the output folder after successful generation. v1/v2 failures and original
audio/cache/models stay preserved. It does not auto-retry an existing run or
change the active iPhone model.

Listen to02host drums and03piano/drum mix, then return the new report and
glitch/groove judgment. This input-fixture recovery alone is not a glitch fix
or full-song three-stem acceptance. Numerical NOT EVALUATED/diagnostic statuses,
BF16 baseline and abs1e-4/rel1e-3/nRMSE1e-3 limits remain unchanged.

[Validation](validation.json) separates actual private-data reconstruction from
six new generic tests for signed integer edits, malformed compression/index/
count/range, immutable target/data checksums, wrong-source/late-target rejection,
protected/existing outputs, and generation gating on recovery failure. No
synthetic musical model is used to claim correctness. No Apple build/device or
full50-step execution was performed here. The private payload is excluded from
the explicit publish list and checked absent from every committed file.
