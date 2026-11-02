# Real-recording source-context: actual native export

The uploaded `piano.mp3` was decoded/resampled on host to the first 5.12 seconds
of 48 kHz stereo Float32 PCM, without peak normalization. The same PCM is stored
in the ignored local model bundle for device comparison. User audio, derived
activation binaries and native weights are not committed.

All VAE encoder tensors load strictly from the checksum-verified publisher
checkpoint. Diffusers Oobleck posterior `.sample(generator=seed31)` is exactly
equal to explicit `mean + (softplus(scale)+1e-4)*epsilon` on host. The pinned
runtime's actual `ConditioningMaskMixin` executes the full-range lego case:
`is_covers=False`, source unchanged, boolean full mask expanded across channels.
Source context concatenates sampled source and the expanded mask. Weight-norm
folding preserves all four outputs bit-exactly before export.

The exporter ran successfully through **coreai-torch 0.4.3 / coreai-core 1.0.0b3**
using the real checkpoint, not a fake converter. Native asset size is
337145559 bytes; complete local bundle 339344144 bytes for this export.
Encoder parameter count after folding is 84264832. The [reference manifest](reference.json)
records hashes, shapes, versions and scope; it contains no audio/latent payload.
Mac-generated metadata/bytes may differ and must be verified against that local
bundle, not assumed byte-identical to the Linux asset.

Source audit is pinned to runtime commit
`ca1e85fe9430179831e6bc6be790c332190a3866`, with six source-file identities in
`configs/coreai-lego-preprocessing.json`. Relevant facts are actual posterior
sampling, minimum 128 latent frames, preserved lego source and full mask. This
does not establish complete upstream MP3/resampler/RNG equivalence or the text
conditioning path. Canonical PCM is explicitly shared for the numerical gate.

Forty-six related host tests pass, including actual Diffusers distribution math,
source/mask channel preservation, PCM channel order, short-input rejection,
contract/corrupt/nonfinite-data rejection, existing-output preservation and the
prior codec/DiT/loop/setup regression checks. Integrity fixture native files in
unit tests are marked non-executable and never used as inference evidence.

The source-context native graph accepts PCM, explicit epsilon and chunk mask;
it emits mean, std, sampled source and context. The iPhone caller uses one
retained model/function and checks all four outputs for three calls. The later
[user-supplied iPhone report](../coreai-iphone17pro-source-context/README.md)
confirms build/execution and all twelve comparisons PASS. Mean backend-await
is 1.037 s, sampled app-process maximum 398.21 MiB; these are not isolated
kernel time, exact peak or total model/device RAM.
There is no Qwen text embedding, ConditionEncoder, denoiser/CFG loop or generated
bass in this gate. See [Mac/device instructions](../../docs/coreai-source-context.md).
