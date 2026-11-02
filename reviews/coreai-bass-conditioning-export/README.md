# Qwen + complete ACE ConditionEncoder: actual native export

The exporter ran through coreai-torch 0.4.3/coreai-core 1.0.0b3 using complete,
checksum-verified publisher checkpoints, PyTorch 2.8.0+cpu and Transformers
4.57.3. No random replacement embeddings, fake converter or emulated CoreAI
inference were used. The [manifest](reference.json) records all 27 native asset
hashes, fixture hashes, prompt, tensor contracts and numerical results. Large
assets and tensors remain local and ignored.

Original Qwen3Model executes token lookup and all 28 blocks on the bass prompt.
Lyric embeddings use its table directly, as the handler does. The original
AceStepConditionEncoder executes all eight lyric and four timbre blocks, text
projection, timbre unpacking and both sequence-packing operations. These full
implementations provide independent FP32 gold references.

The partitioned FP32 path matches all three gold outputs **bit-exactly** on
host. FP16-stored/FP32-compute partitions pass unchanged numerical bounds:

| Output | Shape | Max absolute error vs full FP32 | Normalized RMSE |
| --- | --- | --- | --- |
| Qwen text embedding | `[1,105,1024]` | 6.67572021484375e-5 | 1.6820213854899026e-6 |
| Packed condition | `[1,117,2048]` | 6.365776062011719e-5 | 1.678264633995912e-6 |
| Condition mask | `[1,117]` | 0 | 0 |

Native assets total **2,417,955,338 bytes** across 27 stages: token embedding,
14 Qwen pairs, Qwen norm, lyric front/four pairs/norm, timbre front/two pairs/norm,
and final projection/packing. Each neural operation is exported; host gold
embeddings are not device inference inputs. Analytical RoPE buffers are recreated
on CPU after meta initialization; learned tensors load strictly from checkpoints.

This is the fixed bass preset with host tokenization, empty lyrics, language
unknown, one unpadded example and publisher silence as the no-timbre reference.
All 750 timbre frames are retained. Fixed valid-token packing preserves the
original lyric/timbre/text order and is compared against original dynamic packing.
Qwen remains causal; ACE encoders use their alternating local/full masks.

The device caller validates stage order, hashes three inputs and six references,
executes the assets with detached CPU copies, and compares Qwen embedding,
condition and mask against both references. The subsequent
[user-supplied iPhone report](../coreai-iphone17pro-bass-conditioning/README.md)
confirms successful Apple build/execution and all six comparisons PASS across
27 stages. Total probe time is 9.463 s; sampled app-process maximum 783.99 MiB,
not exact peak or total model/device RAM. No denoiser, CFG or bass waveform is executed or
generated. Host tokenization does not establish arbitrary prompt entry on iPhone.

Fifty related host tests pass, including corruption, checkpoint/stage-order
contract rejection, invalid/rehashed token/mask/nonfinite data, existing-output
preservation and previous component/loop/setup checks. Integrity-only native
fixtures are explicitly non-executable and never used as inference evidence.

See [Mac/device instructions](../../docs/coreai-bass-conditioning.md).
