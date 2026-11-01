# Native CoreAI model: first successful iPhone inference

The user's [full report](report.txt) confirms the native `.aimodel` validates,
loads and runs function `main` on the selected iPhone, iOS 27.0.1 (24A446),
CoreAI architecture `h18p`. All three outputs exactly match the four-value
PyTorch reference, maximum absolute error 0. Structured evidence: [report.json](report.json).

| Measurement | Reported time |
| --- | ---: |
| Load/specialization | 299.960625 ms |
| First inference in this probe | 117.172042 ms |
| Warm inference 1 | 0.817167 ms |
| Warm inference 2 | 0.722708 ms |
| Mean of those two warm runs | 0.7699375 ms |

These inference timings include output extraction/read-back and comparison.
Model cache state was not reported, so do not call the load time a verified
cold-cache measurement. Two warm timings are not a latency distribution or a
throughput benchmark. The requested policy is CPU-only; hardware placement was
not independently measured. This provides no GPU/Neural Engine execution result.

The expected project commit is `94e3556`; neither the actual source commit,
device asset hash nor precise hardware model identifier was emitted by the app.
The user had previously selected an iPhone 17 Pro. Retain that provenance rather
than treating architecture/family as a separate hardware-identification test.

The native export -> valid asset -> device specialization -> inference -> value
read-back path is established for this model and input. ACE-Step operator
coverage, actual audio generation, musical quality, peak RAM, thermals, energy
and multi-minute operation remain unverified. This result is not a reason to
estimate audio inference time by multiplying the tiny model's sub-ms warm time.

Next: inspect/export a representative audio component, verify numerical parity
with Python, and measure its device memory and latency. Then attempt one short
instrument generation before three instruments or a full song. The accepted
CUDA/GarageBand audio stays the musical quality reference.
