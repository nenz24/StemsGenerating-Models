# First CoreAI device probe

The user reports running the CoreAIProbe app on the previously selected iPhone
17 Pro. Preserve their output in [report.txt](report.txt); structured facts and
evidence scope are in [report.json](report.json).

Observed output establishes a successful Float32 NDArray construction/read-back
on an iPhone device build, OS 27.0.1 (24A446), CoreAI architecture string `h18p`.
The compute-availability query reports CPU, GPU and Neural Engine. Those are
reported available kinds, not measured execution placement for a model.

The expected project source is commit `6967da1`. The report itself does not
contain a source commit or hardware model identifier; retain that distinction
instead of treating the family string or architecture as a model-ID measurement.
The run was performed by the user, not reproduced in the Linux workspace.

No model was loaded or run, and no audio was generated. RAM, inference latency,
thermal behaviour and numerical parity with a converted model are unmeasured.
The next gate is a documented export path producing a valid local CoreAI asset.
First validate a tiny numerical model against reference input/output, then try
an audio component and one short instrument generation before full arrangements.
