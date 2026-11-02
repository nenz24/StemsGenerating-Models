# Mac preparation stopped at strict trajectory comparison

The user supplied a terminal log for revision `2497523`. The [structured
excerpt](report.json) retains the four printed comparison checkpoints and the
SHA-256 of the original paste; local paths are not published. This is host
preparation evidence, not iPhone inference. The numerical root cause is not
isolated; it must not be attributed to a particular Apple backend without tests.

Step 50 latent normalized RMSE is **6.788929154513494e-5**, but 0.00439453125
of elements (about 0.44%) exceed the combined pointwise bound. Strict parity is
therefore **NOT PASSED**, even though normalized RMSE is below 1e-3. The old
preparer stopped before decoding/export. The log does not supply all 50 latent
metrics, identical-input APG results, or decoded waveform metrics.

The update keeps every strict tolerance and flag intact. It permits an explicit
`bounded_diagnostic` bundle only when every latent and decoded audio comparison
is finite and normalized RMSE <=1e-3, while identical-input APG still passes the
full strict bound. Larger/nonfinite errors, inconsistent flags, failed APG,
integrity or shape errors still stop preparation. Pointwise failure remains
failure and cannot be relabeled as numerical PASS.

The iPhone caller checks the same host metrics/status. A diagnostic host bundle
forces the combined numerical result to remain NOT PASSED, even if all sampled
device-output comparisons pass. It may produce WAVs for listening review; this
does not accept a numerical or musical baseline. Existing strict-pass bundles
remain reusable without re-exporting or replacing their contents.

Seven preview tests include the actual Mac step-50 metrics in an explicitly
non-executable integrity fixture, verify diagnostic status propagation, reject
false PASS flags and excessive/nonfinite/same-input APG errors. The original
strict-pass Linux bundle also verifies under the updated contract. No native
models were simulated by these tests.

Updated Mac preparation and the new Swift build/device caller are **pending**
user execution; the earlier Linux strict-pass export remains separate evidence.
See the [retry instructions](../../docs/coreai-bass-preview.md).
