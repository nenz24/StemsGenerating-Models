# Export saved frozen30 iPhone data without another generation

Use this only for a completed `reference30-live50-window256-v1` run. The app
already saves five native checkpoints, final latent/momentum and raw drum audio
in its private Documents. This helper copies those existing files; it performs
no inference and never changes model assets, active profile, BF16 or tolerances.

1. Connect the iPhone to Xcode. Open **Window > Devices and Simulators**.
2. Select the device, then CoreAIProbe in **Installed Apps**. In its actions
   menu choose **Download Container…** and save the `.xcappdata` on the Mac.
   Preserve the app and downloaded container.
3. Update the repository using the existing project update helper. From the
   repository folder, run the following command, replacing both placeholders:

```bash
python3 scripts/package_coreai_drum_reference30_run.py \
  --source "/path/to/CoreAIProbe.xcappdata" \
  --run-id UUID-FROM-YOUR-DEVICE-REPORT \
  --out .runtime-cache/drums-native30-live-review-v1.zip &&
open -R .runtime-cache/drums-native30-live-review-v1.zip
```

You can drag the downloaded container into Terminal to enter its path. The
UUID is the `run` identifier on the first line of the device report. The helper
also accepts the exact run folder or its Documents directory as `--source`.

The ZIP contains the unchanged report, eight tensors and a transport manifest.
Revisionv2 additionally requires all50 actual native velocity tensors and a
device report-bound `velocity-trace.json`; the ZIP then has58 tensors and three
metadata files. See the [velocity trace guide](coreai-drum-velocity-trace.md).
Model weights, unrelated Documents and listening WAV copies are excluded. The
raw audio included in the ZIP is private user-derived audio; share the archive
privately for diagnosis and never commit it or its contents to GitHub.

The helper checks the completed-run identity, tensor sizes/finite values, final
latent hash against the original report and equality with checkpoint50. It
hashes the exact archived bytes and retains the original PASS/NOT PASSED status.
Hashes for tensors not independently identified in the original report describe
the exported bytes only. Missing states are not reconstructed. Existing output
ZIPs are preserved; choose a new filename or use the already-created archive.

An existing failed strict comparison remains failed after export. Listening
feedback and numerical/resource acceptance must be assessed separately. A CPU
decoder replay using the actual native final latent can distinguish effects of
trajectory drift from same-input decoder differences; export alone does not
establish their cause. There is no need to press Generate again for this export.

After the same-input decoder evidence is reviewed, the next diagnostic is the
[CPU sampler replay](coreai-drum-sampler-replay.md). It uses the existing capture
and exported ZIP, without another iPhone run or native model export.
