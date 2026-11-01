# ACE-Step context bridge

[Notebook 15](../notebooks/15_colab_drum_context_90.ipynb) generates drum in bounded
90-second contexts and stitches them using the bass method, while keeping the
drum preset unchanged. It reuses the active runtime/weights. Its feedback includes
pre-stitch chunk previews and actual reports; GPU/music quality is not yet tested.

[Notebook 14](../notebooks/14_colab_drum_full_song.ipynb) tests the full drum song
in one continuous context using the active runtime/weights, without stitching.
The uploaded full-song result was rejected for unclear groove/timbre/timing.
It is retained for opt-in reproduction with DRY_RUN=True by default, not as a
quality-approved full-song workflow. The approved 30-second baseline is preserved.

[Notebook 13](../notebooks/13_colab_drum_probe.ipynb) starts a separate 30-second
drum probe on the active VM, reusing existing runtime/weights. Bass reference
audio and known defects are saved in `baselines/electric-bass-context90/`.

[Notebook 12](../notebooks/12_colab_fresh_context_90.ipynb) is for a new Colab
account/VM. It embeds the runtime bridge source, checks GPU/disk, installs the
pinned runtime and weights, and generates the full song with 90-second contexts.
Select a GPU and upload the original piano only. No old source ZIP, old VM or
GitHub login inside Colab is required. GPU access must be available on that
account. A new VM downloads the runtime/weights again (about 45–50 GiB free disk).
The previous duration probes are not repeated. Save WAV and feedback ZIPs before
the session ends; live processes/files do not transfer automatically between VMs.

[Notebook 11](../notebooks/11_colab_context_chunks.ipynb) includes the two Python
helpers and verifies their SHA-256 checksums before loading them. Upload only the
original piano audio. Use the existing Colab CUDA VM with the bridge checkout,
model runtime and weights from notebook 10. This notebook does not install models.

[ace_context_tools.zip](ace_context_tools.zip) is an optional downloadable copy:

- `ace_context_chunks.py`: context planning, generation orchestration and stitching.
- `electric_full_song.py`: the existing single-pass CUDA FP32 worker.
- `11_colab_context_chunks.ipynb`: the notebook with embedded helpers.
- `12_colab_fresh_context_90.ipynb`: fresh-VM notebook with embedded bridge source.
- `electric-context.json`: the selected 90-second cap, margins and fade settings.
- `ace-context-chunks.md`: usage, evidence and limitations.
- `requirements-context-chunks.txt`: orchestration dependencies, separate from the
  locked model runtime.

The ZIP contains no source audio, generated audio, model weights or credentials.
Use GitHub's **Download raw file** button on the ZIP page if needed. The notebook
works without downloading this ZIP.

After changing the helper scripts, notebook, configuration or included docs, run:

```bash
python scripts/build_context_bridge.py
python scripts/build_context_bridge.py --check
```

The builder embeds the current helpers, clears no user outputs, refuses executed
notebooks, and creates the ZIP with deterministic timestamps. Commit both the
source changes and refreshed notebook/ZIP.
