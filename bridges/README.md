# ACE-Step context bridge

[Notebook 11](../notebooks/11_colab_context_chunks.ipynb) includes the two Python
helpers and verifies their SHA-256 checksums before loading them. Upload only the
original piano audio. Use the existing Colab CUDA VM with the bridge checkout,
model runtime and weights from notebook 10. This notebook does not install models.

[ace_context_tools.zip](ace_context_tools.zip) is an optional downloadable copy:

- `ace_context_chunks.py`: context planning, generation orchestration and stitching.
- `electric_full_song.py`: the existing single-pass CUDA FP32 worker.
- `11_colab_context_chunks.ipynb`: the notebook with embedded helpers.
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
