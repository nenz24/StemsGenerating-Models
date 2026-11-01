# ACE-Step context bridge

[Notebook 21](../notebooks/21_colab_guitar_layers.ipynb) implements the selected
direction: simultaneous strumming and picking in the preferred opening-30s
sound style. Two aligned performances share the same piano and frozen style
reference. It offers preview30 (two calls) and full90 (two whole-song roles,
six calls if uncached for this piano). The style MP3, helpers and compatible
engine/chunk runner are embedded. See the [layering guide](../docs/guitar-layers.md).

The user clarified that the original 90s candidate is useful, liking strumming
in chunk 0 and picking in chunk 2. The failed original/picked-v2 trial is scoped
to its own 30s interval; its uploaded archive is now verified and preserved.
Notebooks 18–20 remain available for reproduction.

[Notebook 20](../notebooks/20_colab_guitar_articulation.ipynb) records the paired
experiment after the listener rejected the 90-second result for noise/strumming.
Two matched 30-second clips at source start 77.130625s compare the original
caption with a picked-note candidate. Same input bytes/model/settings, existing
CUDA cache, no full-song regeneration. The original short tone history is preserved.

[Notebook 19](../notebooks/19_colab_guitar_context_90.ipynb) extends the accepted
30-second clean guitar candidate to the full piano song with contexts capped at
90 seconds, 6-second margins and 0.58-second equal-power fades. Same guitar
preset, existing runtime/weights, resumable chunks, no bass/drum regeneration.
The listener later clarified it is useful but changes sound across chunks.
It is preserved as a candidate, with earlier feedback kept as history; it
defaults to DRY_RUN=True so the existing result need not be regenerated.

[Notebook 18](../notebooks/18_colab_clean_electric_guitar.ipynb) starts the selected
clean electric guitar with one 30-second piano-conditioned probe. It embeds
helpers and a compatible guitar engine for old bridges, reuses the active CUDA
runtime/weights and preserves bass/drum outputs. See the
[guitar guide](../docs/clean-electric-guitar.md). Guitar tone is not yet approved.

The complete notebook-17 output has been listener-approved and frozen in
[`acoustic-drums-shared-kit-full-v1`](../baselines/acoustic-drums-shared-kit-full-v1/).

[Notebook 17](../notebooks/17_colab_drum_shared_kit_song.ipynb) extends the approved
kit-reference comparison across the song. It reuses approved chunk 0, generates
only chunks 1/2 with the exact same reference WAV, and stitches the result. Cached
notebook-16 WAVs or `drum_kit_result.zip` plus original piano are required. It
reuses model/runtime cache and preserves earlier outputs.

[Notebook 16](../notebooks/16_colab_drum_kit_reference.ipynb) runs one controlled
reference-conditioned comparison: kit reference from chunk 1, target chunk 0.
It reuses the active model runtime and verifies the reviewed control/reference
WAV hashes. If the old chunk files are missing, upload the lossless result ZIP
and original piano. It does not re-generate the full song or promise kit locking.

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
