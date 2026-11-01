# Simultaneous strumming and picked guitar in the selected sound style

For an account/VM change, use [notebook 22 and the cache-transfer guide](colab-cache-transfer.md):
CPU backup if old checkpoint files remain, then prepare/restore on the new GPU
VM. It includes this same preview30/full90 layering workflow. A new notebook
does not by itself restore files from an erased Colab VM.

The user wants **both techniques playing together**, and selected the uploaded
opening-30s guitar clip for its sound/style. The latest upload is byte-identical
to the original `guitar_feedback.zip` (SHA-256
`cca5c57ee8513b34c7c40526a0905b6ae819836837672278ae7b3e17a47e2be6`).
The original 90s chunk result remains a useful candidate: strumming in chunk 0
and picking in chunk 2 are liked. The failed original/picked-v2 comparison at
77.130625s is a separate experiment, not a rejection of these other examples.

Chunks 0/2 have different song positions and chords. Shifting one onto the
other would not preserve the piano arrangement. The new workflow makes two
performances on the **same piano timeline**, both conditioned on the same
frozen sound reference, then mixes them. This is new generation, not a mix of
the existing chunk-0/chunk-2 waveforms.

## Run notebook 21

Open [`21_colab_guitar_layers.ipynb`](../notebooks/21_colab_guitar_layers.ipynb).
Copy its cells into the active CUDA Colab notebook to keep the complete runtime
and weight cache. Helpers, a compatible engine/chunk runner and the selected
guitar-only MP3 style preview are embedded with checksums. Upload original
piano only if `SOURCE_PATH` is no longer available. No old guitar result ZIP or
GitHub token is required. An empty new VM still needs notebook 12 setup first.

Choose `MODE` in the generation cell:

| Mode | Output | Model calls if uncached |
| --- | --- | --- |
| `preview30` (default) | Two guitar tracks on piano 0–30s, layered guitars and piano mix | 2 |
| `full90` | Entire song, two guitar tracks each using contexts capped at 90s | 6 for this 4:03 piano |

`START` applies to preview30; full90 covers the whole source. Both full-song
role plans have identical cuts, frames and source bytes. Each uses 6s margins,
0.58s equal-power stitching and the same style-reference hash in every chunk.
The 90s cap includes margins and integer padding. A failed full-song run resumes
verified completed chunks with the same `FULL_LAYERS_OUT`, and exports the two
roles before making the combined mix. An interrupted final mix uses a new mix
attempt folder while completed guitar chunks remain reusable.

The strum/pick layer captions are separate positive descriptions in the same
warm, undistorted electric-guitar style. The picked layer does not reuse failed
picked-v2. Model settings remain Base / LEGO / CUDA FP32 / seed 0 / CFG 7 /
50 steps. The two layers begin with weights 0.45/0.45, with one global headroom
gain for the combined guitar and piano mix. The source piano is included once.

Listen to each role, the layered guitars and the piano mix. Confirm both
techniques, playing feel, chord/timing agreement, sound consistency and glitches.
Reference audio and captions request style/roles; they do not enforce exact
notes or prove that both intended techniques are present. GPU inference and
musical quality of these new layers have not been evaluated on the chat machine.

## Save outputs

- Preview: keep `guitar_layers_result.zip`; send `guitar_layers_feedback.zip`.
- Full song: keep `guitar_layers_full_result.zip`; send
  `guitar_layers_full_feedback.zip`.

Result ZIPs include separate strum/pick WAV exports, combined guitars, piano
mix, reports and the frozen reference. The full result contains final role WAVs;
raw chunk WAVs remain in the Colab run directories. Feedback includes MP3s and
reports; full feedback also has chunk reports and seam excerpts. Original piano,
weights and bass/drum tracks are not included or regenerated.

The reference is the approved **lossy MP3 preview**, not reconstructed bytes of
the original lossless guitar WAV. Earlier archives, preferences and diagnostics
are preserved in the [style baseline](../baselines/clean-electric-guitar-30-v1/)
and [90s candidate review](../reviews/clean-electric-guitar-context90/).
