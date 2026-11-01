# Colab CPU backup and reuse in another account/VM

Use [`22_colab_cache_transfer.ipynb`](../notebooks/22_colab_cache_transfer.ipynb).
It includes current source, compatible guitar engine and the selected guitar
style preview, then supports two actions:

| Action | Where | GPU needed? |
| --- | --- | --- |
| `backup` (default) | A session that still has the old checkpoint files | No: inventory, hashes and file copying use CPU |
| `prepare` | A destination VM/account with available GPU access | Yes for CUDA validation and guitar generation |

The **Connect without GPU / Sambungkan tanpa GPU** button is sufficient for
backup. It may attach a new empty CPU VM instead of retaining the old disk.
The notebook checks actual files before attempting export. It cannot recover
an erased Colab filesystem, transfer a running VM between accounts or transfer
GPU quota. Notebook outputs still visible in the browser do not establish that
their old runtime files exist.

## Old session: backup on CPU

Copy the notebook cells into the old connected session to retain any accessible
cache and globals. Leave `ACTION="backup"`, connect without GPU and Run all.
The backup branch does not import the model, check CUDA, install runtime
packages or generate audio. It locates existing checkpoint files independently
of whether the old Python venv still works.

By default, mount Google Drive and save to
`/content/drive/MyDrive/StemsGeneratingCache/ace-base-<manifest-id>.tar`.
The archive contains only the 34 publisher-pinned asset files, approximately
**14,883,891,741 bytes / 13.86 GiB / 14.9 GB**, plus small TAR metadata. Ensure
destination storage is sufficient; Drive storage also shares quota with other
account data. `USE_DRIVE=False` and `ARCHIVE_PATH` support another storage path.
Local `/content` files must be copied somewhere persistent before losing the VM.

The exporter verifies publisher SHA-256/Git-blob hashes while streaming each
file. It excludes the Python venv, duplicate Hugging Face download cache,
credentials, source repository internals and generated audio. Existing archives
and pre-existing partial files are preserved. A failed transfer never becomes a
completed `.tar`; interruptions that kill the process can leave `.partial`
files, which are not usable backups.

## New account: restore and run

1. Give the destination account access to the saved archive through Drive's
   normal sharing controls, or transfer it yourself to accessible storage.
   The new account does not automatically see the old account's MyDrive.
2. Add a shortcut to the shared archive/folder in the new account's MyDrive and
   set `ARCHIVE_PATH` to its mounted path if different from the default.
3. Select a Colab GPU runtime that is available to that account. Set
   `ACTION="prepare"`, then Run all.

Complete local assets are reused and verified. Otherwise, the archive is
restored to the pinned upstream checkout. Restore accepts only the expected
regular asset members and verifies them against the current trusted manifest;
it does not extract arbitrary TAR paths or links. Verified files already
restored are reused on retry. Wrong or changed existing assets are preserved
and reported, not overwritten silently.

Runtime Python/dependencies are installed for the new VM if needed. A `.venv`
is not copied between paths/VMs because its interpreter links and environment
may not be portable. Runtime package downloads can therefore still be needed.
Restoring a model archive avoids downloading the model again from Hugging Face,
but reading it from Drive still transfers the model bytes to the new VM disk.

`ALLOW_MODEL_DOWNLOAD=False` is the default. If neither local assets nor a
readable archive are found, the notebook stops before a fresh model download,
so a missing Drive shortcut cannot silently trigger another 14.9 GB download.
If the old files are truly gone and no backup exists, explicitly set this flag
to True to acquire the pinned model again. Source checkout/dependencies still
need normal Internet access. The notebook does not require GitHub credentials.

The remaining cells offer the same guitar workflow as notebook 21:
`MODE="preview30"` or `MODE="full90"`. The backup action skips these cells.
Audio outputs, recordings and chunk progress need their own saved result files;
the weight archive does not contain them.

## Local commands and validation

The source bundle also includes the CPU helper:

```bash
python scripts/colab_weight_cache.py inventory --root /path/to/old/bridge
python scripts/colab_weight_cache.py export --root /path/to/old/bridge --archive /persistent/ace-base.tar
python scripts/colab_weight_cache.py restore --root /path/to/new/bridge --archive /persistent/ace-base.tar
```

Use the pinned upstream checkout/setup before restore for a model-ready new
bridge. Helper tests exercise both digest algorithms, CPU-only export/restore,
retry, corruption rejection, protected existing files, archive paths/links and
the notebook's backup/prepare dry-run branches. Tests use small assets; the full
14.9 GB archive, Drive account access and destination GPU must be validated in
the user's Colab sessions.
