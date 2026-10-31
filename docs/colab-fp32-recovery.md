# Melanjutkan Colab setelah latent NaN pada T4

Pakai **sesi Colab yang sama**, setelah sel generasi berhenti dengan error
`Generation produced NaN or Inf latents`. Runtime dan bobot sudah tersedia.
Tambahkan satu sel kode di bawah sel yang gagal, salin seluruh kode berikut,
lalu jalankan. `ROOT` dan `REFERENCE` berasal dari notebook 09 yang sudah berjalan.
Sel ini menjalankan ulang satu input B dengan FP32 untuk DiT, text encoder dan
VAE. Hasil masuk ke folder baru; bobot terverifikasi yang ada dipakai kembali.

Ini adalah retry dengan precision berbeda, bukan hasil CUDA yang sudah terbukti
bersih. FP32 memerlukan lebih banyak memori dan dapat lebih lambat daripada FP16.
Tidak mengubah CFG, seed, input B, source upstream atau isi ZIP.

```python
import json, uuid
from unittest.mock import patch
import cuda_reference

FP32_WORKER = r'''
import runpy, sys
from pathlib import Path
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parents[2] / "vendor/ACE-Step-1.5"))
import torch
import acestep.handler as ace_handler

class Float32Handler(ace_handler.AceStepHandler):
    def _load_main_model_from_checkpoint(self, *args, **kwargs):
        if kwargs.get("device") != "cuda":
            raise RuntimeError("FP32 CUDA worker requires device=cuda")
        self.dtype = torch.float32
        print("[Precision] Override pinned CUDA loader: DiT/text/VAE=float32", flush=True)
        return super()._load_main_model_from_checkpoint(*args, **kwargs)

    def _get_vae_dtype(self, device=None):
        return torch.float32

ace_handler.AceStepHandler = Float32Handler
sys.argv = sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
'''

WORK = ROOT / "runs" / ("cuda_fp32_" + uuid.uuid4().hex[:12])
original_run = cuda_reference.run_logged

def run_fp32(command, **kwargs):
    if command[1] != "-c":
        command = [command[0], "-c", FP32_WORKER, *command[1:]]
    return original_run(command, **kwargs)

try:
    with patch.object(cuda_reference, "run_logged", run_fp32):
        report = cuda_reference.run_reference(REFERENCE, WORK, execute=True)
finally:
    report_file = WORK / "cuda_reference.json"
    if report_file.is_file():
        report = json.loads(report_file.read_text())
        report["precision_override"] = "float32"
        report["precision_policy"] = "Explicit FP32 recovery in existing Colab session"
        report_file.write_text(json.dumps(report, indent=2) + "\n")

print("Folder hasil:", WORK)
print(json.dumps(report, indent=2))
```

Jika berhasil, jalankan sel pembanding audio dan ekspor `cuda_result.zip` yang
ada di bawah sel generasi pada notebook 09. Variabel `WORK` sudah menunjuk run
baru. Dengarkan dan bandingkan `cuda_listen.wav` dengan `mps_baseline_listen.wav`.
Jika kembali error, kirim traceback terakhir. Report tetap mencatat status gagal;
sel ini tidak mengubah NaN menjadi angka atau melewati pemeriksaan validitas.

Pesan upstream menyarankan `ACESTEP_DTYPE=float32`, tetapi pada SHA pinned
`ca1e85fe9430179831e6bc6be790c332190a3866` variabel itu hanya disebut dalam pesan
error, tanpa dibaca oleh loader. Worker di atas mengatur dtype sebelum pemuatan
bobot dan mengatur dtype VAE saat pindah CPU/GPU. Ia berjalan dalam proses model
baru dan tidak memasang PyTorch ke kernel Colab.

Perbaikan permanen pada `scripts/cuda_reference.py` memilih worker ini otomatis
untuk compute capability di bawah 8.0, mencatat kebijakan precision, dan memeriksa
dtype model/decoder yang ditangkap. Untuk sesi baru: pull di Mac, ekspor ZIP baru,
lalu upload notebook 09 dan ZIP baru. Sesi saat ini cukup memakai sel di atas.
