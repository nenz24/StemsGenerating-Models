# StemsGenerating-Models

POC JupyterLab untuk membuat bass/drum pengiring dari audio memakai pretrained
ACE-Step 1.5 Base. Model belum dilatih ulang atau dikonversi ke Core AI/iPhone.

**Status kualitas:** pengguna melaporkan hasil B pada CUDA/T4 FP32 lebih bersih
dari glitch dibanding MPS, tetapi timbre bass masih terlalu dalam. Paket hasil
berisi latent finite dan capture FP32. Ini satu contoh, belum membuktikan glitch
hilang untuk semua input. Lihat [hasil CUDA](docs/cuda-fp32-result.md).

Langkah aktif adalah **satu kandidat bass elektrik petikan natural**, dengan
caption baru dan input/seed/CFG/langkah/FP32 yang sama. Ikuti
[panduan Colab bass elektrik](docs/bass-tone-colab.md) untuk memakai sesi dan
bobot yang sudah ada. Cukup upload `scripts/compare_bass_tone.py` ke sesi lama;
baseline dipertahankan dan kandidat masuk ke direktori baru.

## Satu uji di Google Colab

Di checkout Mac yang menyimpan hasil percobaan sebelumnya:

```bash
git pull --rebase --autostash
python3 scripts/cuda_reference.py --export
```

Perintah mencetak lokasi `cuda_reference_*.zip`. Paket berisi input B 30 detik
byte-identik, baseline MPS, setting/hash, dan snapshot kode commit Git saat ini.
Tidak menjalankan model lagi di Mac. Bobot, kredensial, run lain dan output
notebook lokal yang belum di-commit tidak disertakan.

1. Buka [Google Colab](https://colab.research.google.com/) → **File → Upload notebook**.
   Pilih **`notebooks/09_colab_reference.ipynb`** dari checkout Mac.
2. Pilih **Runtime → Change runtime type → GPU** (misalnya T4 jika tersedia).
3. **Run all**, lalu upload `cuda_reference_*.zip` saat diminta pada sel pertama.
   Tidak memerlukan terminal atau token GitHub di Colab.
4. Bandingkan `mps_baseline_listen.wav` dan `cuda_listen.wav`. Sel terakhir
   mengunduh `cuda_result.zip` berisi audio/report tanpa bobot model.

Notebook memeriksa GPU/disk, menyiapkan runtime Python 3.11 terisolasi, menyediakan
aset pinned, dan menjalankan **satu** generasi: Base, seed 0, CFG 7, 50 langkah,
30 detik. Bukan sweep seed/CFG dan bukan generasi seluruh lagu. Jika CUDA tidak
tersedia di runtime model, proses gagal dengan jelas; tidak beralih ke CPU.

VM baru memerlukan ruang sekitar 45–50 GiB untuk runtime, bobot dan cache.
Notebook Colab memasang ffmpeg jika belum tersedia. GPU mengikuti ketersediaan
akun, dan sesi Colab bersifat sementara; unduh hasil sebelum sesi berakhir.
Paket yang sama memakai kembali runtime/cache pada VM yang sama. Dtype CUDA dipilih
upstream sesuai GPU dan dicatat; RNG/dtype CUDA dapat berbeda dari MPS. Satu hasil
CUDA saja belum membuktikan bug MPS. Environment chat tidak mempunyai GPU NVIDIA;
hasil CUDA nyata yang diperiksa berasal dari run Colab pengguna.

Lihat [panduan CUDA](docs/cuda-reference.md) untuk rincian dan penggunaan CLI.
Untuk server JupyterLab NVIDIA milik sendiri, notebook 08 tetap tersedia.
Jika T4 gagal dengan latent NaN, tersedia [sel retry FP32](docs/colab-fp32-recovery.md)
untuk melanjutkan sesi yang sama. Runner terbaru memilih FP32 otomatis pada
GPU pre-Ampere. Hasil pertama dan batas kesimpulannya ada di catatan hasil CUDA.

## Batas POC saat ini

- ACE-Step masih **audio-only conditioning**. BPM/key/time signature dari SongMap
  belum dikirim sebagai parameter model; chord/beat disimpan untuk evaluasi.
  Click B/C tersedia pada adapter STAGE. Hipotesis manfaat metadata SongMap belum
  diuji pada jalur ACE.
- Input pembanding tetap 32 kHz PCM16 hasil persiapan sebelumnya. Versi native/
  48 kHz harus diuji terpisah setelah baseline, bukan dicampur ke perubahan device.
- Batas 10/30 detik adalah batas bridge saat ini, bukan bukti batas arsitektur
  ACE-Step. Inferensi 60–180 detik memerlukan uji memori/kualitas tersendiri.
- Chunk independen dapat berbeda karakter/groove. Crossfade tidak memperbaiki
  audio yang sudah rusak sebelum stitching. Mode lagu penuh masih eksperimen.
- Ini belum model generatif on-device siap iPhone. Feature/model integration,
  kualitas musik dan ukuran model akhir masih perlu diselesaikan.

## Notebook dan catatan lain

- [01_quality_poc](notebooks/01_quality_poc.ipynb): hardware dan plan aman.
- [02_mac_24gb_full](notebooks/02_mac_24gb_full.ipynb): setup/inferensi Mac lokal.
- Notebook 03–07: percobaan lagu penuh dan diagnosis lama; lihat
  [arsip diagnosis](docs/diagnostics-history.md) sebelum mengulangnya.
- [Catatan model](docs/model-notes.md) dan [format feedback](feedback/README.md).

## Validasi kode

```bash
python3 -m unittest discover -s Tests/StemEvaluationPython -v
```

Tes menilai audio/timeline, hash/cache, CLI plan, kontrak adapter, dan simulasi
kegagalan tanpa bobot model. Hasil tes kode tidak membuktikan mutu bass atau
keberhasilan CUDA nyata. Notebook 08 dan 09 diuji dalam mode dry-run dengan
fixture sintetis, termasuk folder kerja bergaya Colab yang terpisah dari source.
Hasil inferensi CUDA harus berasal dari runtime GPU yang benar-benar dijalankan.
