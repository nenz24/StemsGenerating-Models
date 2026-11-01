# StemsGenerating-Models

POC JupyterLab untuk membuat bass/drum pengiring dari audio memakai pretrained
ACE-Step 1.5 Base. Model belum dilatih ulang atau dikonversi ke CoreAI/iPhone.

**Baseline drum seluruh lagu sudah disetujui.** Hasil 4:03 memakai chunk maksimal
90 detik dengan referensi kit yang sama; arsip, parameter, dan hash disimpan di
[`acoustic-drums-shared-kit-full-v1`](baselines/acoustic-drums-shared-kit-full-v1/).
Tahap berikutnya adalah uji gitar elektrik clean yang mengiringi piano.

**Target produk: seluruh inferensi lokal di iPhone / iOS 27 / CoreAI.** Rekaman
piano menghasilkan gitar+bass+drum; rekaman gitar menghasilkan piano+bass+drum.
Rekaman asli tetap disimpan. [Kebutuhan dan rancangan on-device](docs/iphone-coreai.md)
memisahkan target produk dari eksperimen CUDA; dukungan SDK/model CoreAI belum
diverifikasi dan belum ada implementasi iPhone di repository ini.

**Pilihan aktif: Electric Bass.** Pengguna memilih hasil bass elektrik petikan
natural pada CUDA/T4 FP32. Preset menyimpan caption dan setting yang sama:
Base, seed 0, CFG 7, 50 langkah. Lihat
[preset terpilih dan generasi langsung](docs/electric-bass-preset.md) serta
[`configs/electric-bass.json`](configs/electric-bass.json).

Generasi klip berikutnya langsung memakai preset ini, tanpa mengulang A/B:

```bash
python3 scripts/compare_bass_tone.py --prepared /path/to/prepared --out runs/electric_plan
```

Tambahkan `--execute` untuk inferensi. Entry point klip ini mendukung 10–30 detik.
Untuk durasi panjang, tersedia **[mode seluruh lagu satu panggilan](docs/electric-full-song.md)**
hingga 600 detik, tanpa wajib chunk 30 detik. Pada T4 FP32, mulai dengan uji
60 detik lalu seluruh tiga menit sesuai hasil memori/kualitas. Fit GPU dan kualitas
durasi panjang belum diuji di chat. Untuk Colab yang masih
aktif, satu helper terbaru dapat diupload dan memakai runtime/bobot yang sudah
ada. [Uji caption sebelumnya](docs/bass-tone-colab.md) dan
[hasil CUDA awal](docs/cuda-fp32-result.md) disimpan sebagai catatan.

Untuk kernel/VM Colab baru, gunakan
[`10_colab_electric_full_song.ipynb`](notebooks/10_colab_electric_full_song.ipynb):
notebook menentukan `ROOT`, memulihkan source dari ZIP jika diperlukan, menyiapkan
runtime/bobot yang belum ada, lalu menjalankan full-song tanpa baseline A/B.

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
- Batas 10/30 detik milik runner diagnosis lama. Runner single-pass Electric Bass
  menerima 10–600 detik secara eksplisit; kemampuan durasi model tidak menjamin
  semua panjang muat pada semua GPU, terutama FP32 pada T4.
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
