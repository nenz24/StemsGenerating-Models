# StemsGenerating-Models

POC JupyterLab untuk membuat bass/drum/gitar pengiring dari audio memakai pretrained
ACE-Step 1.5 Base. Komponen VAE dan denoiser sudah diekspor ke CoreAI;
generasi pengiring lengkap di iPhone belum diuji. Model belum dilatih ulang.

**Pindah akun/VM Colab:** [notebook 22](notebooks/22_colab_cache_transfer.ipynb)
memisahkan backup bobot via CPU (`ACTION="backup"`) dan restore pada VM GPU
tujuan (`ACTION="prepare"`), lalu melanjutkan gitar berlapis. Cache lama harus
masih ada atau sudah dibackup; VM yang terhapus tidak dapat dipulihkan lewat
notebook. Lihat [panduan transfer cache](docs/colab-cache-transfer.md).

**Baseline drum seluruh lagu sudah disetujui.** Hasil 4:03 memakai chunk maksimal
90 detik dengan referensi kit yang sama; arsip, parameter, dan hash disimpan di
[`acoustic-drums-shared-kit-full-v1`](baselines/acoustic-drums-shared-kit-full-v1/).
**Gaya gitar dipilih dari contoh 30 detik.** Pengguna memperjelas bahwa chunk
90 detik tetap berguna: strumming chunk 0 dan petikan chunk 2 disukai, tetapi
suaranya berubah antar-chunk. Target sekarang adalah kedua teknik **dilapis
bersamaan**, dengan [referensi suara yang dipilih](baselines/clean-electric-guitar-30-v1/).
[Notebook 21](notebooks/21_colab_guitar_layers.ipynb) membuat dua track pada piano
yang sama dengan referensi suara bersama, mode preview30 atau seluruh lagu full90.
Lihat [panduan layering](docs/guitar-layers.md). Uji original/picked-v2 yang gagal
adalah eksperimen terpisah; notebook 18–20 disimpan untuk reproduksi.
Hasil bass/drum lama tetap disimpan untuk mix gabungan.

**Target produk: seluruh inferensi lokal di iPhone / iOS 27 / CoreAI.** Rekaman
piano menghasilkan gitar+bass+drum; rekaman gitar menghasilkan piano+bass+drum.
Rekaman asli tetap disimpan. [Kebutuhan dan rancangan on-device](docs/iphone-coreai.md)
memisahkan target produk dari eksperimen CUDA. **[Proyek Xcode CoreAIProbe](prototypes/ios27-coreai/README.md)**
siap dibuka setelah pull untuk uji runtime/tensor pada iPhone 17 Pro / iOS 27.
Pengguna sudah menjalankan [probe tensor pada iPhone](reviews/coreai-iphone17pro-device-probe/README.md):
Float32 round-trip **PASS**, iOS 27.0.1, arsitektur CoreAI `h18p`.
[Model native numerik juga lulus di iPhone](reviews/coreai-iphone17pro-native-affine/README.md):
validasi aset true, model dimuat dan 3 inference tepat (error absolut 0).
Generasi audio CoreAI dan penggunaan GPU/Neural Engine oleh model belum diuji.
Uji `.mlpackage` ditolak saat pemuatan (`Missing hash file`); jalur berikutnya
sudah diganti dengan export native `.aimodel` melalui `coreai-torch` resmi.
Lihat [jalur native](docs/coreai-native-export.md). Ekspor dan referensi PyTorch
sudah dibuat dan inference model numerik native pada iPhone sudah dilaporkan lulus.

**Audit komponen audio CoreAI:** [hasil dan rencana decoder VAE](reviews/coreai-ace-component-audit/README.md).
Empat probe operator sintetis berhasil diekspor native. Decoder pretrained 0,2 dan 1 detik
sudah lulus parity di iPhone; generasi pengiring lengkap belum diuji. Ukuran bobot dan feature map diaudit
terpisah dari waktu inference model numerik kecil.

**Decoder pretrained 1 detik PASS:** [laporan iPhone](reviews/coreai-iphone17pro-pretrained-decoder-1s/README.md),
rata-rata dua run hangat 159,77 ms dan maksimum footprint proses tersampel 849,3 MiB.
**Encoder pretrained 1 detik juga PASS:** [laporan iPhone](reviews/coreai-iphone17pro-pretrained-encoder-1s/README.md),
rata-rata dua run hangat 131,33 ms. Footprint proses tersampel 97 MiB bukan ukuran
alokasi model; encoder FP32 yang diekspor tetap sekitar 338 MB.
**Blok DiT pretrained pendek juga PASS:** [laporan iPhone](reviews/coreai-iphone17pro-pretrained-dit-block/README.md),
rata-rata dua run hangat 4,51 ms untuk satu blok, 16 hidden / 8 condition token.
**Dua blok DiT konteks 160/64 juga PASS:** [laporan iPhone](reviews/coreai-iphone17pro-pretrained-dit-context/README.md),
rata-rata dua run hangat 31,10 ms.
**Bobot tersimpan FP16/hitungan FP32 juga PASS pada kedua referensi:**
[laporan iPhone](reviews/coreai-iphone17pro-pretrained-dit-storage16/README.md),
dua run hangat rata-rata 32,46 ms. Footprint tersampel 638,2 MiB tidak menunjukkan
penghematan RAM dibanding profil FP32 sebelumnya.
**Denoiser monolitik mencapai limit memori saat loading/specializing:**
[laporan perangkat](reviews/coreai-iphone17pro-full-denoiser-memory-failure/README.md).
Jalur itu dinonaktifkan. [Uji denoiser bertahap](docs/coreai-full-denoiser.md)
memakai 14 asset berurutan, masing-masing sekitar 252 MB atau kurang, dengan
copy tensor CPU antar bagian dan guard footprint sebelum tahap berikutnya.
Jalankan `bash scripts/prepare_coreai_denoiser.sh`, build ulang, lalu tekan
**Uji denoiser bertahap**. Checkpoint cache dipakai ulang; paket monolitik diarsipkan
di luar resource aplikasi. [Hasil iPhone](reviews/coreai-iphone17pro-sharded-denoiser/README.md)
sudah PASS 14/14 dan kedua referensi pada pengulangan dalam aplikasi yang sama,
setelah run pertama gagal di BNNS. Penyebab kegagalan pertama belum pasti.
**Uji kestabilan 3x kini PASS 3/3 di iPhone:**
[laporan lengkap](reviews/coreai-iphone17pro-sharded-denoiser-stability/README.md).
Waktu total 4,68 / 3,46 / 2,04 detik untuk kasus sintetis yang sama; latensi,
cold launch dan kebutuhan RAM model belum tervalidasi sebagai anggaran produksi.
**Loop denoising 4 langkah kini PASS di iPhone:**
[laporan perangkat](reviews/coreai-iphone17pro-denoiser-loop/README.md), 4/4 langkah,
56/56 tahap dan seluruh 16 pembandingan velocity/latent lolos. Waktu total
53,00 detik; normalized RMSE latent akhir terhadap FP32 asli sekitar `7,01e-6`.
**Profil waktu juga PASS:** [laporan iPhone](reviews/coreai-iphone17pro-denoiser-loop-timing/README.md).
Total 51,88 detik; 50,75 detik berada di interval `function.run`, atau 99,36%
waktu tahap terukur. Opening, binding dan copy masing-masing sekitar 0,10–0,12
detik. Berikutnya perlu membedakan biaya run pertama dan run berulang dengan
satu pasangan layer/fungsi tetap dimuat. [Panduan profiling](docs/coreai-loop-timing.md).
Ini belum generasi instrumen lengkap.

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
