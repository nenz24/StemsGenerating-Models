# Uji decoder pretrained pada iPhone

Paket ini menguji satu komponen audio ACE-Step: VAE decoder asli, dengan input
latent tetap sepanjang 0,2 detik. Ini uji kesesuaian hitungan, bukan generasi
instrumen dari piano atau penilaian groove/akor. Tidak perlu GPU CUDA atau training.

## Jalankan di Mac

Tutup Xcode, lalu dari folder repository:

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_decoder.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip menyiapkan environment terpisah di `.runtime-cache`, memakai Python 3.12
melalui `uv` jika tersedia. Tanpa uv, skrip mencari Python 3.11–3.13 yang sudah
terpasang. Jika belum tersedia, pasang Python 3.12 atau uv lalu jalankan lagi.
Python 3.14 belum dipilih karena wheel Torch 2.8 yang dipin memakai versi sebelumnya.
Managed Python, cache dan direktori executable uv juga diarahkan ke
`.runtime-cache/uv/` untuk menghindari permission error pada `~/.local/share/uv`
atau cache global. Pengaturan ini hanya berlaku pada proses persiapan; tidak
memerlukan sudo atau perubahan ownership direktori home.

Skrip akan:

1. Memakai checkpoint VAE yang sudah ada di vendor/cache bila tersedia.
2. Jika belum ada, mengunduh hanya VAE sekitar 337 MB dari revisi publisher
   yang dipin. Ukuran dan seluruh SHA-256 diperiksa sebelum bobot digunakan.
3. Memuat hanya tensor decoder dan mengubahnya ke FP32. Weight normalization
   dilipat dari bobot asli; keluaran sebelum/sesudah wajib identik pada input uji.
4. Membuat referensi PyTorch serta ekspor native dengan coreai-torch.
5. Memeriksa hash semua file paket dan menyimpan hasil di folder lokal:
   `prototypes/ios27-coreai/ModelCandidates/decoder-pretrained/`.

Paket akhir sekitar 338 MB. Checkpoint/cache, environment, staging dan paket besar
diabaikan Git. Repo hanya berisi kode, dokumentasi dan manifest validasi kecil.
Skrip tidak mengunduh Base generator, Turbo, Qwen atau language model. Credential
dan pengaturan sertifikat tetap memakai lingkungan yang tersedia; VAE bersifat publik.

Ekspor yang gagal meninggalkan staging di `.runtime-cache/coreai-decoder-staging`
untuk diperiksa, di luar folder yang disalin ke aplikasi. Paket lama tidak
ditimpa. Eksekusi ulang dengan paket lengkap akan memverifikasi dan memakainya
kembali, tanpa download/re-export. Untuk checkpoint yang berada di tempat lain,
gunakan opsi `--checkpoint` pada `scripts/export_coreai_decoder.py` dalam Python
environment tersebut. Checkpoint eksplisit tetap wajib lolos hash publisher.

## Jalankan di iPhone

Pilih iPhone 17 Pro di Xcode dan **Cmd+R**, lalu tekan **Uji decoder pretrained**.
Signing Team tetap dibaca dari `Signing.local.xcconfig`. Folder resource yang
sudah ada menyalin paket lokal beserta referensinya ke aplikasi saat build.
Bagikan laporan lengkap setelah pengujian. Jika compile gagal, kirim error Xcode;
source Swift baru belum dikompilasi di workspace Linux ini.

Jika laporan menyebut paket belum disiapkan, pastikan skrip Mac selesai dan
aplikasi dibuild ulang setelah persiapan. Paket besar tidak tersedia hanya dari
`git pull`: ekspor lokal adalah langkah yang diperlukan.

Laporan memuat:

- Validasi aset, waktu validasi dan waktu pemuatan/specialization.
- Run pertama dan dua run hangat, termasuk read-back output, di luar waktu perbandingan.
- Max absolute error, RMSE dan normalized RMSE terhadap referensi lokal.
- Baseline dan maksimum footprint proses yang tersampel setiap 10 ms.
- Status model dimuat, inference berjalan dan parity lulus/gagal.

Maksimum memori tersampel bukan peak persis dan mencakup seluruh proses, bukan
alokasi model terisolasi. Lonjakan di antara sampel dapat terlewat. Kebijakan
yang diminta adalah CPU-only; penempatan hardware tidak diukur. Cache model
CoreAI dapat mengubah waktu pemuatan antar-pengujian.

## Kontrak dan batas bukti

Checkpoint `ACE-Step/Ace-Step1.5` revisi `19671f406d603126926c1b7e2adc169acbcade22`,
`vae/diffusion_pytorch_model.safetensors`, SHA-256:
`da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0`.

Input adalah Float32 [1, 64, 5], latent standard-normal seed 0. Output Float32
[1, 2, 9600], 48 kHz stereo. Referensi disimpan sebagai raw little-endian Float32.
Aplikasi memeriksa ukuran dan SHA-256 referensi sebelum membaca tensor.

Syarat awal parity: setiap error elemen <= `1e-4 + 1e-3 * abs(reference)`, dan
normalized RMSE <= `1e-3`. Syarat RMSE tambahan mencegah keluaran nol terlihat
lulus hanya karena amplitudo referensi kecil. Ini gerbang numerik awal, bukan
ambang mutu musik. Jangan memperlonggar toleransi hanya untuk membuat tes lulus.

[Validasi ekspor di Linux](../reviews/coreai-pretrained-decoder-export/README.md)
mencatat checkpoint lengkap terverifikasi, reference FP32 dan ekspor native yang
berhasil. Hash ekspor/timestamp dapat berbeda pada Mac; aplikasi membaca manifest
paket lokalnya sendiri, bukan memaksa hash ekspor Linux. Belum ada hasil iPhone
untuk decoder pretrained ini; affine PASS tidak membuktikan codec/audio PASS.
