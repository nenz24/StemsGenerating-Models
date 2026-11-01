# Uji decoder pretrained pada iPhone

Paket ini menguji satu komponen audio ACE-Step: VAE decoder asli, dengan input
latent tetap sepanjang 0,2 atau 1 detik. Ini uji kesesuaian hitungan, bukan generasi
instrumen dari piano atau penilaian groove/akor. Tidak perlu GPU CUDA atau training.

[Hasil iPhone 0,2 detik sudah PASS](../reviews/coreai-iphone17pro-pretrained-decoder-02s/README.md):
tiga run cocok dengan referensi, rata-rata dua run hangat 33,88 ms, dan maksimum
footprint proses tersampel 792,3 MiB.
[Hasil 1 detik juga PASS](../reviews/coreai-iphone17pro-pretrained-decoder-1s/README.md):
rata-rata dua run hangat 159,77 ms dan maksimum tersampel 849,3 MiB.
Tahap berikutnya adalah [encoder pretrained](coreai-pretrained-encoder.md).
Langkah di bawah tetap tersedia untuk mengulang pengujian decoder.

## Jalankan di Mac

Tutup Xcode, lalu dari folder repository:

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_decoder.sh --frames 25 &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

`--frames 25` membuat paket 1 detik terpisah di
`prototypes/ios27-coreai/ModelCandidates/decoder-pretrained-1s/`.
Paket 0,2 detik tetap tersimpan. Tanpa opsi (atau `--frames 5`), skrip menyiapkan
paket 0,2 detik seperti sebelumnya. Jangan menghapus paket baseline untuk uji ini.

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
   `prototypes/ios27-coreai/ModelCandidates/decoder-pretrained-1s/` untuk 1 detik,
   atau `decoder-pretrained/` untuk 0,2 detik.

Paket akhir sekitar 338 MB. Checkpoint/cache, environment, staging dan paket besar
diabaikan Git. Repo hanya berisi kode, dokumentasi dan manifest validasi kecil.
Jika kedua profil tersedia, Xcode menyalin keduanya: total resource model sekitar
676 MB. Penggandaan ini untuk membandingkan profil pengujian, bukan ukuran produk akhir.
Skrip tidak mengunduh Base generator, Turbo, Qwen atau language model. Credential
dan pengaturan sertifikat tetap memakai lingkungan yang tersedia; VAE bersifat publik.

Ekspor yang gagal meninggalkan staging di `.runtime-cache/coreai-decoder-staging`
untuk diperiksa, di luar folder yang disalin ke aplikasi. Paket lama tidak
ditimpa. Eksekusi ulang dengan paket lengkap akan memverifikasi dan memakainya
kembali, tanpa download/re-export. Untuk checkpoint yang berada di tempat lain,
gunakan opsi `--checkpoint` pada `scripts/export_coreai_decoder.py` dalam Python
environment tersebut. Checkpoint eksplisit tetap wajib lolos hash publisher.

## Jalankan di iPhone

Pilih iPhone 17 Pro di Xcode dan **Cmd+R**, lalu tekan **Uji decoder 1 detik**.
Untuk baseline lama, tombol **Uji decoder pretrained** tetap memakai 0,2 detik.
Untuk perbandingan footprint yang lebih mudah dibaca, buka ulang aplikasi dan
jalankan profil yang ingin diukur terlebih dahulu; hasil tetap dipengaruhi cache.
Signing Team tetap dibaca dari `Signing.local.xcconfig`. Folder resource yang
sudah ada menyalin paket lokal beserta referensinya ke aplikasi saat build.
Bagikan laporan lengkap setelah pengujian. Jika compile gagal, kirim error Xcode;
profil decoder sudah dijalankan pengguna di perangkat, tetapi perubahan terbaru
untuk menambahkan encoder belum dikompilasi dengan Apple SDK di workspace Linux ini.

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

Input adalah latent standard-normal seed 0, dengan kontrak masing-masing:

| Profil | Input Float32 | Output Float32 | Tombol |
| --- | --- | --- | --- |
| 0,2 detik | [1, 64, 5] | [1, 2, 9600] | Uji decoder pretrained |
| 1 detik | [1, 64, 25] | [1, 2, 48000] | Uji decoder 1 detik |

Output 48 kHz stereo. Referensi disimpan sebagai raw little-endian Float32.
Aplikasi memeriksa ukuran dan SHA-256 referensi sebelum membaca tensor.

Syarat awal parity: setiap error elemen <= `1e-4 + 1e-3 * abs(reference)`, dan
normalized RMSE <= `1e-3`. Syarat RMSE tambahan mencegah keluaran nol terlihat
lulus hanya karena amplitudo referensi kecil. Ini gerbang numerik awal, bukan
ambang mutu musik. Jangan memperlonggar toleransi hanya untuk membuat tes lulus.

[Validasi ekspor di Linux](../reviews/coreai-pretrained-decoder-export/README.md)
mencatat checkpoint lengkap terverifikasi, reference FP32 dan ekspor native yang
berhasil. Hash ekspor/timestamp dapat berbeda pada Mac; aplikasi membaca manifest
paket lokalnya sendiri, bukan memaksa hash ekspor Linux.
[Ekspor 1 detik](../reviews/coreai-pretrained-decoder-export-1s/README.md) juga sudah
berhasil dan diverifikasi di Linux. Hasil iPhone tersedia untuk 0,2 dan 1 detik.
Encoder, conditioning dan diffusion generator belum diuji di iPhone. Angka decoder ini
belum menunjukkan kecepatan generasi pengiring lengkap atau kelayakan real-time.
