# Denoiser lengkap dijalankan per bagian

Uji monolitik berhenti dengan
[`EXC_RESOURCE`, limit 3376 MB](../reviews/coreai-iphone17pro-full-denoiser-memory-failure/README.md).
Tahap terakhir yang tersimpan adalah `loading/specializing complete model`;
belum ada hasil inference. Jalur monolitik pada aplikasi kini dinonaktifkan.
Nama thread `llvm-worker-0` tidak membuktikan adanya proses worker terpisah.

Jika Xcode paused pada exception, tekan **Stop (■)** terlebih dahulu. UI dapat
terlihat hang karena debugger menghentikan proses. Menunggu atau menekan Continue
tidak memperbaiki beban pemuatan model monolitik.

## Jalankan perbaikan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_denoiser.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip kini menyiapkan **14 native asset**: proyeksi/embedding awal, 12 pasangan
layer asli yang mencakup seluruh layer 0–23, lalu normalisasi/proyeksi akhir.
Bagian terbesar sekitar **252 MB**, menggantikan pemuatan satu asset 3,15 GB.
Checkpoint Base dan environment yang sama dipakai ulang dari cache. Jika cache
hilang, unduhan checkpoint tetap 4,79 GB, dengan SHA-256 penuh diverifikasi.

Paket baru berada di `ModelCandidates/denoiser-sharded/`, total sekitar 3,15 GB.
Paket monolitik lama dipindahkan utuh ke direktori unik di
`.runtime-cache/coreai-retired-bundles/`, di luar resource yang disalin Xcode.
Lokasi arsip dicetak oleh skrip. Enam baseline komponen sebelumnya tetap tersedia.
Pengulangan skrip memverifikasi paket yang sudah ada tanpa ekspor ulang.

## Jalankan di iPhone

Setelah Stop dan persiapan selesai, pilih **Product → Clean Build Folder**
(**Shift+Cmd+K**) sekali agar resource monolitik lama tidak tertinggal di hasil
build. Jalankan **Cmd+R**, lalu tekan **Uji denoiser bertahap**.
Sebaiknya buka ulang aplikasi dan jalankan ini terlebih dahulu. Laporan pada
layar diperbarui per bagian, dan tahap terakhir disimpan ke Documents secara
best effort. Jika berhenti, kirim laporan lengkap atau tahap terakhirnya.
Pengguna sudah menjalankan caller ini di iPhone: lihat hasil di bawah.
Workspace Linux tidak menyediakan Apple SDK untuk kompilasi independen.

Caller memegang satu model bagian di dalam scope async tersendiri. Outputnya
disalin ke NDArray CPU baru sebelum scope berakhir, supaya tensor antara tidak
mempertahankan buffer milik model sebelumnya. Cache internal CoreAI tidak
dianggap langsung terhapus: scope Swift saja belum menjamin unload pada backend.

Sebelum setiap bagian berikutnya, caller membaca footprint proses dan berhenti
jika sudah mencapai **2000 MiB**, menyisakan ruang di bawah limit yang dilaporkan.
Ini pemeriksaan sebelum tahap, bukan pembatas memori OS. Lonjakan saat kompilasi
satu bagian masih dapat melewati sampling/guard. Pengurangan kebutuhan RAM model
dan peak memori yang sebenarnya belum terukur.

## Hasil pengulangan pada perangkat

[Dua laporan iPhone](../reviews/coreai-iphone17pro-sharded-denoiser/README.md)
mencatat kegagalan BNNS di bagian 13 pada percobaan pertama, kemudian **PASS
14/14 bagian dan kedua referensi** setelah tombol ditekan lagi dalam aplikasi
yang sama. Run kedua selesai dalam 3,79 detik; normalized RMSE terhadap FP32 asli
sekitar `3,30e-6`. Penyebab kegagalan pertama belum dipastikan. Perubahan timing
sesuai dengan pengaruh cache/inisialisasi, tetapi bukan bukti cache rusak.
Footprint tersampel 40,22 MiB pada run kedua bukan seluruh RAM model atau perangkat.

Untuk memeriksa kestabilan tanpa mengubah model, ambil pembaruan dan buka proyek:

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Jika paket 14 bagian sudah disiapkan seperti pada laporan tersebut, tidak perlu
menjalankan skrip persiapan atau mengunduh bobot ulang. Build dengan **Cmd+R**,
tekan **Uji kestabilan 3x**, lalu bagikan laporan lengkap. Tombol ini mengulang
forward tiga kali dalam proses yang sama, dengan tensor input baru, tanpa
menghapus cache. Semua percobaan dipertahankan; PASS memerlukan tiga dual-parity
PASS. Guard footprint atau kesalahan persiapan menghentikan percobaan berikutnya.
Ini bukan uji tiga cold launch. Workspace Linux tidak menyediakan Apple SDK.

**Hasil terbaru: [uji tiga kali PASS 3/3 pada iPhone](../reviews/coreai-iphone17pro-sharded-denoiser-stability/README.md).**
Setiap percobaan menyelesaikan 14 bagian dan lolos kedua referensi; waktu total
4,68 / 3,46 / 2,04 detik. Ada lonjakan run+copy sekitar satu detik pada pasangan
yang berbeda antar percobaan. Footprint proses maksimum tersampel naik
35,72 → 60,35 → 68,35 MiB. Belum ada bukti plateau jangka panjang atau penyebab
kenaikan; angka tersebut bukan seluruh RAM model. Tidak perlu mengulang uji yang
sama atau menyiapkan bobot ulang untuk mengesahkan hasil ini.

Gate berikutnya sudah disiapkan: [loop denoising empat langkah](coreai-denoiser-loop.md)
memakai latent hasil sebelumnya dan jadwal/update ACE yang dipin. Referensi host
lolos pembandingan FP32; hasil CoreAI di iPhone masih menunggu. Referensi kecil
ikut GitHub, sehingga cukup pull dan build tanpa ekspor model ulang, lalu tekan
**Uji loop denoising 4 langkah**. Conditioning piano/teks dan waveform menyusul.

## Cakupan dan pembandingan

Input tetap noise `[1,25,64]`, context `[1,25,128]`, condition `[1,64,2048]`,
timestep `[1] = 0,5` dan timestep_r `[1] = 0,25`, semuanya Float32 sintetis.
Proyeksi awal menghasilkan hidden `[1,13,2048]`, conditioning terproyeksi,
modulasi timestep, embedding waktu akhir, RoPE dan mask. Pasangan layer memakai
hasil aktual bagian sebelumnya. Proyeksi akhir mengembalikan velocity `[1,25,64]`,
termasuk padding 25→26 dan crop kembali ke 25 frame.

Bobot disimpan FP16 dan perhitungan tetap FP32. Semua parameter diperiksa masuk
tepat satu partisi. [Ekspor semua bagian berhasil di Linux](../reviews/coreai-sharded-denoiser-export/README.md),
dan hasil rangkaian lolos pembandingan dengan model utuh serta bobot FP32 asli.
Toleransi tetap `abs(error) <= 1e-4 + 1e-3 * abs(reference)`, normalized RMSE <= `1e-3`.

Probe iPhone awal menjalankan **satu forward penuh**, dengan dua pembandingan
akhir terhadap referensi partisi dan FP32 asli. Laporan mencatat bagian yang
selesai, load/run+copy per bagian, waktu total dan footprint proses tersampel
tiap 10 ms. Uji berikutnya membuktikan tiga pengulangan kasus sintetis yang sama
dalam satu proses. Belum ada loop diffusion/CFG, conditioning dari piano/teks,
atau waveform instrumen.
