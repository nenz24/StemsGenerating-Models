# Uji performa preview bass CoreAI

[Run iPhone yang lengkap](../reviews/coreai-iphone17pro-bass-preview-cpu-complete/generation-report.json)
menghasilkan audio **5,12 detik dalam 85,95 detik**, dengan policy CPU-only
source. Ini membuktikan eksekusi dan playback, tetapi kecepatan masih perlu
ditingkatkan sebelum durasi diperpanjang. Maximum proses yang tersampling
417,19 MiB bukan peak eksak atau total RAM perangkat.

| Bagian | Waktu run yang dilaporkan |
| --- | ---: |
| Encoder/context piano | 1,48 s |
| Qwen/ConditionEncoder | 10,30 s |
| 50 langkah sampling | 71,79 s |
| Sisa pipeline, termasuk decoder dan ekspor | 2,39 s |
| Total | 85,95 s |

Sampling pertama 14,72 s; langkah 2–50 rata-rata 1,165 s. Biaya buka/bind yang
kecil dan backend-await awal yang besar tidak cukup untuk mengidentifikasi
semua penyebab latency. Backend-await adalah interval API, bukan kernel time.
Laporan lama tidak merekam build flags. Build log sebelumnya memakai Debug
`-Onone`; cache, kondisi termal dan debugger juga belum dikontrol.

## Satu pengukuran berikutnya

Tutup Xcode, lalu dari repository:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Di toolbar Xcode pilih scheme **CoreAIProbe Performance** dan iPhone yang sama.
Scheme ini menjalankan **Release (`-O`)** pada target/app yang sama. Scheme
**CoreAIProbe** biasa tetap memakai Debug. Profil resource tetap `auto`, yaitu
source CPU-only; AOT tidak ikut app.

1. Cmd+R. Hasil lama dapat muncul sebagai audio tersimpan.
2. Tekan **Generate bass 5,12 detik** sekali untuk pengukuran baru.
3. Setelah selesai, kirim **Bagikan laporan generasi** serta WAV jika berubah.

Tidak perlu preparation model, penghitungan referensi atau kompilasi AOT.
Build Release pertama membuat produk build sendiri dan menyalin resource aktif
ke produk tersebut; source model/cache Mac tetap dipakai ulang. Output baru
tersimpan di folder UUID baru; baseline terdahulu tidak ditimpa. Debugger LLDB
scheme tetap tersedia, jadi ini belum benchmark tanpa debugger.

Laporan baru mencatat `App build configuration: Release`, `Swift app optimization: -O`,
WAV hashes, serta jumlah/waktu API per aset. Total panggilan pertama per nama
aset dipisahkan dari panggilan ulang, tanpa mengklaim cold/warm cache atau
retained-model execution. Ringkasan waktu ditambahkan setelah sampling;
progress native per-stage tetap sama. Data tersebut menentukan apakah biaya
utama berada di runtime backend, copy/validasi Swift, load/bind atau bagian di
luar aset. Release **tidak menjanjikan** speedup native atau memperbaiki AOT.

## Status numerik dan musikal

Laporan lengkap yang diterima menilai sepuluh output: delapan PASS, dua latent
langkah ke-50 NOT PASSED. Dua waveform terhadap referensi Mac PASS. Host
`bounded_diagnostic` juga mencegah label keseluruhan strict PASS. Pengukuran
baru mempertahankan semua batas dan status tersebut; jumlah langkah/preset
atau dtype tidak diturunkan untuk mendapat angka waktu yang lebih cepat.

[Baseline uji dengar 5,12 detik](../baselines/iphone-coreai-bass-preview-512-v1/README.md)
tetap diterima pengguna. Itu berbeda dari numerical baseline. Penggantian
wording laporan berikutnya hanya memperjelas perbedaan ini. Build scheme/parser
serta regresi host dapat diperiksa di Linux; build/latency/memori Release nyata
masih harus diukur di perangkat pengguna. Jangan mengekstrapolasi 85,95 s
secara linear untuk 90 detik atau full song.
