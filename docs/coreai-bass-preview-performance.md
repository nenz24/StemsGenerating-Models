# Performa preview bass CoreAI

Tiga laporan perangkat kini tersedia untuk audio yang sama:

| Pengukuran | Awal, satu scope | Ulang, satu scope | Release + reuse |
| --- | ---: | ---: | ---: |
| Durasi audio | 5,12 s | 5,12 s | 5,12 s |
| Waktu pipeline | 85,95 s | 755,29 s | **59,24 s** |
| Sampled maximum proses app | 417,19 MiB | 436,47 MiB | **450,42 MiB** |
| Sampling | 50/50 | 50/50 | 50/50 |
| Build flags tercatat | Tidak tersedia | Unrecorded | **Release / -O / DEBUG OFF** |
| Dua waveform vs referensi Mac | PASS | PASS | PASS |

[Laporan reuse lengkap](../reviews/coreai-iphone17pro-bass-preview-reuse/README.md)
mengonfirmasi 15 opens, 15 binds, 735 reuse hits dan 750 sampling calls, lalu
pool dilepas sebelum decoder. Hash dua WAV persis sama dengan baseline uji
dengar yang disetujui. Delapan output comparison PASS, dua latent terakhir
NOT PASSED; host `bounded_diagnostic` menjaga overall gate NOT PASSED.

Waktu 59,24 s sekitar 12,75 kali lebih cepat daripada run 755,29 s yang diamati,
dan 1,45 kali daripada run awal 85,95 s. Ini perbandingan antar-run, bukan
controlled cold-cache/thermal benchmark atau bukti pengaruh Release terpisah.
Untuk audio 5,12 s, rasio wall/audio masih 11,57: belum real-time.

Backend-await run reuse adalah 37,14 s; copy output 7,71 s. Step pertama
15,07 s, langkah 2–50 rata-rata 651,59 ms. Backend-await adalah interval API,
bukan kernel time. Penyebab internal seluruh perbedaan antar-run belum
terisolasi. Angka memori adalah sampled app process, bukan exact peak atau
total model/device RAM, dan bukan jaminan skalabilitas full song.

## Jalur prototype yang dipilih

**Profil 10,24 detik sudah menyelesaikan satu run**, dengan [laporan dan audit
WAV](../reviews/coreai-iphone17pro-audio-import-1024/README.md): 772,415 detik,
footprint maksimum tersampel 1.276,660 MiB, 30 opens/30 binds/720 hits/750 calls.
Langkah pertama memakai scope tunggal per aset, lalu reuse mulai langkah kedua.
Langkah 3–50 tetap rata-rata 15,191 detik; backend-await 744,691 detik atau
96,41% total pipeline. Reuse handles tercatat tetapi tidak membuktikan percepatan
backend pada profil ini. Penyebab internal, thermal dan cache state belum
diisolasi. Klip diterima melalui uji dengar, bukan sebagai anggaran performa.
Recovery cache 0/1 berarti jalur repair belum teruji oleh run yang selesai ini.
[Uji pasangan layer BF16 versus FP32](coreai-precision-probe.md) sekarang
disiapkan untuk mengisolasi pengaruh penyimpanan bobot dengan nilai dan input
yang sama, tanpa mengulang 50 langkah atau mengubah baseline.

**Source CPU memakai reuse secara default saat fresh launch tanpa override**,
termasuk membuka app Release dari ikon iPhone. AOT yang sengaja disertakan
masih memakai mode AOT satu-scope; hasil CPU ini tidak memvalidasinya. Default
baru hanya mengubah pemilihan lifecycle ke jalur yang sudah diuji pada klip ini.
Model, guidance 7, 50 langkah APG, dtype, referensi dan batas numerik tetap sama.

Pool lazy berisi maksimal 14 shard DiT + satu APG model/fungsi, dilepas sebelum
decoder dan saat gagal. Encoder piano dan Qwen tidak masuk pool. Guard proses
2000 MiB sebelum stage dan sesudah bind tetap aktif. Retention dapat menaikkan
live memory; guard tidak menjamin transient compiler peak/native eviction.
Untuk 5,12 detik tidak ada retry baru. Profil 10,24 detik mencatat satu budget
recovery khusus BNNS cache bind sebelum inference; [syarat dan accounting](coreai-bass-preview-1024.md)
tetap eksplisit. Ini tidak mengulang langkah atau mengganti metode sampling.

## Scheme dan penggunaan berikutnya

- **CoreAIProbe Reuse**: Release, `COREAI_PREVIEW_REUSE=1`; jalur rekomendasi.
- **CoreAIProbe Performance**: Release, override `0`; kontrol satu-scope.
- **CoreAIProbe**: Debug, override `0`; kontrol satu-scope.

Override hanya berlaku untuk proses yang diluncurkan scheme tersebut. Fresh
launch tanpa override memilih reuse untuk source CPU. Membuka ikon saat proses
lama masih hidup tidak mengubah environment proses itu.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih **CoreAIProbe Reuse** untuk pengembangan berikutnya. Tidak perlu generate
ulang untuk menyimpan hasil yang sudah diterima ini. Bila mengukur perubahan
selanjutnya, gunakan input yang sama dan **Bagikan laporan generasi**, termasuk
hash WAV, flag build dan ringkasan API timing. Base `CoreAIProbe-Info.plist` dan
DEBUG fallback kini terbukti bekerja pada report perangkat. SDK/build untuk
pemilihan default terbaru masih perlu verifikasi Mac; jalur reuse sendiri
sudah teramati selesai.

Baseline musikal 5,12 detik dan baseline runtime pendek ini tidak menggantikan
baseline CUDA 90 detik. Cold starts/repeatability, input rekaman baru langsung
di iPhone, durasi lebih panjang, instrumen lain dan full song tetap belum
tercakup. Jangan mengekstrapolasi waktu/memori klip ini secara linear ke lagu
penuh atau mengklaim numerical PASS dari persetujuan pendengaran.
