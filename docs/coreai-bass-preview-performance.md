# Uji performa preview bass CoreAI

Dua laporan perangkat kini tersedia untuk klip dan waveform yang sama:

| Pengukuran | Run awal | Run berikutnya |
| --- | ---: | ---: |
| Durasi audio | 5,12 s | 5,12 s |
| Waktu pipeline | 85,95 s | 755,29 s (12 menit 35 detik) |
| Sampled maximum proses app | 417,19 MiB | 436,47 MiB |
| Sampling | 50/50 | 50/50 |
| Dua waveform vs referensi Mac | PASS | PASS |

[Laporan awal](../reviews/coreai-iphone17pro-bass-preview-cpu-complete/generation-report.json)
dan [laporan berikutnya](../reviews/coreai-iphone17pro-bass-preview-repeat/summary.json)
bukan controlled cold-cache benchmark. Sampled process footprint bukan exact
peak atau total model/device RAM. Laporan kedua mengandung SHA-256 WAV yang
persis sama dengan baseline uji dengar yang disetujui, tetapi **build flags
masih unrecorded**. Karena itu Release belum bisa dikonfirmasi atau dituduh
sebagai penyebab perlambatan.

Pada run kedua, backend-await memakan **730,38 s / 96,7% total pipeline**;
model-open dan bind gabungan hanya 2,09 s, copy output 6,45 s. Panggilan backend
ulang tetap mahal. Interval ini dapat memuat deferred work/initialization dan
execution; bukan bukti kernel time atau kompilasi ulang tertentu.

## Satu pengukuran berikutnya: reuse model dan fungsi

Tutup Xcode, lalu dari repository:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Di toolbar pilih **CoreAIProbe Reuse** dan iPhone yang sama, lalu Cmd+R.
Scheme ini memakai Release dan mengaktifkan `COREAI_PREVIEW_REUSE=1` hanya
untuk peluncuran tersebut. Saat membuka app dari ikon sendiri, environment
scheme tidak diasumsikan tersedia; mode biasa tetap default.

Tekan **Generate bass 5,12 detik** sekali. Laporan baru harus mencatat:

```text
App build configuration: Release
Swift app optimization: -O
Swift DEBUG compilation condition: OFF
Sampling handle reuse requested: true
Sampling retained-stage scope: 15 opens, 15 binds, 735 hits, 750 calls
```

Setelah selesai, kirim **Bagikan laporan generasi**. Hash WAV dicatat untuk
memeriksa kesamaan audio. Tidak perlu preparation, ekspor model/referensi atau
kompilasi AOT. Hasil lama tetap tersimpan; keluaran run baru tidak menimpanya.

Empat belas shard DiT dan satu APG model/fungsi ditahan lintas 50 langkah,
dimuat secara lazy. Pool dibatasi **15 entri**, lalu dilepas sebelum decoder.
Encoder piano dan Qwen tetap selesai sebelum pool sampling dibuat terisi.
Ini memakai jalur retained-stage yang pernah
[PASS pada uji sintetis empat langkah](coreai-retained-loop.md), dengan scope
real-conditioning/batch yang sekarang masih perlu diuji.

Guard proses 2000 MiB sebelum stage dan sesudah bind tetap aktif. Retention
bisa meningkatkan memori; guard itu tidak menjamin transient compiler peak
atau native eviction. Jika gagal, pool Swift dibersihkan dan laporan dipertahankan;
tidak ada fallback/retry diam-diam. AOT/default yang pernah crash Metal tidak
dipakai untuk eksperimen reuse CPU source ini.

## Konfigurasi dan pembanding

- **CoreAIProbe**: Debug, satu scope per aset seperti baseline eksekusi.
- **CoreAIProbe Performance**: Release, satu scope per aset; kontrol tanpa reuse.
- **CoreAIProbe Reuse**: Release, reuse 15 sampling handle per satu generasi.

Pencatatan build yang dulu memakai custom `INFOPLIST_KEY_*` kini menggunakan
base `CoreAIProbe-Info.plist` yang nyata dengan ekspansi build settings.
Compile-time DEBUG field menjadi cross-check tambahan. Scheme/plist dapat
diaudit di host; Apple build dan runtime berikutnya tetap perlu divalidasi.
Tidak ada perubahan model, guidance 7, 50 langkah APG, dtype, referensi atau
batas numerik. Satu run berikutnya menguji hipotesis lifecycle model/fungsi;
cache, kondisi termal dan native residency tetap belum terisolasi.

Release pertama memakai produk build sendiri dan menyalin resource aktif;
model/cache source dipakai ulang, probe lama/AOT tidak ditambahkan ke app.
Debugger LLDB masih tersedia. Waktu API per aset, pertama/ulang, total pipeline,
serta sampled footprint dicatat. Jangan mengekstrapolasi klip ini secara linear
ke 90 detik/full song, atau menjanjikan speedup sebelum pengukuran.

## Status numerik dan musikal

Kedua run mengevaluasi sepuluh output: delapan PASS, dua latent langkah ke-50
NOT PASSED. Dua waveform terhadap referensi Mac PASS. Host
`bounded_diagnostic` menjaga overall strict gate NOT PASSED. Baseline uji dengar
pendek diterima pengguna; numerical baseline terpisah. Wording laporan hanya
memperjelas perbedaan itu. Seluruh gate tersebut tetap dipakai pada uji reuse.
