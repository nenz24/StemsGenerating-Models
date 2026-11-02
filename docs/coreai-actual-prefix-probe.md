# Diagnosis tiga langkah pertama pada input aktual

Pengujian pool [K=12](../reviews/coreai-iphone17pro-interleaving-k12/README.md)
selesai tanpa mereproduksi perlambatan berulang besar. Fixture pasangan tidak
memakai front/tail/APG atau feedback. Mode berikutnya mengukur **prefix aktual**
dari pipeline 10,24 detik agar hotspot per-aset dapat diidentifikasi.

## Kontrak

- Input harus Files import profil 10,24 detik dan source CPU reuse aktif.
  Rekaman asli, PCM tanpa normalisasi, preset dan fixture noise tetap.
- Encoder, Qwen dan conditioning lengkap berjalan seperti jalur generasi.
  Sampling memakai tiga langkah pertama dari schedule **50 langkah yang sama**,
  dengan latent/momentum feedback aktual. Tidak ada schedule tiga langkah baru.
- Step 1 memakai scope tunggal per aset; step 2 mulai mempertahankan maksimal
  15 handle. Target 45 calls, 30 opens, 30 binds, 15 reuse hits. Satu targeted
  bind-repair yang sudah ada tetap boleh menambah satu open, tanpa replay.
- Setiap step mencatat timing 14 shard dan APG: open/bind/validation,
  backend-await, detached CPU copy, other, thermal; total per-aset dan sampled
  footprint proses tetap dilaporkan. Perhitungan/weights/shape/binding tidak berubah.
- BF16 conditioning/DiT, FP32 compute dan VAE, abs 1e-4/rel 1e-3/normalized
  RMSE 1e-3, APG/guidance dan guard tetap. Jalur produksi default tetap 50 langkah.
- Mode berhenti setelah step 3 dan melepas pool; decoder tidak dijalankan,
  tidak ada WAV baru. Tombol diagnostik mempertahankan tautan/audio lama.
  Berhasil 3/3 prefix tetap **3/50**, bukan generasi lengkap atau musical PASS.
- Input impor tidak memiliki referensi host cocok: output checks 0/0,
  numerical parity **NOT EVALUATED**. Pemeriksaan conditioning tetap memiliki
  lingkup sendiri. Hasil sweep komponen yang gagal tetap NOT PASSED.

## Jalankan tanpa ekspor ulang

Tutup Xcode, dari checkout Mac yang sudah memiliki bundle generasi panjang:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_preview_window.py --seconds 10.24 \
  --activate-only --diagnostic-trajectory &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Flag trajectory ini untuk bundle baseline yang sudah berstatus diagnostic;
bundle strict biasa memakai aktivasi tanpa flag tersebut. Tidak ada ekspor
ulang atau penghapusan cache. Bundle probe pasangan tetap ada di Mac;
selector build kini memilih source/conditioning/preview generasi panjang.

Pilih **CoreAIProbe Reuse**, Release, iPhone. Cmd+R. Tunggu kondisi thermal
nominal, pilih rekaman piano yang sama dengan baseline yang diterima, lalu
tekan **Uji 3 langkah aktual (tanpa WAV)** sekali. Jangan tekan Generate bass
untuk percobaan ini. Bagikan laporan dari bagian hasil diagnostik, bukan
laporan WAV lama. Header harus memuat `actual-prefix-bounded-v1`.

Progres sampling live pendek, breadcrumb native terpisah di
`last-sampling-diagnostic-phase.txt`, dan laporan lengkap dicheckpoint setelah
setiap step selesai serta pada hasil/error tertangkap. Crash dapat kehilangan
detail step berjalan. Guard 2000 MiB tidak menjamin transient peak native.
Biarkan app foreground; catat kegagalan tanpa melakukan retry berulang.

Bandingkan step 1 (bukan guaranteed cold) dengan step 2 dan 3 untuk tiap aset.
Apabila front/tail/APG mendominasi, fokuskan eksperimen pada aset itu; apabila
pasangan layer melambat hanya di actual input, isolasi input/lifecycle tersebut.
Apabila seluruh prefix cepat, jangan mengklaim perbaikan 50 langkah: fase lanjut
belum diukur. Backend-await tetap API wall time, bukan kernel time.

## Validasi implementasi

17 regresi host lulus: 11 integrity/activation/staging dan enam kontrak preview
termasuk urutan 24 layer, guidance, schedule, finite gate dan preservasi output.
Pemeriksaan whitespace lulus. Pemeriksaan sumber memastikan default 50 langkah
dan branch diagnostik tanpa decoder/WAV. Linux tidak memiliki Apple SDK/Swift
compiler: build baru, prefix aktual dan accounting perangkat belum diverifikasi.
