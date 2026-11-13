# Decode pembanding dari latent bass/gitar yang sama

Helper ini memeriksa raw audio iPhone terhadap CPU FP32 VAE pada final latent
native yang sama. Ia tidak menjalankan DiT, Qwen, source encoder atau generation
baru. Semua waveform diperiksa sebelum gain pada abs1e-4/rel1e-3/nRMSE1e-3.
Baseline BF16, preset,50 langkah APG dan file hasil sebelumnya dipertahankan.

## Buka pembanding lokal dengan posisi waktu yang terlihat

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/audit_coreai_instrument_transition_decode.py --execute &&
open .runtime-cache/instrument-transition-decode-v1/listen.html
```

Input default adalah ZIP yang sudah dibuat:
`.runtime-cache/bass-guitar-transition-review-v1.zip`. Checkpoint VAE wajib
tersedia lokal dengan checksum asli; tidak ada download atau export native.
`--archive` dan `--vae-checkpoint` menerima file existing lain. Output existing
dipertahankan; gunakan `--out` baru jika perlu. Tanpa `--execute`, helper hanya
mencetak plan dengan stdlib dan tidak membaca input/memuat model/menulis output.

Halaman lokal menyediakan pasangan **Dari iPhone** dan **Decode CPU penuh**
untuk bass serta gitar. Semuanya cuplikan detik22–30 pada gain per-run asli;
posisi lagu tampil selama playback dan saat seek. Tidak ada filter, crossfade,
limiter atau normalisasi tambahan. Gunakan output audio/volume yang sama untuk
kedua versi. Halaman dan semua audio tetap lokal; tidak memakai jaringan.

Pemeriksaan ini membantu memisahkan hasil audio yang memang tersimpan dari
masalah pada jalur playback app/perangkat. Jika glitch masih terdengar pada
kedua salinan di pemutar yang sama, fokus berikutnya mencakup kualitas hasil
latent/generasi/conditioning atau sifat decoder asli, bukan otomatis masalah
sambungan window. Jika kedua salinan bersih di luar app, jalur playback app/
perangkat perlu ditelusuri. Kelulusan numerik tidak menilai glitch secara dengar.

## Kontrol dan scope

Mode default menjalankan dua full750 decoder CPU, satu per instrumen. Ini
diagnosis host dan bukan pengganti decoder berjendela iPhone yang telah menjaga
memori. Jangan menafsirkan peak/waktu host sebagai budget iPhone atau estimasi
lagu penuh. Tambahkan `--include-window-control` untuk mereproduksi enam tile256
per instrumen selain decode penuh, sehingga total14 panggilan CPU decoder.

ZIP harus berisi tepat21 entry yang ditentukan, termasuk16 array finite dengan
shape/ukuran/SHA256 yang benar, dua report dan dua cuplikan. Metadata diikat ke
report, final latent harus identik checkpoint50, dan raw audio pada gain FP32
asli harus mereproduksi WAV iPhone dengan SHA256 persis sama sebelum model
dimuat. Cuplikan asli juga direkonstruksi dari raw dan wajib identik hash-nya.

Tiap pasangan waveform memiliki13 perbandingan: keseluruhan,6 core,5 seam
±80ms dan8 detik terakhir. Global nRMSE kecil tidak mengesampingkan satu
pelanggaran pointwise. Kegagalan finite dicatat sebagai NOT PASSED tanpa
mengubah batas; data nonfinite/invalid menghentikan run. Input invalid ditolak
sebelum inference; kegagalan selama eksekusi disimpan dalam report gagal.
Hasil yang belum lengkap tidak diterima. Tidak ada gold latent/momentum reset.

`report.json`, WAV dan `listen.html` bersifat privat. Jangan commit input ZIP,
output audio, tensor atau report perangkat. Tidak ada penggantian gate sampler
drum sebelumnya, numerical parity generation bass/gitar, atau penerimaan musik
lagu penuh dari hasil decode ini.
