# Persiapan lagu penuh: source dan jadwal, tanpa menjalankan model

Target tetap satu lagu penuh dengan bass, drum dan gitar di iPhone. Helper ini
menyiapkan input penuh dan rencana pembagian workload; ia belum menghasilkan
stem atau mengaktifkan profil iPhone. Seluruh gate baseline tetap independen.
Komponen yang lolos tidak mengubah trajectory yang belum lolos menjadi PASS.

## Source yang konsisten dengan frontend acuan

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_full_song.py --execute \
  --choose-source \
  --out .runtime-cache/full-song-source-v1
```

Pada Mac, pilih audio asli dari dialog yang muncul; tidak perlu mengetik path
contoh. `--source` juga menerima lokasi berkas sebenarnya untuk penggunaan CLI.
`--baseline-prefix` menerima lokasi WAV acuan yang sudah ada, bersifat opsional
dan hanya dipakai bila file berasal dari potongan
awal lagu yang sama. Jika prefix berbeda, helper menyimpan report gagal dan
berhenti sebelum membuat plan. Jangan mengganti acuan atau melonggarkan hash.
Output lama selalu dipertahankan. Source/baseline/models tidak ditimpa.

FFmpeg men-decode stereo32kHz dengan gain0.7 sekali. Konversi PCM16 memakai
nearest-even eksplisit seperti frontend30s. Streaming decode/quantization tidak
menahan seluruh waveform dalam array. `source32.wav`, hash dan jumlah frame
aktual disimpan. Tidak ada DiT, VAE, Qwen, RNG baru, export atau aktivasi model.
Untuk metadata plan saja tanpa dependency/model/file I/O:

```bash
python3 -S scripts/prepare_coreai_full_song.py --frames32 5760000
```

Jumlah frame acuan adalah hasil decode PCM, bukan estimasi durasi kontainer MP3.
Output WAV/report bersifat privat; jangan commit audio atau hash input pengguna.

## Jadwal yang diusulkan

Setiap run tetap750 latent frame/30 detik,50 Euler/APG, BF16 learned storage dan
FP32 compute/I/O/VAE, serta abs1e-4/rel1e-3/nRMSE1e-3. Window ditempatkan pada
grid40ms, overlap minimal8 detik; jarak start dibagi rata agar tidak ada hop
terakhir yang sangat pendek. Durasi output dipangkas tepat ke jumlah sampel48kHz
yang diturunkan dari source32. Padding terakhir dicatat pada level latent dan
sampel. Ini rencana future normalized overlap-add, belum bukti seam musik bersih.

Instrumen diproses satu per satu dengan bobot sampler bersama dan conditioning
yang berbeda. Maksimal15 sampling handle direncanakan, dilepas sebelum satu
decoder256 memakai enam window lokal yang sudah diuji. Guard2000MiB dan sampled
peak tetap diperlukan; jumlah handle tidak membuktikan total native residency.
Angka call dalam plan dihitung dari geometry, bukan pengukuran performa baru.

Noise global per instrumen dan epsilon source harus disimpan sebagai array
aktual, lalu overlap menggunakan slice timeline yang konsisten. Seed saja tidak
membuktikan kesamaan lintas CPU. Numerical comparison dilakukan pada raw output;
gain dengar diterapkan bersama setelah peak seluruh lagu diketahui, bukan gain
otomatis per-window. Tidak ada EQ/limiter atau klaim DSP menyelesaikan drift.

## Gate yang belum diselesaikan

Plan selalu `ready_for_native_execution: false`. Source encoder native bounded
untuk input berubah, conditioning bass/gitar/drum yang cocok127 token, provenance
noise/epsilon global, assembler overlap-add, uji dengar sambungan dan full-song
resource/thermal/latency perlu diselesaikan. Sampler full50-step tetap dinilai
pada toleransi asli; hasil diagnosis sebelumnya tidak menerima numerical gate.

[Panduan performa](coreai-bass-preview-performance.md) dan
[precision A/B/A](../reviews/coreai-iphone17pro-precision-aba/README.md)
tetap berlaku: jangan memperbesar sampler ke FP32 berdasarkan hasil komponen
atau mengekstrapolasi waktu/memori klip pendek secara linear ke lagu penuh.
Helper tidak memberi estimasi waktu dan tidak mengubah scheme/model/baseline.
