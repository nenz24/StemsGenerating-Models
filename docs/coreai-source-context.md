# Tahap 2a: rekaman piano menjadi audio-context

Jalur audio-context sekarang diimplementasikan dan diekspor menggunakan checkpoint
VAE asli serta potongan nyata 5,12 detik dari `piano.mp3`. Ini bagian audio dari
conditioning: posterior mean/std → sampled source latent → context dengan mask
`lego`. Embedding teks, ConditionEncoder, sampling bass dan WAV pengiring belum
disambungkan dalam perubahan ini.

**Uji iPhone sudah PASS:** [laporan perangkat](../reviews/coreai-iphone17pro-source-context/README.md)
memuat 3/3 run dan 12/12 pembandingan output. Rata-rata backend 1,037 detik;
footprint aplikasi maksimum tersampel 398,21 MiB, bukan seluruh RAM model atau
peak eksak. Bagian audio tahap 2 selesai untuk rekaman/fixture ini. Tidak perlu
mengulang uji ini sekarang; lanjut conditioning teks/instrumen dan ConditionEncoder.

## Jalankan di Mac dan iPhone

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_source_context.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih rekaman piano saat dialog file muncul. Skrip mengambil 5,12 detik pertama;
rekaman harus cukup panjang dan tidak diam. Atau tentukan path secara langsung:

```bash
bash scripts/prepare_coreai_source_context.sh "/path/to/piano.mp3"
```

Jika `ffmpeg` belum tersedia, pasang dengan `brew install ffmpeg`, kemudian
jalankan ulang. Environment Python dan checkpoint VAE yang sudah ada dipakai
ulang. Skrip mengekspor **satu graph encoder-posterior baru**, sekitar 337 MB;
model DiT dan bundle encoder/decoder lama tidak ditimpa atau diekspor ulang.
Checkpoint VAE hanya diunduh jika tidak ditemukan dalam cache yang tervalidasi.

Build **Cmd+R** di Xcode, lalu tekan **Uji audio-context piano** dan bagikan
laporan lengkap. Targetnya:

```text
Real-recording audio-context parity: PASS (three runs, all four outputs)
Completed audio-context parity runs: 3/3
```

Ketiga run memakai model/fungsi yang sama dan fixture epsilon yang sama untuk
pembandingan numerik, total 12 pembandingan. Footprint diperiksa sebelum load
dan setiap call; guard 2000 MiB tetap bukan batas peak OS.

## Kontrak yang diikuti

Audit source dilakukan pada commit runtime CUDA yang dipin,
`ca1e85fe9430179831e6bc6be790c332190a3866`. Hash file terkait tercatat di
[`configs/coreai-lego-preprocessing.json`](../configs/coreai-lego-preprocessing.json).

- Handler memakai `latent_dist.sample()`, sehingga encoder mean saja belum
  mewakili jalur audio conditioning. Sampling Oobleck adalah
  `mean + (softplus(scale) + 1e-4) * epsilon`.
- Batch latents dipad ke minimal 128 frame. Klip 5,12 detik menghasilkan tepat
  128 frame pada 48 kHz, sehingga uji ini tidak memerlukan silence-padding.
- Untuk `lego`, tanpa audio-code hints dan full-range repaint, `is_covers` false
  dan source latent tetap utuh. Helper mask asli dijalankan pada host, bukan
  disimulasikan. Mask boolean diperluas ke 64 channel; context menggabungkan
  source 64 channel dan mask 64 channel.
- Seluruh parameter encoder dimuat strict dari checkpoint SHA-256 yang sama.
  Sampling dengan epsilon eksplisit cocok persis dengan distribusi Diffusers;
  folding weight norm juga diverifikasi persis sebelum native export.

Input native: audio `[1,2,245760]`, epsilon `[1,64,128]`, mask `[1,128,64]`.
Output: mean/std `[1,64,128]`, source `[1,128,64]`, context `[1,128,128]`.
Semua Float32, dengan batas numerik `abs(error) <= 1e-4 + 1e-3*abs(reference)`
dan normalized RMSE `<=1e-3`. Output ini belum bisa langsung dimasukkan ke probe
DiT lama yang hanya menerima 25 latent frame/64 condition token.

## Data dan batas pengujian

Audio asli dan semua tensor yang diturunkan darinya berada dalam bundle lokal
ignored `ModelCandidates/source-context/`; tidak ikut GitHub. Bundle lama yang
sudah ada diverifikasi dan dipertahankan. Rekaman/offset berbeda ditolak pada
reuse; gunakan exporter dengan `--out` ke direktori baru untuk persiapan lain.

MP3 didekode/resample oleh ffmpeg di Mac menjadi PCM 48 kHz stereo tanpa peak
normalization. **PCM yang sama** dipakai referensi host dan iPhone. Pengujian ini
belum memvalidasi kesamaan bit decode/resampler torchaudio atau AVFoundation,
belum menerima rekaman baru langsung di iPhone, dan tidak menyamakan RNG CPU
fixture dengan stream RNG CUDA pada run musikal lama. Epsilon tetap hanya
untuk membandingkan implementasi posterior yang sama.

[Ekspor host sudah berhasil](../reviews/coreai-source-context-export/README.md).
Build dan eksekusi Swift telah dilaporkan pengguna pada iPhone; workspace Linux
tidak memiliki Apple SDK untuk mereproduksinya. Tahap 2 berikutnya adalah embedding prompt/lyrics,
condition encoder dan mask aktual, lalu integrasi ke sampler/decoder pada tahap 3.
