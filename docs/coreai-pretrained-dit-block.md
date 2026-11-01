# Uji blok generator pretrained pada iPhone

Encoder posterior mean dan decoder VAE sudah lulus parity secara terpisah di
iPhone. Tahap ini memeriksa satu blok **diffusion transformer (DiT)** dari model
Base yang dipakai pada baseline CUDA. Blok ini mencakup self-attention,
cross-attention, RoPE, RMSNorm, modulasi timestep dan MLP dengan bobot asli.

[Hasil iPhone sudah PASS](../reviews/coreai-iphone17pro-pretrained-dit-block/README.md):
tiga run cocok, rata-rata dua run hangat 4,51 ms. Tahap berikutnya adalah
[dua blok dengan konteks lebih panjang](coreai-dit-context.md), yang menguji
mask lokal dan penuh di atas radius 128 token. Langkah di bawah tetap tersedia
untuk mengulang baseline satu blok pendek.

## Jalankan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_dit_block.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip memakai kembali environment CoreAI yang sama, memasang dependency generator
yang dipin, lalu mencari checkpoint Base di vendor/cache yang dikenal.
**Jika belum tersedia, unduhan checkpoint Base adalah 4,79 GB**, berbeda dari
VAE 337 MB. Seluruh ukuran dan SHA-256 publisher wajib cocok sebelum bobot
digunakan. Checkpoint dapat dipakai ulang setelah unduhan pertama.
File source model dan config juga diverifikasi terhadap manifest publisher.

Hanya tensor `decoder.layers.0.*` yang dimuat, sekitar 62,9 juta parameter.
Hasil ekspor FP32 sekitar 252 MB disimpan terpisah di
`prototypes/ios27-coreai/ModelCandidates/dit-block-pretrained/`.
Ketiga paket codec tetap disimpan. Resource pengujian yang terpasang bersama
akan membesarkan ukuran aplikasi; itu bukan ukuran model produk akhir atau
pengukuran RAM. Checkpoint, staging dan native asset besar diabaikan Git.

Paket yang sudah ada diverifikasi saat skrip diulang; tidak ditimpa otomatis.
Jika checkpoint berada di direktori lain, jalankan exporter dengan
`--checkpoint /path/model.safetensors` melalui Python environment tersebut.
Hash publisher tetap wajib cocok. Tidak memerlukan GPU CUDA atau training baru.

## Jalankan di iPhone

Pilih iPhone 17 Pro → **Cmd+R** → **Uji blok generator**. Jalankan tombol ini
terlebih dahulu setelah membuka ulang aplikasi agar footprint awal lebih mudah
dibaca. Cache CoreAI tetap dapat memengaruhi load. Bagikan laporan lengkap.
Profil pendek telah dijalankan pengguna di iPhone. Perubahan Swift terbaru untuk
profil konteks panjang belum dikompilasi dengan Apple SDK di workspace Linux.

## Cakupan uji

Tujuh tensor Float32 masuk melalui fungsi `main`:

| Input | Shape | Isi |
| --- | --- | --- |
| hidden | [1, 16, 2048] | Aktivasi tersembunyi sintetis |
| condition | [1, 8, 2048] | Conditioning sintetis |
| temb | [1, 6, 2048] | Modulasi timestep sintetis |
| rope_cos / rope_sin | [1, 16, 128] | RoPE posisi 0–15 |
| self_mask | [1, 1, 16, 16] | Mask aditif, satu key dimask |
| cross_mask | [1, 1, 16, 8] | Mask aditif, satu key dimask |

Output `hidden_out` Float32 `[1, 16, 2048]` dibandingkan terhadap PyTorch CPU
dengan eager attention publisher. Semua input tetap menjadi argumen runtime;
conditioning dan timestep tidak dibekukan menjadi konstanta model. Cache KV
dimatikan. Panjang 16 token ini tidak dinyatakan sebagai durasi audio dan belum
menguji perilaku sliding window di atas 128 token, semua layer atau semua mask.

Toleransi absolute `1e-4`, relative `1e-3`, normalized RMSE `1e-3`, tiga run,
serta pengukuran latensi/footprint memakai aturan probe codec sebelumnya.
CPU-only diminta; hardware aktual tidak diukur. Input sintetis ini belum
berasal dari piano, teks instrumen atau output conditioning encoder.

Kelulusan blok belum membuktikan bahwa generator lengkap muat atau menghasilkan
musik di iPhone. Checkpoint Base berisi sekitar 2,39 miliar elemen tensor;
penyimpanan dense FP32 seluruhnya sekitar 9,58 GB, sebelum VAE dan aktivasi.
Angka itu adalah inventaris bobot, bukan peak RAM terukur. Export/generasi penuh,
conditioning, loop sampling, precision/kuantisasi, memori dan mutu musik tetap
memerlukan pengujian tersendiri.
