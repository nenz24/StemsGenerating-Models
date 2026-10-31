# Hasil CUDA FP32 pertama yang valid

Feedback pengguna pada 1 November 2026 (Asia/Jakarta): `cuda_listen.wav` lebih
bersih dari glitch, tetapi suara bass terlalu dalam dan berbeda dari chunk 0.
Pengguna memilih target berikutnya: bass elektrik petikan yang jelas dan natural.

Paket `cuda_result.zip` memuat B 30 detik, baseline MPS B, hasil CUDA, settings,
session dan latent capture. Chunk 0 yang disukai tidak ada dalam paket ini,
sehingga belum dapat dibandingkan langsung secara spektral atau timbre.

## Verifikasi paket

- Report berstatus `generated`: Tesla T4, PyTorch 2.10.0+cu128, CUDA 12.8.
- Base, seed 0, CFG 7, 50 langkah, task `lego`, audio-only conditioning.
- DiT dilaporkan FP32; 12 panggilan VAE yang ditangkap memakai CUDA FP32 dan
  seluruhnya returned. Output VAE 48 kHz diekspor sebagai pratinjau 32 kHz.
- Array latent `[1, 750, 64]` seluruhnya finite dan hash-nya sesuai capture.
- Kedua pratinjau stereo berdurasi 30 detik dengan gain ekspor yang sama, 0.7.

Ini bukti satu keluaran CUDA FP32 valid beserta penilaian pengguna, bukan bukti
bahwa semua glitch hilang atau satu bug MPS sudah teridentifikasi. RNG, encoder
dan dtype jalur VAE berbeda dari run MPS.

## Estimasi spektrum pratinjau

Welch, window 16.384 sampel pada 32 kHz, power kedua kanal dirata-ratakan.
Persentase adalah proporsi energi spektrum estimasi, bukan deteksi pitch/f0
atau ukuran otomatis mutu musik.

| Rentang | CUDA FP32 | MPS B |
| --- | ---: | ---: |
| <40 Hz | 0.31% | 22.50% |
| 40–80 Hz | 68.07% | 59.53% |
| 80–150 Hz | 29.14% | 16.62% |
| 150–300 Hz | 2.13% | 0.81% |
| 300–1.000 Hz | 0.34% | 0.40% |
| >1.000 Hz | 0.01% | 0.13% |

Pada CUDA, sekitar 97.52% energi berada di bawah 150 Hz. Energi sangat rendah
di bawah 40 Hz justru lebih kecil daripada MPS, tetapi bunyinya tetap sangat
terkonsentrasi pada wilayah rendah. Ini sesuai feedback timbre yang terlalu
dalam. Percobaan berikutnya mengarahkan artikulasi petikan dan harmonik melalui
conditioning model.

Caption saat ini hanya berbunyi `Isolated bass accompaniment following the
source harmony, timing and dynamics.` Ia belum menentukan bass elektrik,
artikulasi petikan atau harmonik yang diinginkan. Langkah berikutnya adalah
[satu kandidat caption bass elektrik](bass-tone-colab.md), sambil mempertahankan
baseline CUDA FP32 dan seluruh setting lainnya. Ini uji pengaruh caption,
bukan klaim bahwa caption saja pasti menyelesaikan masalah timbre.
