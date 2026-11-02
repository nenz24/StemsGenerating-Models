# Preview bass 10,24 detik dalam satu jendela

Profil ini memperpanjang input Files menjadi 10,24 detik dengan satu konteks
penuh dan satu lintasan 50 langkah APG. Hasil tidak dibuat dengan menyambung dua
generasi 5,12 detik. Jalur source CPU dan reuse model/fungsi tetap digunakan.
[Ekspor host aktual](../reviews/coreai-bass-preview-1024-export/README.md) menghasilkan
status `bounded_diagnostic`: normalized RMSE latent 0,000446 dan waveform
0,000373 berada di bawah 0,001, tetapi gate pointwise belum lolos.
Build Apple, pemakaian memori/waktu perangkat dan kualitas musikal profil baru
harus diukur pada iPhone; keberhasilan dua klip 5,12 detik tidak membuktikannya.

## Persiapan sekali di Mac

Tutup Xcode, lalu jalankan dari checkout:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_bass_preview_1024.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Picker di Mac meminta **file audio piano minimal 10,24 detik** untuk referensi
host. Pilih `piano.mp3` sebelumnya, bukan project Xcode. Path dapat diberikan
langsung bila lebih mudah:

```bash
bash scripts/prepare_coreai_bass_preview_1024.sh "/path/ke/piano.mp3"
```

Checkpoint yang ada dipakai ulang. Persiapan menghitung referensi sumber dan
conditioning dengan metadata durasi baru, dua lintasan 50 langkah dan waveform,
lalu mengekspor aset dengan shape baru. Proses ini lebih lama daripada build;
tunggu **Ready**. Menjalankan perintah lagi memverifikasi bundle yang sudah
lengkap. Bundle lama tidak ditimpa. Jika gate host gagal, tidak ada aktivasi
profil baru; jangan membangun seolah persiapannya sudah selesai.

Checkpoint dipakai kembali, tetapi aset untuk profil panjang diekspor ulang
dengan penyimpanan BF16. Aset FP16 lama tidak disalin sebagai BF16 hanya dengan
mengganti metadata. Reuse native hanya diizinkan untuk shape, binding dan dtype
yang identik. Seluruh conditioning tetap dibandingkan dengan implementasi asli.

## Jalankan di iPhone

Pilih scheme **CoreAIProbe Reuse**, target iPhone, Cmd+R. Antarmuka harus menyebut
**10,24 detik**, bukan 5,12. Kemudian:

1. Tekan **Pilih rekaman piano** dan pilih `piano.mp3` yang sudah diterima pada
   uji pendek. Input baru tidak perlu diproses ulang di Mac.
2. Dengarkan bagian piano untuk memeriksa cuplikan yang dipakai.
3. Tekan **Generate bass dari rekaman** dan tunggu **50/50 langkah** serta
   `WAV generated on device: YES`.
4. Dengarkan hasil, lalu bagikan laporan, info rekaman, WAV piano, bass dan mix.

Input diambil dari awal file, mono atau stereo, kemudian dikonversi di iPhone
menjadi Float32 stereo 48 kHz tanpa normalisasi. File terlalu pendek ditolak;
tidak ada padding diam. Original disalin tanpa diubah. Output/input preview
berdurasi tepat 491520 frame per kanal atau 10,24 detik. Tombol baseline memakai
PCM host untuk jendela aktif dan membandingkan referensi; mode Files memakai
PCM impor dan tidak membandingkannya dengan waveform host lama.

Biarkan app di foreground selama proses. Auto-lock dinonaktifkan saat app aktif
dan sedang bekerja, lalu dipulihkan saat selesai atau tidak aktif; ini bukan
implementasi generasi background.

Tidak ada perubahan gain mix: piano 0,78, bass 0,55, gain bersama bila peak
melebihi 0,98. Stem bass mentah tetap Float32. Kontrol level per track, lagu
panjang/chunking, offset cuplikan, prompt bebas dan RNG perangkat belum termasuk.

## Hanya satu profil masuk app

Bundle baru disimpan lokal dalam `ModelCandidates/source-context-1024`,
`bass-conditioning-1024` dan `bass-preview-1024`. Setelah ketiganya lolos
verifikasi, driver menulis selector lokal `active-preview.json` secara atomik.
Build `auto` hanya menyalin tiga bundle untuk jendela aktif; aset 5,12 detik,
AOT dan probe lama tidak ikut pada profil 10,24 detik. Sumber/checkpoint di Mac
serta baseline yang sudah diterima tetap disimpan. Selector dan bobot diabaikan
Git. CoreAI dapat menyimpan cache turunan tersendiri di perangkat; ini bukan
jaminan ukuran keseluruhan data aplikasi atau RAM native.

Untuk kembali ke bundle 5,12 detik yang sudah ada tanpa export ulang:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_preview_window.py --seconds 5.12 --activate-only
```

Lalu Stop/Cmd+R. Untuk mengaktifkan lagi 10,24 detik yang sudah lengkap, gunakan
perintah yang sama dengan `--seconds 10.24`. Menghapus app tidak diperlukan.

## Kontrak dan lingkup referensi

| Bagian | Profil 5,12 detik | Profil 10,24 detik |
| --- | --- | --- |
| PCM sumber | `[1,2,245760]` | `[1,2,491520]` |
| Latent | `[1,128,64]` | `[1,256,64]` |
| Token teks / lyric / timbre | 105 / 11 / 1 | 106 / 11 / 1 |
| Conditioning | `[1,117,2048]` | `[1,118,2048]` |
| Hidden DiT, dua cabang guidance | `[2,64,2048]` | `[2,128,2048]` |
| Cross mask | `[1,1,64,117]` | `[1,1,128,118]` |

Metadata prompt memakai formatter publisher untuk durasi baru (teks `10 seconds`
untuk jendela 10,24 detik). Checkpoint, preset bass, 24 layer, guidance 7,
shift 1, 50 langkah dan FP32 APG tetap sama. Profil pendek mempertahankan FP16
storage; conditioning dan denoiser profil panjang memakai BF16 storage dengan
hitungan tetap FP32. Encoder/decoder VAE tetap memakai bobot FP32.
Epsilon seed 31/noise seed 0 masih fixture host; tidak ada klaim RNG iPhone.

Referensi penuh tetap menggunakan conditioning FP32, denoiser FP32 asli dan
APG publisher FP64. Kandidat memakai conditioning hasil pipeline dan bobot
16 bit dengan hitungan FP32, serta APG FP32. Batas abs `1e-4`, rel `1e-3` dan
normalized RMSE `1e-3` tidak diubah. Host `strict_pass` atau `bounded_diagnostic`
dilaporkan sesuai hasil, bukan dipaksakan PASS.

Eksperimen awal 10,24 detik dengan FP16 storage melewati batas lintasan
(normalized RMSE final sekitar 0,00133). Mengganti proyeksi APG saja menjadi
FP64 masih melewati batas (sekitar 0,00117), sehingga FP64 native tidak dipakai.
Pemeriksaan bobot menemukan nilai kecil BF16 yang berubah saat dijadikan FP16;
BF16 memiliki rentang eksponen lebih luas. Profil panjang mempertahankan
penyimpanan BF16 dan cast FP32 sebelum operasi projection. Toleransi, referensi
conditioning penuh dan seluruh 50 langkah tidak diganti untuk menutupi selisih.
Profil 5,12 detik yang sudah diterima tetap memakai precision semula.

Pada mode impor, input baru tidak memiliki referensi host yang cocok: output
checks `0/0`, numerical parity **NOT EVALUATED**. Enam pembandingan conditioning
preset masih berlaku. Musical approval, latency/memori, cold start dan lagu
panjang dinilai terpisah dari gate tersebut.

## Validasi implementasi

85 uji regresi host lulus. Ketiga bundle hasil ekspor dan hash seluruh payload
diverifikasi, lalu driver aktivasi dan staging auto diuji: hanya profil panjang
yang terpilih (6,261 GB resource pada workspace ini). Gate pointwise yang gagal
tetap tercatat gagal. Ini belum bukti runtime CoreAI/Apple di iPhone.
