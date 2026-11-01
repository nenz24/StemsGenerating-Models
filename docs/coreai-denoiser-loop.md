# Uji empat langkah denoising di iPhone

Uji pengulangan denoiser sebelumnya sudah **PASS 3/3**. Uji berikutnya sekarang
menjalankan empat prediksi velocity berurutan dan memperbarui latent, sehingga
setiap langkah menerima hasil aktual langkah sebelumnya. Input tetap sintetis;
ini belum menghasilkan pengiring dari rekaman piano atau audio instrumen.

## Jalankan

Untuk iPhone yang sudah berhasil menjalankan 14 asset denoiser:

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build dengan **Cmd+R**, lalu tekan **Uji loop denoising 4 langkah**.
Bagikan laporan lengkap setelah selesai. Target hasilnya:

```text
Four-step Euler loop dual parity: PASS (feedback at all four steps)
Completed parity steps: 4/4; completed stages: 56/56
```

Tidak perlu ekspor model atau unduh checkpoint ulang. Enam belas referensi kecil
dan manifest, total sekitar 109 KB, ikut GitHub di
`ModelCandidates/denoiser-loop-4/`. Empat belas native asset tetap berada di folder
lokal `denoiser-sharded/` yang sebelumnya sudah disiapkan. Bila folder itu belum
ada di checkout ini, siapkan dengan `bash scripts/prepare_coreai_denoiser.sh`.
Semua tombol pengujian sebelumnya tetap tersedia.

## Jadwal dan pembandingan

Jadwalnya `1 → 0,75 → 0,5 → 0,25 → 0`, shift=1, guidance=1, tanpa CFG atau KV cache.
`timestep_r` sama dengan `timestep` pada setiap langkah. Pembaruan Euler Float32
memakai `latent_baru = latent − velocity × (t − t_berikutnya)`. Jadwal dan rumus
diambil dari metode `generate_audio` pada source publisher yang dipin di manifest
aset Base. Source dan checkpoint diverifikasi sebelum referensi dibuat.

Referensi host menggunakan dua jalur independen: model asli FP32 utuh dan
rangkaian partisi dengan bobot tersimpan FP16/perhitungan FP32. Keduanya
memperbarui latent sendiri pada setiap langkah. iPhone juga meneruskan latent
hasilnya sendiri; tidak mengisi ulang latent dari referensi. Velocity dan latent
setiap langkah harus lolos terhadap kedua jalur, total 16 pembandingan, dengan
batas sebelumnya: `abs(error) <= 1e-4 + 1e-3 * abs(reference)` dan normalized RMSE
`<= 1e-3`. Tidak ada pelonggaran toleransi untuk akumulasi drift.

Sebelum eksekusi, aplikasi memeriksa manifest jadwal, bentuk tensor, identitas
checkpoint dan SHA-256 seluruh referensi serta input bersama. Laporan mencatat
load/run+copy per asset, hasil pembandingan, waktu per langkah dan footprint
proses. Pada error atau kegagalan numerik, loop berhenti tanpa retry tersembunyi.
Guard footprint 2000 MiB diperiksa sebelum setiap asset; ini bukan batas peak OS.
Cache internal, accelerator yang benar-benar digunakan, seluruh kebutuhan RAM
model/perangkat dan peak di antara sampel 10 ms tetap belum diukur.

[Referensi host sudah PASS](../reviews/coreai-four-step-denoiser-loop/README.md).
Kode Swift baru belum dikompilasi dengan Apple SDK di workspace Linux; hasil
CoreAI empat langkah perlu diuji di iPhone. Ini bukan pengujian kualitas musik,
CFG, conditioning dari rekaman, atau jaminan stabil untuk lagu 90 detik.

Untuk audit ulang referensi yang sudah ikut repository, tanpa model/bobot:

```bash
python3 scripts/prepare_coreai_denoiser_loop.py --verify-only
```

Generator referensi juga tersedia bagi pengembang melalui script yang sama tanpa
`--verify-only`, menggunakan environment Python generator dan checkpoint cache.
Output yang sudah ada dipertahankan; gunakan `--out` ke direktori baru untuk
perhitungan ulang. Tidak perlu menjalankannya untuk uji iPhone di atas.
