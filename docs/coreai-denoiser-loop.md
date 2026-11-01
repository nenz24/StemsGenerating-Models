# Uji empat langkah denoising di iPhone

Uji pengulangan denoiser sebelumnya sudah **PASS 3/3**. Uji berikutnya sekarang
menjalankan empat prediksi velocity berurutan dan memperbarui latent, sehingga
setiap langkah menerima hasil aktual langkah sebelumnya. Input tetap sintetis;
ini belum menghasilkan pengiring dari rekaman piano atau audio instrumen.

**Hasil terbaru sudah PASS di iPhone:** [laporan lengkap](../reviews/coreai-iphone17pro-denoiser-loop/README.md).
Seluruh 4/4 langkah, 56/56 tahap dan 16 pembandingan lolos. Total 53,00 detik;
normalized RMSE latent akhir terhadap FP32 asli `7,0143e-6`. Tidak perlu mengulang
uji yang sama untuk mencatat hasil ini. Instruksi berikut tersedia untuk reproduksi.

## Jalankan

Untuk iPhone yang sudah berhasil menjalankan 14 asset denoiser:

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Setelah memperbarui dari versi pertama yang gagal `invalidManifest`, lakukan
**Product → Clean Build Folder (Shift+Cmd+K)** sekali agar resource fixture ikut
diperbarui. Build dengan **Cmd+R**, lalu tekan **Uji loop denoising 4 langkah**.
Bagikan laporan lengkap setelah selesai. Target hasilnya:

```text
Four-step Euler loop dual parity: PASS (feedback at all four steps)
Completed parity steps: 4/4; completed stages: 56/56
```

Tidak perlu ekspor model atau unduh checkpoint ulang. Enam belas referensi,
tiga input uji persis dari pembuat referensi, dan manifest, sekitar 0,65 MB, ikut GitHub di
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

Fixture schema v2 membawa inputnya sendiri. Input acak dari persiapan Mac tidak
harus sama byte demi byte dengan hasil RNG Linux; kedua sisi uji membaca data
uji yang sama dari fixture. Pemeriksaan identitas model tetap berlaku.

Sebelum eksekusi, aplikasi memeriksa manifest jadwal, bentuk tensor, identitas
checkpoint dan SHA-256 seluruh referensi serta ketiga input fixture. Jika gagal,
pesan menyebut nama field, nilai yang diharapkan, dan nilai aktual. Laporan mencatat
load/run+copy per asset, hasil pembandingan, waktu per langkah dan footprint
proses. Pada error atau kegagalan numerik, loop berhenti tanpa retry tersembunyi.
Guard footprint 2000 MiB diperiksa sebelum setiap asset; ini bukan batas peak OS.
Cache internal, accelerator yang benar-benar digunakan, seluruh kebutuhan RAM
model/perangkat dan peak di antara sampel 10 ms tetap belum diukur.

[Referensi host sudah PASS](../reviews/coreai-four-step-denoiser-loop/README.md).
Percobaan perangkat pertama [berhenti saat validasi manifest, 0/56 tahap](../reviews/coreai-iphone17pro-denoiser-loop-manifest-failure/README.md).
Field penyebabnya belum diketahui dari pesan generik lama. Paket input mandiri
dan diagnostik field kemudian berhasil melewati validasi dan seluruh loop di
iPhone, tanpa mengubah referensi. Build/eksekusi itu dilaporkan pengguna;
workspace Linux tidak menyediakan Apple SDK untuk kompilasi independen.
Ini bukan pengujian kualitas musik,
CFG, conditioning dari rekaman, atau jaminan stabil untuk lagu 90 detik.

Pada laporan PASS, seluruh 48 pasangan layer memerlukan sekitar 1,01–1,14 detik
per interval run+copy. Constructor model hanya sekitar 1,6–1,9 ms pada kebanyakan
tahap. Interval run+copy juga memuat binding fungsi dan validasi, sehingga belum
diketahui apakah biaya utamanya berasal dari binding, backend, paging atau copy.
Tahap berikutnya adalah memisahkan pengukuran tersebut sebelum memilih optimisasi;
instrumentasi itu belum dibuat dalam pembaruan laporan ini. Footprint maksimum
tersampel 74,02 MiB tetap bukan seluruh RAM model/perangkat.

Untuk audit ulang referensi yang sudah ikut repository, tanpa model/bobot:

```bash
python3 scripts/prepare_coreai_denoiser_loop.py --verify-only
```

Generator referensi juga tersedia bagi pengembang melalui script yang sama tanpa
`--verify-only`, menggunakan environment Python generator dan checkpoint cache.
Output yang sudah ada dipertahankan; gunakan `--out` ke direktori baru untuk
perhitungan ulang. Tidak perlu menjalankannya untuk uji iPhone di atas.
