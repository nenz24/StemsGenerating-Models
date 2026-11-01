# Uji kedua tahap 1: reuse seluruh 24 layer selama loop

[Uji pasangan layer sudah PASS](../reviews/coreai-iphone17pro-retained-pair/README.md):
backend-await pertama 1042,53 ms dan rata-rata run ulang 8,52 ms. Uji berikutnya
mempertahankan model/fungsi pada loop penuh. Rasio komponen 122× tidak berlaku
otomatis untuk seluruh loop.

**Hasil iPhone sudah PASS:** [laporan lengkap](../reviews/coreai-iphone17pro-retained-loop/README.md),
4/4 langkah, 56/56 tahap, 16 pembandingan dan seluruh penghitung retention
sesuai target. Total 18,52 detik; langkah 1–4 adalah 13,43 / 3,93 / 0,50 / 0,53
detik. Guard tidak terpicu pada kasus ini. Kedua uji utama tahap 1 selesai untuk
memilih jalur prototipe; pekerjaan berikutnya adalah conditioning rekaman.
Tidak perlu mengulang uji ini untuk mencatat hasilnya.

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build **Cmd+R**, tekan **Uji loop 24 layer dengan reuse**, lalu bagikan laporan
lengkap. Tidak perlu unduh checkpoint, ekspor model atau referensi baru; fixture
empat langkah yang sudah PASS tetap dipakai.

Setiap asset dibuka dan dibind saat pertama diperlukan pada langkah 1. Kedua
handle disimpan agar langkah 2–4 memakai fungsi yang sama dengan latent/input
aktual langkah tersebut. Semua 14 asset tetap terpisah: front, 12 pasangan layer
dan tail. Tidak ada warmup/retry tambahan atau perubahan jadwal, CFG, dtype,
referensi maupun toleransi. Input tetap divalidasi pada setiap call dan output
disalin ke NDArray CPU mandiri.

Footprint proses diperiksa sebelum tiap tahap dan lagi setelah bind/validasi
sebelum backend dijalankan. Caller berhenti jika mencapai **2000 MiB** atau
pembacaan footprint gagal. Ini bukan cap OS: lonjakan di dalam operasi masih
bisa terlewat. Menahan semua handle dapat menambah memori; shared/file-backed
dan backend allocation tidak dianggap sudah tercakup sebagai seluruh RAM model.
Reuse handle juga tidak menjamin bobot tetap resident di RAM.

Semua handle dipertahankan sampai loop selesai atau gagal, kemudian pool Swift
dilepas. Eviction cache native tidak dijamin; waktu cleanup masuk wall time di
luar tahap. Jalur monolitik yang gagal tetap dinonaktifkan.

Target keberhasilan:

```text
Four-step Euler loop dual parity: PASS (feedback at all four steps)
Retention totals: model opens 14; function binds 14; cache hits 42; completed calls 56; cached entries before release 14
Completed parity steps: 4/4; completed stages: 56/56
```

Seluruh 16 pembandingan velocity/latent dan jumlah open/bind/hit juga wajib
lolos sebelum PASS ditulis. Laporan fase menunjukkan biaya first-use langkah 1
dan reuse langkah 2–4. Timestep/input berbeda antar langkah sesuai sampler;
ini uji pipeline dengan feedback, bukan pengulangan input identik.

Ini **uji kedua dari dua uji utama tahap 1 yang direncanakan**. Jika parity dan
hasil waktu/memori mendukung jalur ini, hasilnya menjadi dasar keputusan eksekusi
untuk masuk conditioning rekaman. Jika guard menghentikan uji, proses gagal atau
reuse tetap mahal, perlu keputusan alternatif yang spesifik. Keberhasilan satu
pasangan layer saja belum cukup untuk menyelesaikan tahap 1.

Pengguna sudah membuild dan menjalankan mode ini di iPhone; workspace Linux
tidak memiliki Apple SDK untuk kompilasi independen. Ini belum conditioning rekaman, waveform atau lagu panjang.
Tombol loop lama tetap tersedia dengan model dilepas setelah setiap tahap.
