# Run pertama versus run ulang dengan satu fungsi tetap dimuat

[Profil loop](../reviews/coreai-iphone17pro-denoiser-loop-timing/README.md)
menempatkan 50,75 dari total 51,88 detik di interval `function.run`. Uji kecil ini
membandingkan pemanggilan pertama dan tiga pemanggilan berikutnya pada **model
dan fungsi yang sama**, agar biaya first-use bisa dibedakan dari eksekusi ulang.

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build **Cmd+R**, tekan **Uji reuse satu pasangan layer**, lalu bagikan laporan
lengkap. Target numeriknya `Retained pair dual parity: PASS (4/4 calls)`;
ringkasan harus mencatat satu model open dan satu function bind. Waktu run
pertama dan rata-rata run 2–4 dilaporkan terpisah; tidak ada target rasio yang
ditentukan terlebih dahulu.

Tidak perlu ekspor model atau unduh checkpoint ulang. Asset yang dipakai tetap
`denoiser-sharded/blocks-00-01.aimodel`. Paket baru sekitar 1 MB hanya berisi delapan
tensor input dan dua referensi output; paket tersebut ikut GitHub di
`ModelCandidates/denoiser-pair-reuse/`. Semua probe sebelumnya tetap tersedia.

Input 13 hidden/64 condition token berasal dari front storage16 pada langkah
pertama loop sintetis (t=1, timestep_r=1). Referensi FP32 dan storage16 dihitung
dengan **input pasangan layer yang sama persis**, sehingga uji ini memisahkan
galat pasangan layer dari galat front. Seluruh input/referensi dicek SHA-256,
shape, dtype, identitas checkpoint dan toleransi yang sama dengan baseline.
Keempat run wajib lolos terhadap kedua referensi host. Referensi itu bukan
output run pertama dari perangkat.

Fungsi dimuat/validasi satu kali, lalu input tetap dipakai untuk empat call.
`defer` dengan `withExtendedLifetime` mempertahankan kedua handle Swift sampai
loop pengukuran selesai; tidak ada reload, penghapusan cache, warmup tersembunyi,
atau empat model yang dimuat sekaligus. Output setiap call dibaca menjadi array
Float CPU mandiri dalam helper terpisah, sehingga output native tidak disimpan
antar call. Timer `output-readback+checks` mencakup salin/validasi Float; probe
komponen ini tidak membuat NDArray output baru untuk diteruskan ke layer lain.

Run pertama pada handle ini belum tentu cold cache. Reuse handle tidak menjamin
backend mempertahankan seluruh bobot di RAM. Jika run ulang jauh lebih cepat,
hasil itu mendukung amortisasi biaya pada **komponen ini**; ia belum membuktikan
loop 24 layer dapat dipercepat dengan rasio yang sama. Jika waktunya serupa,
handle reuse saja belum memberi penghematan pada kasus ini. Kedua hasil tetap
berguna untuk memilih strategi eksekusi berikutnya, termasuk evaluasi backend
lain atau kompresi model bila diperlukan.

Guard footprint 2000 MiB diperiksa sebelum model dan tiap call. Ini bukan batas
peak OS. Footprint tersampel 10 ms bukan seluruh RAM model/perangkat. Kebijakan
yang diminta tetap `cpuOnly`; penempatan accelerator sebenarnya tidak diukur.

[Referensi host sudah lolos](../reviews/coreai-retained-pair-reference/README.md).
Kode Swift baru perlu build dan pengujian di iPhone; workspace Linux tidak
memiliki Apple SDK. Uji ini tidak menjalankan full denoiser, CFG, rekaman piano,
atau waveform generation. Lihat [tahapan menuju generasi](iphone-generation-roadmap.md).
