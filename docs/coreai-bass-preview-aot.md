# AOT bass preview untuk iPhone h18p

Metal Toolchain **27B47** sudah terpasang di Mac pengguna dan `coreai-build`
tersedia. [Output help yang diberikan](../reviews/coreai-aot-toolchain-27b47/compile-help.txt)
mengonfirmasi opsi `--architecture` dan preferensi `gpu`, `neural-engine`,
`none`. **Tidak ada nilai `cpu` atau `cpuOnly` pada CLI ini.**

## Jalankan

Dari folder repository di Mac:

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_bass_preview_aot.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip memakai 16 model `.aimodel` dalam bundle preview yang sudah dibuat.
Tidak ada ekspor PyTorch, penghitungan ulang 50 langkah referensi, pemilihan
rekaman baru atau unduhan checkpoint. Yang dilakukan adalah kompilasi AOT:

```bash
xcrun coreai-build compile path/to/front.aimodel \
  --output path/to/output/ \
  --platform iOS --min-deployment-version 27.0 \
  --preferred-compute none --architecture h18p
```

Perintah tersebut dijalankan berurutan untuk 14 shard DiT, APG update dan
decoder. Hanya varian **h18p** yang diminta. Terminal menampilkan `AOT 1/16`
sampai `AOT 16/16`, nama aset dan waktu setiap kompilasi. Ukuran hasil AOT
belum diketahui; skrip menampilkannya saat selesai. Paket menambah resource
aplikasi, dan working copy lokal juga disimpan untuk resume.

Jika kompilasi terputus, jalankan ulang skrip yang sama. Hasil per aset yang
lengkap diverifikasi checksum dan dipakai ulang jika source, compiler dan
arsitektur sama. Percobaan parsial dipertahankan di `.runtime-cache`; tidak
ditandai selesai. Bundle final hanya dipasang setelah seluruh 16 hasil valid.
Bundle lama yang telah ada tidak ditimpa; source/arsitektur yang berubah ditolak.
Jika compiler menghasilkan link ke payload lain, skrip menyalin isinya menjadi
file mandiri sebelum checksum/packaging, sehingga paket tidak bergantung pada
path working directory Mac.

Tunggu **Ready**, lalu **Stop → Cmd+R** ke iPhone dan tekan tombol yang sama:
**Generate bass 5,12 detik**. Bagikan laporan lengkap serta WAV jika tersedia.
Tidak perlu menjalankan gate komponen lagi. Target mode laporan:

```text
Sampling/decoder assets: AOT .aimodelc; architecture h18p; 16 compiled assets SHA-256 VERIFIED
Sampling/decoder requested policy: default, matching compiler --preferred-compute none
```

## Kebijakan runtime dan integritas

`none` berarti tanpa preferensi compute, bukan CPU-only. Sesuai
[panduan Apple](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time),
loader aset AOT menggunakan `SpecializationOptions.default`, agar sesuai dengan
preferensi kompilasi `none`. **Source encoder dan Qwen tetap `cpuOnly`.** Untuk
aset sampling/decoder sumber tanpa paket AOT, jalur lama tetap `cpuOnly`.
Perubahan kebijakan ini dicatat di laporan. Pemakaian GPU, Neural Engine atau
CPU aktual tetap belum diukur dan tidak boleh ditebak dari opsi tersebut.

Paket final berada di `ModelCandidates/bass-preview-512-aot/` yang diabaikan Git.
Aplikasi memeriksa arsitektur aktual, source-reference SHA-256, pemetaan tepat
16 nama aset dan hash seluruh file compiled. Hash dibaca dalam potongan 4 MiB
agar tidak memuat satu aset besar sekaligus. Bundle yang salah arsitektur,
stale, tidak lengkap atau rusak menghentikan proses; tidak ada fallback diam-diam
ke source compilation. Jika folder AOT tidak ada, perilaku source lama berlaku.
Pemeriksaan checksum dan pipeline dijalankan dalam task background; pembaruan
laporan dan kontrol pemutar tetap pada MainActor UI.

Untuk kembali ke jalur source, pindahkan seluruh folder `bass-preview-512-aot`
keluar dari `ModelCandidates` ke lokasi cache/backup, lalu rebuild. Source
preview dan referensinya tetap tersedia. Jangan mengubah isi source bundle
untuk mencoba memperbaiki kegagalan AOT.

Scope model/fungsi tetap dibatasi satu aset per saat. Pemulihan cache CPU blok
18–19 hanya berlaku pada jalur source; paket AOT memakai resource/policy baru
dan tidak memicu penghapusan cache source tersebut. Output CoreAI aktual tetap
menjadi input tahap berikutnya; I/O, sampling, toleransi serta label
`bounded_diagnostic` tidak diubah. Parity AOT dinilai dari hasil perangkat.

## Status

Perintah berasal dari help toolchain pengguna dan dokumentasi Apple. Pengujian
argumen, packaging, hash, penolakan paket rusak/stale, resume dan preservasi
lolos di Linux. Native compiler tidak dijalankan atau disimulasikan pada host
Linux; fixture unit jelas bukan model yang bisa dieksekusi. Build Apple untuk
loader baru, kompilasi AOT nyata serta eksekusi/memori iPhone **masih menunggu**.

AOT memindahkan pekerjaan kompilasi yang mahal ke Mac, tetapi masih menyisakan
sebagian specialization pada perangkat. Ini mitigasi untuk kegagalan BNNS IR
mmap yang diamati, bukan bukti masalah memori sudah teratasi. Laporan akan tetap
menyebut aset, fase, resource `.aimodelc` dan error jika gagal.
