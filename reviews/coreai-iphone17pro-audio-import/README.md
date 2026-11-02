# Impor piano dan generasi bass pertama di iPhone

Laporan dan metadata yang diunggah pengguna mencatat satu generasi lengkap dari
Files pada iPhone, iOS 27.0.1 (24A446), CoreAI h18p, Release `-O` / DEBUG OFF.
Run `2DCBE0FA-DE16-4EF8-A516-49F904093F89` dimulai pada 2 November 2026,
19:54:43 WIB. Bukti ini tidak dijalankan ulang di workspace Linux.

Input `piano.mp3` adalah file yang sama dengan baseline, berdasarkan SHA-256
asli `ccd94ea37159f69c272623efb319a62984564f1799d3744d096b7add49246189`.
Metadata mencatat stereo 44,1 kHz, durasi 243,3306 detik, diubah pada iPhone
menjadi 245760 frame stereo 48 kHz dari awal rekaman tanpa normalisasi.
Hash asli dan PCM konsisten antara `input.json` dan `report.txt`; hash input
WAV yang dilaporkan juga cocok dengan metadata preview. File WAV aktual untuk
run ini belum diterima, sehingga hash itu belum diverifikasi dari byte WAV.

Encoder menghasilkan empat output dengan shape/finite checks sesuai kontrak.
Conditioning preset tetap lolos 6/6 pembandingan, seluruh 50 langkah sampling
selesai, dan decoder menghasilkan WAV bass serta campuran. Sampling melaporkan
15 open, 15 bind, 735 reuse dan 750 call, dengan pool dilepas sebelum decoder.

Total generasi **65,877 detik**, footprint maksimum proses tersampel
**472,706 MiB**. Timer/sampler generasi dimulai setelah tahap impor; angka ini
**tidak mencakup penyalinan file, decode/resampling dan hashing sebelumnya**.
Footprint bukan peak pasti atau total RAM model/perangkat. Angka 59,24 detik
pada fixture host sebelumnya bukan pembanding terkontrol untuk mengisolasi
biaya impor atau perubahan kecepatan.

`output checks: 0/0` dan `numerical parity NOT EVALUATED` sesuai mode impor:
tidak ada trajectory atau waveform host yang cocok untuk PCM ini. Enam gate
conditioning masih sah karena input teks/timbre tidak berubah. Hash WAV bass
dan mix berbeda dari baseline yang disetujui pengguna; metadata PCM baru bukan
bukti kesamaan dengan decoder/resampler Mac. Kualitas musikal run baru belum
dikonfirmasi dan tidak diwarisi dari persetujuan baseline lama.

Peak bass yang dilaporkan **1,037155**, atau sekitar **+0,317 dBFS**. WAV Float32
bisa menyimpan nilai di atas unity; nilai ini sendiri tidak membuktikan hard
clipping pada file. Perlu WAV aktual untuk memeriksa jumlah/bentuk sample,
headroom saat playback/konversi PCM dan level campuran. Tidak ada gain,
normalisasi atau limiter yang diubah berdasarkan laporan ini.

Berkas asli laporan dan metadata disimpan byte-for-byte; `summary.json` memuat
hash bukti dan angka yang diekstrak. Tidak ada bobot atau audio yang ditambahkan.
Langkah berikut: periksa WAV input/bass/mix dan dengarkan hasil, lalu coba
rekaman piano berbeda. Mono, stereo 48 kHz, penolakan audio pendek, pembatalan
picker, prompt bebas, RNG perangkat dan lagu panjang tetap belum divalidasi.
