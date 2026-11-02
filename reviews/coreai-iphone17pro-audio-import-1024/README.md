# Impor piano 10,24 detik: generasi selesai, uji dengar diterima

Run `95E09157-5A7E-4C46-9231-B83793094718`, mulai 2 November 2026
16:19:08 UTC / 23:19:08 WIB. Pengguna mengirim [laporan](report.txt), `bass 4.wav`
dan `piano-with-bass 4.wav`, lalu menilai **“Sudah cocok dan enak didengar”**.
Ini bukti perangkat dan penilaian pengguna, bukan reproduksi native atau uji
dengar independen dari workspace Linux.

| Pengukuran | Hasil |
| --- | ---: |
| Perangkat / runtime | iPhone 17 Pro, iOS 27.0.1 (24A446), CoreAI h18p |
| Build / policy | Release, Swift `-O`, cpuOnly diminta |
| Cuplikan | 10,24 detik awal `piano.mp3`, impor/decode di iPhone |
| Source / conditioning | Selesai; conditioning 27/27 tahap, 6/6 pembandingan PASS |
| Sampling / decoder | 50/50 langkah APG, 750 sampling calls + 1 decoder |
| Sampling lifecycle | 30 opens, 30 binds, 720 reuse hits |
| Pemulihan cache | 0/1 percobaan; jalur recovery tidak terpicu |
| Waktu generasi | 772,415 detik, sekitar 12 menit 52 detik |
| Footprint proses maksimum tersampel | 1.276,660 MiB, sekitar 1,247 GiB |
| Output | Dua WAV stereo Float32 48 kHz, 491.520 frame per kanal |

Kedua hash WAV aktual cocok dengan laporan. Semua sample finite. Peak bass
mentah **1,195408 / +1,550 dBFS**: 1.307 channel-sample di atas unity, pada
682 frame (14,208 ms gabungan). Nilai ini tersimpan pada Float32; ini bukan
bukti tersendiri bahwa distorsi clipping terdengar. Peak mix **0,972835 /
−0,239 dBFS**, tanpa sample di atas unity. Tidak ada gain/limiter/normalisasi
yang diterapkan ke file pengguna selama audit. Stem membutuhkan headroom saat
dipakai pada jalur playback atau ekspor yang membatasi nilai pada unity.

`piano.wav` 10,24 detik tidak disertakan. Hash PCM input dan hash WAV piano
tetap berstatus dilaporkan oleh app; rekonstruksi mix terhadap input aktual
tidak diverifikasi. Tidak ada klaim bahwa dua WAV output membuktikan hash
input atau numerical parity. [Metrik waveform](waveform-metrics.json) dan
[ringkasan](summary.json) mencatat batas bukti tersebut.

Input impor ini tidak memiliki referensi lintasan/waveform host yang cocok:
output checks **0/0, numerical parity NOT EVALUATED**. Host preparation tetap
`trajectory_drift_diagnostic`, dengan full numerical gate NOT PASSED. Enam
conditioning checks tidak mengubah status seluruh generasi menjadi PASS.

Run selesai pada kode yang memiliki recovery satu kali, tetapi recovery tidak
dipakai. Ini belum bukti perbaikan semua error BNNS, kestabilan cold launch,
atau kontrol native cache. Ownership sequential di langkah pertama dan reuse
mulai langkah kedua tercatat; penempatan accelerator dan total RAM perangkat
tetap tidak diukur. Footprint adalah sampel proses setiap 10 ms, bukan exact
peak; waktu generasi mengecualikan impor/decode/hash sebelumnya.

Performa masih menjadi pekerjaan utama. Rata-rata langkah 3–50 **15,191 detik**;
tidak terlihat penurunan besar setelah handle ditahan. Backend-await mencapai
744,691 detik (**96,41%** total pipeline), sedangkan output-copy 12,163 detik
dan model-open 1,418 detik. Ini interval wall API, bukan kernel time atau bukti
penyebab internal. BF16, durasi, thermal, native residency dan kondisi cache
belum dipisahkan melalui eksperimen terkontrol. Rasio wall/audio sekitar
75,43; tidak ada klaim real-time atau latency produksi yang diterima.

[Baseline uji dengar](../../baselines/iphone-coreai-bass-import-1024-v1/README.md)
menyimpan penerimaan klip ini. Audio dan bobot tidak masuk Git. Langkah berikutnya
adalah mengisolasi biaya backend pada profil panjang sebelum memperpanjang
durasi atau menambah instrumen.
