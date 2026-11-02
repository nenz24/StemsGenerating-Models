# Impor kedua di iPhone: Goodness of God

Run `84AD4570-68FC-4AB4-8A3C-BD0BF178B03D` dimulai 2 November 2026,
20:49:35 WIB. Laporan pengguna mencatat iOS 27.0.1 (24A446), CoreAI h18p,
Release `-O`, DEBUG OFF dan source CPU-only yang diminta. Eksekusi Apple ini
tidak dijalankan ulang di Linux dan penempatan accelerator tidak diukur.

Input yang dilaporkan bernama `Goodness of God - piano instrumental cover with lyrics.mp3`.
Hash sumber berbeda dari `piano.mp3` sebelumnya; hash PCM juga berbeda dan
terverifikasi dari WAV yang diunggah. File asli MP3 dan `input.json` run ini
tidak disertakan, sehingga format/rate/durasi file asli tidak disimpulkan.
Input kanonis aktual: 245760 frame stereo 48 kHz Float32, 5,12 detik pertama.

Encoder sumber menyelesaikan shape/finite checks. Conditioning preset tetap
lolos 6/6 pembandingan, sampling menyelesaikan 50/50 langkah dengan 15 open,
15 bind, 735 reuse dan 750 call, lalu pool dilepas sebelum satu call decoder.
Laporan dan metadata hash konsisten dengan ketiga WAV. Mix Float32 tepat sama
dengan 0,78 × piano + 0,55 × bass, tanpa offset/gain fitting (residual nol).

Generasi memerlukan **77,624 detik**, sampled process maximum **455,393 MiB**.
Angka itu mengecualikan penyalinan, decode/resampling dan hash impor sebelumnya;
bukan peak pasti atau total RAM perangkat/model. Run 65,877 detik pada piano
sebelumnya bukan pembanding terkontrol untuk menyimpulkan sebab perbedaan waktu.

Ketiga WAV finite dan seluruh sample berada di dalam unity. Peak bass
**0,946443 (−0,478 dBFS)**, piano **0,105362 (−19,546 dBFS)**, mix
**0,526527 (−5,572 dBFS)**. Piano dalam jendela ini lebih pelan; weighted bass
RMS sekitar 16,48 dB di atas weighted piano RMS. Ini ukuran level komponen,
bukan penilaian loudness perseptual/akor atau alasan mengubah audio yang
disetujui. Headroom sample yang cukup tidak membuktikan bebas semua distorsi.

Pengguna menerima hasil baru: **“Sudah cocok dan enak didengar”**.
[Baseline uji dengar kasus kedua](../../baselines/iphone-coreai-bass-import-goodness-512-v1/README.md)
berlaku untuk klip ini saja. `output checks: 0/0` dan `NOT EVALUATED` tetap
sesuai karena tidak ada referensi trajectory/waveform host untuk input ini.
Persetujuan pada dua rekaman pendek tidak membuktikan lagu panjang, prompt
bebas, RNG iPhone, repeatability/cold start, GPU/Neural Engine atau batas thermal.

`report.txt` disimpan byte-for-byte. `summary.json` dan `waveform-metrics.json`
memuat hash, angka yang diekstrak dan audit file aktual. Tidak ada audio, bobot
atau perubahan inference yang ditambahkan. Pemeriksaan musikal berasal dari
pengguna; audit sample tidak mengklaim mendengar atau menilai akor.
