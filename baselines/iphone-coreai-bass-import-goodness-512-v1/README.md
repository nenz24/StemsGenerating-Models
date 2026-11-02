# Baseline impor kedua: Goodness of God, 5,12 detik

Pengguna menerima nada/feel: **“Sudah cocok dan enak didengar”**. Ini kasus
rekaman berbeda dari `piano.mp3`, dengan hash sumber yang dilaporkan dan hash
PCM hasil impor berbeda. Ketiga WAV aktual cocok dengan hash laporan; input
PCM diverifikasi dari WAV dan mix tepat sama dengan 0,78 × piano + 0,55 × bass.

File berdurasi 5,12 detik, stereo Float32 48 kHz, semua sample finite dan tidak
melewati unity. Peak bass −0,478 dBFS, mix −5,572 dBFS. Tidak ada editing,
normalisasi atau limiter yang diterapkan untuk audit ini.

Run Release CPU-request menyelesaikan 50 langkah dalam 77,624 detik, dengan
footprint maksimum proses tersampel 455,393 MiB; pengukuran mengecualikan
impor sebelumnya dan bukan exact peak/total RAM atau pembanding terkontrol.

[Laporan dan audit](../../reviews/coreai-iphone17pro-audio-import-goodness-of-god/README.md)
memisahkan bukti eksekusi, integritas file dan persetujuan musikal. Tidak ada
referensi host yang cocok untuk PCM ini; numerical parity tidak dievaluasi.
Persetujuan hanya berlaku untuk klip ini, bukan lagu panjang atau prompt bebas.
Tidak ada audio atau bobot yang ditambahkan ke Git.
