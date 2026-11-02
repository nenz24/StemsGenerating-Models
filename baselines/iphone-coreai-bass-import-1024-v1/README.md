# Baseline bass impor 10,24 detik

Pengguna menerima kecocokan nada dan groove: **“Sudah cocok dan enak didengar”**.
Run ini menghasilkan bass dari 10,24 detik awal `piano.mp3` melalui impor Files
di iPhone, encoder VAE, Qwen/ConditionEncoder, seluruh 24 layer DiT dengan
50 langkah APG, lalu decoder VAE. Kedua WAV output cocok dengan hash laporan.

Waktu generasi 772,415 detik dan footprint proses maksimum tersampel
1.276,660 MiB. Pemulihan cache tidak terpicu. Penerimaan musikal klip ini
tidak menetapkan latency produksi, exact peak/total RAM, cold-start repeatability
atau keberhasilan recovery BNNS.

Bass mentah Float32 memiliki peak +1,550 dBFS; mix −0,239 dBFS, tanpa sample
di atas unity. Tidak ada editing/normalisasi/limiter pada file asli. WAV piano
input 10,24 detik tidak diterima, sehingga hash PCM input dan rekonstruksi
mix belum diverifikasi secara independen.

[Laporan dan audit](../../reviews/coreai-iphone17pro-audio-import-1024/README.md)
memisahkan eksekusi, integritas dua output, status numerik dan penilaian pengguna.
Numerical parity output impor NOT EVALUATED; host preparation tetap
`trajectory_drift_diagnostic`, full numerical gate NOT PASSED.
Audio/bobot tidak ditambahkan ke Git. Prioritas berikutnya adalah biaya backend
profil 10,24 detik sebelum perluasan durasi.
