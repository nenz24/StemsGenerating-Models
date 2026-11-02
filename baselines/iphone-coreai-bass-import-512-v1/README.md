# Baseline bass dari impor piano iPhone 5,12 detik

Pengguna menerima hasil baru: **“Sudah cocok dan enak didengar”**. Ini persetujuan
uji dengar untuk tiga WAV dari run impor Files yang tercatat di
[laporan](../../reviews/coreai-iphone17pro-audio-import/README.md), terpisah dari
baseline PCM host sebelumnya. Hash asli piano sama, tetapi PCM hasil konversi
iPhone berbeda. Semua hash WAV dan hash PCM input telah dicocokkan dengan bukti.

Ketiga file stereo Float32 48 kHz, 245760 frame, finite dan berdurasi 5,12 detik.
Mix persis sama dengan 0,78 × piano + 0,55 × bass pada posisi frame yang sama.
Bass mentah melewati unity sebanyak 186 sample kanal pada 96 frame (total 2 ms),
peak +0,317 dBFS; mix tetap di bawah unity dengan peak −0,617 dBFS. Nilai bass
Float32 dipertahankan tanpa gain/limiter, bukan bukti bahwa playback selalu
bebas clipping. Atur headroom sebelum konversi ke PCM integer bila diperlukan.

Manifest menyimpan identitas input/output, lingkup persetujuan dan pengukuran.
Tidak ada audio atau bobot yang ditambahkan ke Git. Ini bukan numerical PASS
atau validasi lagu panjang, input lain, prompt bebas maupun accelerator.
