# Tahap 3: piano menjadi WAV bass pendek

Pipeline pertama sekarang diimplementasikan: **PCM piano → VAE posterior →
Qwen/ConditionEncoder → 50 langkah sampling → VAE decoder → WAV**. Dua gate
conditioning sebelumnya sudah PASS. Pipeline gabungan sudah lolos pembandingan
host; build Apple, eksekusi gabungan iPhone dan penilaian suara masih menunggu.

## Jalankan di Mac dan iPhone

Dari folder repository:

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_bass_preview.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Tidak perlu memilih rekaman lagi. Skrip memakai bundle `source-context` dan
`bass-conditioning` yang sudah dibuat. Jika salah satunya belum ada, skrip
memberi perintah persiapannya lalu berhenti. Bundle lama tetap disimpan.
Checkpoint Base/VAE yang sama dipakai ulang dari cache, tanpa kebutuhan CUDA.

Persiapan menjalankan 50 langkah referensi utuh dan 50 langkah lintasan terpisah,
kemudian mengekspor 16 aset baru: 14 shard denoiser dengan bentuk tensor nyata,
APG/update Euler, dan decoder 5,12 detik. **Total aset native baru sekitar
3,49 GB** pada ekspor Linux; ukuran/byte di Mac dapat berbeda. Ini menambah
resource aplikasi, bukan mengganti bundle lama. Proses CPU dapat memakan waktu;
terminal menampilkan langkah gold, langkah partition dan nama aset ekspor.

Setelah terminal menampilkan **Ready**:

1. Jalankan **Cmd+R** di Xcode ke iPhone.
2. Tekan **Generate bass 5,12 detik**, lalu biarkan aplikasi terbuka sampai selesai.
3. Tekan **Dengarkan bass** dan **Dengarkan bersama piano**.
4. Bagikan laporan serta WAV bass dan piano+bass menggunakan tombol yang muncul.

Ini satu pipeline generasi, tidak perlu menekan ulang tombol gate komponen.
Laporan harus menunjukkan 50/50 langkah dan `WAV generated on device: YES`.
Target pembandingan adalah sepuluh pemeriksaan output lolos: latent pada empat
checkpoint terhadap dua referensi, serta waveform terhadap dua referensi.

Jika pembandingan numerik gagal tetapi data tetap finite dan pipeline selesai,
WAV tetap bisa dibagikan sebagai **hasil diagnostik**. Laporan secara eksplisit
menulis `numerical parity: NOT PASSED`; ini bukan baseline yang sudah diterima.
Error kontrak/hash, tensor, memori atau kegagalan backend menghentikan pipeline.
Jika error `missingPreparedBundle`, kirim hasil persiapan terminal; pastikan
Ready tercapai sebelum rebuild. Clean Build Folder dapat dipakai bila resource
yang telah dibuat belum ikut aplikasi.

## Yang benar-benar berjalan di perangkat

Encoder sumber dijalankan sekali pada PCM/epsilon yang sama dengan gate lalu
output context aktualnya diteruskan ke sampling. Qwen dan ConditionEncoder
dijalankan kembali seluruhnya pada prompt bass; output aktual condition/mask
juga diteruskan. Referensi latent dan embedding host hanya dibaca untuk
pembandingan, tidak menjadi input pengganti hasil model perangkat.

Denoiser menerima 128 latent frame/117 condition token, menghasilkan dua
velocity sekaligus untuk conditional dan null branch. Null condition adalah
parameter publisher yang diverifikasi; kedua branch memakai source context
piano yang sama. Seluruh 24 layer dijalankan pada setiap langkah.

Presetnya 50 langkah Euler ODE, shift 1, guidance 7. Jalur publisher menggunakan
**APG**, momentum -0,75, threshold norm 2,5 dan projection pada dimensi waktu
latent (dim 1), bukan sekadar rumus CFG linear. Native update menghitung
projection FP32; host membandingkannya pada input identik dengan projection
FP64 asli setiap langkah. Timestep-r sama dengan timestep; KV cache dimatikan.

Ada 15 aset model/fungsi yang dipertahankan selama sampling: 14 denoiser shard
+ APG update. Targetnya 15 open, 15 bind, 735 reuse dan 750 call. Pool Swift
dilepas sebelum decoder penuh `[1,64,128] → [1,2,245760]`; eviction backend
belum dijamin. Guard footprint 2000 MiB diperiksa sebelum aset dan sesudah bind,
bukan jaminan peak OS. Sampler memori mencakup aplikasi selama seluruh pipeline,
namun tidak mengukur total RAM model/perangkat atau peak eksak.

## WAV dan batas prototipe

`bass.wav` menyimpan output asli tanpa normalisasi dalam Float32 stereo 48 kHz.
`piano-with-bass.wav` adalah campuran dengar: piano 0,78 dan bass 0,55, diturunkan
bersama jika peak melebihi 0,98. Stem bass tetap tidak diubah. Setiap run membuat
folder baru di Documents aplikasi sehingga hasil sebelumnya tidak ditimpa.
Gunakan ShareLink sebelum menutup aplikasi; kontrol pemutar memakai hasil run
saat ini. Format RIFF/Float32 sudah diperiksa dengan ffprobe/FFmpeg; pemutaran
AVAudioPlayer pada Apple masih perlu dicoba.

PCM, tokenisasi, epsilon dan initial noise masih dipersiapkan di Mac. Seluruh
neural inference dan sampling dijalankan di iPhone saat tombol ditekan, tetapi
ini belum alur merekam file baru dan mengetik prompt bebas langsung di iPhone.
Noise memakai fixture CPU seed 0 dan epsilon fixture; RNG iPhone serta kesamaan
stream CUDA belum divalidasi. Klip ini bukan pengganti baseline musikal 90 detik.

[Catatan host](../reviews/coreai-bass-preview-export/README.md) memisahkan gate
APG pada input identik, lintasan latent, waveform dan diagnostik velocity.
Setelah generasi iPhone berjalan, akor, groove, timbre dan timing dinilai pada
tahap 4. Kualitas musikal, lagu panjang dan anggaran thermal/baterai masih terbuka.
