# Tahap 3: piano menjadi WAV bass pendek

> Pembaruan perangkat: [AOT/default crash di compiler Metal/MPS](../reviews/coreai-iphone17pro-aot-metal-crash/README.md).
> Build `auto` memakai source CPU-only; AOT tetap tersimpan di Mac dan tidak ikut
> app default. Load/bind/backend/readback dicatat sebelum panggilan native.
> [Console dan konfirmasi pengguna](../reviews/coreai-iphone17pro-bass-preview-cpu-complete/README.md)
> kini mencatat satu run 50 langkah + decoder lengkap; bass berhasil diputar.
> Parity numerik, elapsed time dan kualitas musikal masih memerlukan laporan
> akhir serta audio hasil run yang sudah ada.


Pipeline pertama sekarang diimplementasikan: **PCM piano → VAE posterior →
Qwen/ConditionEncoder → 50 langkah sampling → VAE decoder → WAV**. Dua gate
conditioning sebelumnya sudah PASS. Pipeline gabungan sudah lolos pembandingan
host Linux dan satu eksekusi gabungan iPhone sudah menghasilkan bass yang dapat
diputar. Penilaian numerik perangkat dan kualitas suara tetap terpisah dari
keberhasilan eksekusi tersebut.

**Catatan Mac:** [laporan pengguna](../reviews/coreai-mac-bass-preview-drift/README.md)
menunjukkan strict pointwise parity gagal setelah akumulasi 50 langkah, sehingga
persiapan versi awal berhenti sebelum ekspor. Perbedaan numerik itu belum
dihilangkan. Persiapan kini dapat mengekspor `bounded_diagnostic` dengan status
gagal tetap tersimpan, sehingga hasil bisa didengar tanpa mengklaim parity PASS.

## Jalankan di Mac dan iPhone

**Metal Toolchain 27B47 kini tersedia:** [jalur AOT](coreai-bass-preview-aot.md)
memakai compiler resmi di Mac untuk 16 aset khusus `h18p`. Tidak perlu mengulang
ekspor/referensi. Pilihan compiler `none` dipasangkan dengan runtime `default`
untuk sampling/decoder; source/Qwen tetap CPU-only. AOT belum divalidasi pada
perangkat dan masih menyisakan specialization, sehingga tidak menjamin error
memori selesai. Laporan baru membedakan resource/policy AOT dari jalur source.

**Update setelah kegagalan `blocks-18-19`:** [laporan terlokalisasi](../reviews/coreai-iphone17pro-bass-preview-block1819/README.md)
berhasil menjalankan sepuluh aset sampling, lalu gagal binding aset ke-11.
Preview kini melepas scope model/fungsi Swift setelah setiap aset dan menyalin
output ke CPU. Jika laporan tersimpan mencatat BNNS graph-load failure pada
blok 18–19, hanya cache kompilasi CPU aset tersebut yang diminta untuk dibangun
ulang. Aksi/hasil pemulihan muncul di laporan. Bobot/referensi tidak berubah.

Untuk mencoba perbaikan ini: `git pull --ff-only`, **Stop** aplikasi di Xcode,
lalu **Cmd+R → Generate bass 5,12 detik**. Tidak perlu menjalankan skrip ekspor
ulang atau menunggu Metal Toolchain. Proses bisa lebih lambat karena model/fungsi
dibuka ulang per aset; hasil numerik dan penggunaan memori aktual masih perlu
divalidasi di iPhone.

**Jika paket sudah Ready dan iPhone berhenti pada `failedToSpecialize`:**
[laporan pertama](../reviews/coreai-iphone17pro-bass-preview-specialization/README.md)
menunjukkan source/conditioning PASS, lalu kegagalan specialization sebelum
sampling pertama selesai. Update pelaporan kini menyimpan nama aset dan operasi
yang gagal. Jalankan `git pull --ff-only`, build Cmd+R, lalu tekan tombol generasi
yang sama sekali lagi. **Tidak perlu menjalankan skrip persiapan/ekspor ulang.**
Bagikan laporan baru, khususnya `Failing asset`, `Failing operation` dan error
domain/code. Console pengguna kemudian menunjukkan BNNS gagal mengalokasikan
mapping `bnns.ir` saat compile. Pembaruan pelaporan tetap diperlukan untuk
mengidentifikasi aset/operasinya; ini belum membuktikan kegagalan runtime teratasi.

Metal Toolchain 27B47 telah terpasang dan help compiler sudah diberikan.
Gunakan [persiapan AOT](coreai-bass-preview-aot.md) untuk kompilasi di Mac:
`--preferred-compute none --architecture h18p`. Compiler tidak menyediakan
preferensi CPU-only; loader AOT memakai runtime `default`, sedangkan jalur
source tetap CPU-only. AOT masih menyisakan sebagian specialization di iPhone
dan belum divalidasi sebagai solusi error ini.

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

Persiapan host juga membedakan **`strict_pass`** dari **`bounded_diagnostic`**.
Paket diagnostik mensyaratkan setiap latent dan waveform finite dengan
normalized RMSE <=1e-3; APG pada input identik tetap wajib lolos batas strict.
Batas absolute/relative/normalized RMSE tidak diubah. Error besar/nonfinite atau
kegagalan APG/integritas tetap menghentikan ekspor. `Ready` berarti paket selesai
disiapkan, bukan otomatis berarti gate numerik PASS. Aplikasi memeriksa status
host itu; paket diagnostik tidak dapat menghasilkan label numerical PASS
keseluruhan walaupun sepuluh pembandingan perangkat kebetulan lolos.

Untuk kegagalan Mac tersebut, jalankan ulang perintah di atas setelah pull.
Checkpoint cache dipakai ulang; referensi dihitung ulang karena percobaan lama
berhenti sebelum ekspor. Bundle strict-pass yang sudah lengkap tetap bisa
diverifikasi/dipakai tanpa diekspor ulang.

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

Ada 15 jenis aset sampling: 14 denoiser shard + APG update. Sekarang hanya satu
scope model/fungsi Swift yang hidup per aset; setiap hasil disalin ke CPU lalu
scope dilepas. Targetnya 750 open, 750 bind, nol reuse dan 750 call sampling.
Decoder penuh `[1,64,128] → [1,2,245760]` juga memakai scope terpisah. Cache backend
tidak dijamin ikut dilepas. Guard footprint 2000 MiB diperiksa sebelum aset dan sesudah bind,
bukan jaminan peak OS. Sampler memori mencakup aplikasi selama seluruh pipeline,
namun tidak mengukur total RAM model/perangkat atau peak eksak.

## WAV dan batas prototipe

`bass.wav` menyimpan output asli tanpa normalisasi dalam Float32 stereo 48 kHz.
`piano-with-bass.wav` adalah campuran dengar: piano 0,78 dan bass 0,55, diturunkan
bersama jika peak melebihi 0,98. Stem bass tetap tidak diubah. Setiap run membuat
folder baru di Documents aplikasi sehingga hasil sebelumnya tidak ditimpa.
Gunakan ShareLink sebelum menutup aplikasi; kontrol pemutar memakai hasil run
saat ini. Format RIFF/Float32 sudah diperiksa dengan ffprobe/FFmpeg; pemutaran
AVAudioPlayer untuk stem bass sudah dikonfirmasi berhasil oleh pengguna pada
run lengkap ini. Berhasil membagikan file/memutar listening mix belum dikonfirmasi.

PCM, tokenisasi, epsilon dan initial noise masih dipersiapkan di Mac. Seluruh
neural inference dan sampling dijalankan di iPhone saat tombol ditekan, tetapi
ini belum alur merekam file baru dan mengetik prompt bebas langsung di iPhone.
Noise memakai fixture CPU seed 0 dan epsilon fixture; RNG iPhone serta kesamaan
stream CUDA belum divalidasi. Klip ini bukan pengganti baseline musikal 90 detik.

[Catatan host](../reviews/coreai-bass-preview-export/README.md) memisahkan gate
APG pada input identik, lintasan latent, waveform dan diagnostik velocity.
Setelah generasi iPhone berjalan, akor, groove, timbre dan timing dinilai pada
tahap 4. Kualitas musikal, lagu panjang dan anggaran thermal/baterai masih terbuka.
