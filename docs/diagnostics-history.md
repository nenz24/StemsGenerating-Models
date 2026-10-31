# Arsip diagnosis POC

Catatan di bawah adalah riwayat eksperimen, bukan urutan instruksi aktif.
Panduan aktif: [README](../README.md) dan [satu referensi CUDA](cuda-reference.md).
Status terakhir sebelum uji CUDA: CFG 1 lebih buruk menurut pengguna; rekonstruksi
piano bersih. CUDA nyata belum diuji di environment chat ini.

---

# StemsGenerating-Models

POC lokal di JupyterLab: rekaman gitar/piano → bass atau drum pengiring memakai
model pretrained ACE-Step. Mulai dari evaluasi kualitas, kemudian putuskan apakah
fine-tuning diperlukan. Belum ada model baru yang dilatih atau konversi iPhone.

## Rekonstruksi piano bersih: uji guidance DiT 7 versus 1

Pengguna mengonfirmasi `source_listen.wav`, `cpu_encoder_listen.wav`, dan
`mlx_encoder_listen.wav` semuanya bersih. Bersama kesamaan output decoder
MLX/CPU, ini mengurangi dugaan jalur VAE sebagai penyebab utama pada sampel uji.
Bass hasil generasi masih glitch; conditioning dan generasi DiT perlu diperiksa.
Rekonstruksi bersih bukan bukti seluruh pipeline benar pada semua input.

Uji satu pengaturan menggunakan B dan baseline yang sudah ada:

```bash
git pull --rebase --autostash
python3 scripts/compare_guidance.py --execute
```

Satu generasi B baru memakai **guidance 1**, dengan sumber byte-identik, seed,
Base/MPS, caption dan 50 langkah sama. **CFG 7 tidak digenerasi ulang**; baseline
diambil dari cache pembanding decoder. Bobot lokal dipakai, tanpa unduhan baru.
Pada source Base yang dipin, guidance 1 melewati penguatan CFG/APG dan cabang
batch conditional/unconditional; prompt dan conditioning audio tetap digunakan.
Ini belum membuktikan penyebab tertentu jika kualitas membaik: perubahan satu
setting juga mengubah cabang implementasi/batching. RNG encoder dan MPS dapat
bervariasi antarsesi meski seed DiT sama.

Di folder `guidance_comparison_*`, bandingkan `cfg7_listen.wav` dan
`cfg1_listen.wav`. Kedua pratinjau memakai gain 0.7 yang sama. Report menyimpan
parameter yang benar-benar dipakai; bila setting lain ikut berubah, uji menolak
menganggapnya perbandingan terkontrol dan tetap menyimpan hasil untuk inspeksi.
Tanpa `--execute`, hanya snapshot dan plan dibuat. Hasil lama dipertahankan.

Default CLI/notebook lama tetap guidance 7. CLI ACE kini menerima
`--guidance-scale` dalam [1,15], dan cache menolak pencampuran hasil guidance
berbeda. Mode lagu penuh belum diberi preset guidance baru sebelum kualitas
percobaan ini dinilai. Jika CFG 1 masih rusak, jangan mengulang lagu penuh atau
menganggap penambahan step otomatis memperbaiki; telusuri conditioning/backend
DiT atau kesesuaian model dengan tugas bass ini.

## Pemeriksaan pemutaran sub-bass (tetap glitch menurut laporan pengguna)

Pengguna mendengarkan melalui speaker bawaan MacBook. Sampel hasil memiliki
sekitar **60% energi di bawah 60 Hz** dan **82% di bawah 80 Hz**; piano sumber
hampir tidak punya energi di wilayah itu. Speaker kecil dapat mendistorsi sinyal
seperti ini. Ini hipotesis pemutaran yang perlu diperiksa, belum bukti bahwa file
atau kualitas bass sudah benar. Sebelum mencoba model lagi, lakukan uji cache:

```bash
git pull --ff-only
python3 scripts/check_playback.py
```

Di folder baru yang dicetak, bandingkan dengan **volume perangkat tetap**:

- `reference.wav`: cuplikan B yang sama dengan paket feedback.
- `quiet_18db.wav`: sinyal sama, dipelankan 18 dB.
- `highpass80_quiet.wav`: gain sama dengan versi pelan, ditambah high-pass 80 Hz.

Mulai pada volume perangkat rendah dan jangan menaikkannya untuk mengompensasi
file yang lebih pelan. Bila tersedia, dengarkan file yang sama melalui headphone.
Jika artefak jauh berkurang pada level rendah/headphone, pemutaran mungkin
berperan. Hilangnya artefak setelah high-pass saja bukan bukti speaker bermasalah:
filter juga menghapus bass yang diinginkan. Ini diagnosis, bukan stem final.
Tidak ada inferensi, unduhan atau perubahan pada hasil lama.

## Analisis sampel dan langkah lanjutan bila playback tetap glitch

Tiga file feedback selaras, 10 detik pada detik lagu 30–40, dan hash sesuai
manifest. Waveform decoder MLX dan CPU memiliki korelasi **0.999999462**;
RMS selisih setelah penyamaan skala hanya sekitar **0.104%** dari RMS default.
Sekitar 99.41% energi output berada di bawah 300 Hz. Piano sumber mempunyai
struktur nada yang jelas. Konsentrasi energi rendah sendiri wajar untuk bass;
angka ini bukan ukuran mutu musikal atau bukti tunggal penyebab glitch.

Kesamaan waveform melemahkan dugaan perbedaan backend decoder sebagai penyebab
utama pada sampel ini. Encoder, conditioning, DiT, dan mutu model LEGO masih
perlu diperiksa. Mengubah chunk/gain/decoder lagi belum didukung bukti ini.

Jika artefak tetap terdengar melalui headphone/level rendah, pemeriksaan berikut
yang sudah disiapkan hanya rekonstruksi sumber piano, **tanpa generasi bass**:

```bash
git pull --ff-only
python3 scripts/check_source_roundtrip.py --execute
```

Sumber diambil dari pembanding decoder terakhir, pada offset B 3–13 detik yang
sama dengan paket feedback. Snapshot tidak diberi gain tambahan. Worker hanya
memuat bobot VAE lokal: encoder PyTorch CPU float32 dan encoder MLX upstream,
kemudian **decoder CPU yang sama** untuk kedua hasil. DiT dan LM tidak dimuat;
tidak ada diffusion atau unduhan baru. Encoder MLX memakai callable native/
compiled upstream. Jika MLX tidak tersedia, uji melaporkan error dan menyimpan
rekonstruksi CPU; tidak diam-diam menganggap fallback sebagai hasil MLX.

Di folder `source_roundtrip_*` yang dicetak, dengarkan:

- `source_listen.wav`: piano sumber.
- `cpu_encoder_listen.wav`: encoder CPU → decoder CPU.
- `mlx_encoder_listen.wav`: encoder MLX → decoder CPU.

Keduanya seharusnya tetap terdengar sebagai rekonstruksi piano, bukan bass.
Semua pratinjau memakai durasi dan gain bersama, tanpa padding. PyTorch dan MLX
memakai sampler posterior dengan RNG berbeda; ini bukan perbandingan bit-identik.
Jika hanya MLX rusak, telusuri jalur encoder MLX. Jika keduanya bersih, fokus ke
conditioning/generasi DiT dan kualitas model. Jika keduanya rusak, periksa VAE/
jalur rekonstruksi bersama. Report mencatat shape/std latent dan hasil/error.
Tanpa `--execute`, hanya snapshot sumber yang dibuat. Hasil lama dipertahankan.

## Ekspor sampel audio tanpa generasi baru (sudah diterima dan dianalisis)

Laporan terakhir: sumber hanya piano, decoder default dan pembanding CPU sama-sama
glitch. `capture.json` memastikan default memanggil MLX VAE float32 dan kembali
tanpa error. Latent `[1,750,64]` memiliki min sekitar -6.07, max 5.58 dan std 1.30.
Statistik hingga ini tidak membuktikan latent sehat atau rusak secara musikal.
Mengganti decoder saja belum menyelesaikan hasil; encoder sumber/DiT/kesesuaian
model tetap perlu diperiksa. Jangan mengulang seluruh lagu untuk langkah ini.

Bukti yang masih diperlukan adalah audio, bukan statistik tambahan. Ekspor dari
hasil pembanding yang sudah ada:

```bash
git pull --ff-only
python3 scripts/export_glitch_feedback.py
```

Perintah memilih `decoder_comparison_*` terbaru yang selesai dan membuat
`feedback.zip` pada folder baru yang dicetak. Isinya hanya 10 detik selaras dari
piano sumber, decoder default, decoder CPU, dan manifest setting/posisi/hash.
Default mengambil offset 3–13 detik dalam B (biasanya detik lagu 30–40).
Ketiga file memakai gain pratinjau 0.7 yang sama. Tidak ada inferensi, unduhan,
atau unggahan otomatis; seluruh rekaman dan file latent tidak dimasukkan.
Sumber/hasil lama dipertahankan dan checksum diperiksa sebelum ekspor.

Lampirkan `feedback.zip` di chat untuk menilai jenis artefak yang sebenarnya.
Untuk memilih run atau interval lain, gunakan `--run /path/to/decoder_comparison_xxx`,
`--start` (offset dalam B), dan `--duration` (maksimal 10 detik).
Ekspor diuji dengan audio sintetis: tiga klip tepat 10 detik pada interval sama,
ZIP valid, sumber tidak berubah, dan audio yang berubah hash ditolak. Ini tidak
memvalidasi kualitas bass atau menyatakan penyebab glitch sudah ditemukan.

## Pembanding decoder dengan latent sama (sudah dilakukan pada laporan pengguna)

Laporan pengguna menunjukkan raw B peak 0.999969, RMS 0.298339, tanpa NaN/Inf.
Sesudah resampling tidak ada sampel di luar [-1,1]. Pratinjau dengan gain lebih
rendah tetap glitch. Audit ini tidak mendukung clipping saat konversi sebagai
penyebab utama; raw lama berformat PCM16 sehingga tidak membuktikan semua hal
yang terjadi sebelum file raw ditulis. Penyebab glitch belum dipastikan.

Ada celah observasi pada integrasi: `use_mlx_dit=False` hanya mematikan MLX DiT.
Upstream tetap mencoba mengaktifkan MLX VAE otomatis di Mac. Model DiT MPS memakai
float32, tetapi VAE PyTorch MPS memilih float16; MLX VAE punya dtype terpisah.
Report lama belum merekam jalur VAE yang benar-benar terpanggil atau fallback-nya.

Jalankan satu pembanding terarah:

```bash
git pull --ff-only
python3 scripts/compare_decoders.py --execute
```

Perintah memilih B dari hasil notebook 07 terbaru, menyalin input byte-for-byte,
dan memakai Base/seed/CFG/steps yang sama. **Hanya satu inferensi B baru** diperlukan
karena hasil lama tidak menyimpan latent. Uji menyimpan latent tepat sebelum
decoder, output float default sebelum ekspor, dan panggilan decoder/dtype aktual.
Proses generasi kemudian keluar, sehingga memorinya dilepas.

Proses kedua **hanya memuat VAE lokal di CPU float32** dan membaca latent tersimpan
yang sama sebelum konversi dtype decoder; diffusion tidak diulang. CPU memakai
window 128 frame latent/overlap 32 untuk membatasi memori. Tiling dan dtype juga
bagian dari perbedaan jalur ini, sehingga hasilnya belum mengisolasi MLX saja.
Tidak ada unduhan baru, perubahan seed/metadata, atau generasi ulang seluruh lagu.
CPU decode bisa lebih lambat; waktu/memori nyata di Mac belum dibenchmark.

Di folder `runs/decoder_comparison_*`, dengarkan:

- `old_B_stem.wav`: B sebelumnya.
- `default_decoder_listen.wav`: decoder default dari generasi baru.
- `cpu_decoder_listen.wav`: decoder CPU dari latent yang sama.

Kedua pratinjau memakai gain 0.7 yang sama, hingga 30 detik tanpa padding.
`capture/capture.json` mencatat jalur/dtype/fallback dan hash latent; `comparison.json`
mencatat CPU, waktu, dan hash output. Pemeriksaan checksum melindungi snapshot.

Bila default rusak tetapi CPU bersih, masalah mengarah ke jalur decoder/dtype/
tiling. Bila keduanya rusak, periksa encoder atau hasil DiT yang membentuk latent.
Bila default baru juga bersih, variasi inferensi/sesi perlu diperiksa lagi.
Ini belum perbaikan otomatis; jangan mengulang seluruh lagu sebelum membandingkan
hasil. Tanpa `--execute`, perintah hanya menyiapkan rencana dan cache input.

## Audit raw audio tanpa inferensi (sudah dilakukan pada laporan pengguna)

Hasil pengguna dari notebook 07: B tetap rusak saat diproses pertama, A masih
terdengar bassnya tetapi punya sedikit glitch, dan raw model B juga rusak. Ini
melemahkan dugaan urutan pemanggilan sebagai penyebab utama. Glitch sudah
terdengar sebelum stitching. Belum ada dasar untuk membesarkan chunk lagi.

Periksa file yang sudah ada, **tanpa inferensi atau unduhan baru**:

```bash
git pull --ff-only
python3 scripts/audit_audio.py
```

Perintah memilih hasil notebook 07 terbaru yang selesai. Untuk memilih eksplisit:

```bash
python3 scripts/audit_audio.py --run /path/to/sequence_diagnostics_xxx/reversed
```

Laporan mencatat puncak, RMS, sampel di luar [-1,1], NaN/Inf, format raw,
stem tersimpan, sumber, serta level float setelah resampling sebelum PCM16.
Pemeriksaan ini penting karena ekspor stem lama langsung mengubah output ke PCM16
dengan gain 1: puncak berlebih dapat terpotong saat ekspor atau playback raw float.
Itu kemungkinan yang dapat diukur, belum diagnosis glitch pengguna.
Decoder upstream juga memiliki pembatasan peak sebelum ekspor. Audit mengukur
file aktual dan resampling; jangan menganggap raw pasti melewati batas.

Folder audit baru berisi `report.json` dan `chunk_001_raw_0_headroom.wav` (B),
serta pratinjau A. Gain hanya diturunkan bila diperlukan untuk memberi ruang
hingga -3 dBFS; audio pelan tidak dinaikkan. Pratinjau berasal dari raw yang sama,
tanpa model baru, dan bukan stem evaluasi pengganti. Sumber/hasil lama dipertahankan.
NaN/Inf menyebabkan pratinjau dilewati, bukan diganti diam-diam dengan nol.

Bagikan output Terminal/report dan dengarkan B versi headroom. Bila peak melewati
batas dan versi headroom membaik, clipping berperan. Bila raw sudah terpotong atau
model menghasilkan artefak musik, menurunkan gain tidak memulihkan informasi itu.
Peak normal juga tidak menjamin audio bebas artefak. Source upstream yang dipin
memilih float32 pada jalur PyTorch MPS; jangan menebak bahwa semua error ini fp16.

## Uji urutan pemanggilan (sudah dilakukan pada laporan pengguna)

Pengguna melaporkan mode lagu penuh 30 detik juga rusak setelah potongan pertama.
Memperpanjang konteks belum menyelesaikan glitch. Keberhasilan 30 detik pertama
pada notebook 05 belum menguji input bagian berikutnya atau pemanggilan berulang.

Uji ini tersedia di **[notebooks/07_sequence_diagnostics.ipynb](../notebooks/07_sequence_diagnostics.ipynb)**.
Notebook memilih run `full_song30_*` terbaru yang selesai, atau isi `RUN` dengan
folder bermasalah. Dengarkan sumber, stem tersimpan dan raw model chunk 1 serta 0.
Set `RUN_REPLAY=False` bila hanya ingin mendengarkan cache tanpa inferensi.

Default Run All menjalankan **dua inferensi saja** dalam satu proses model baru,
dengan urutan dibalik **[1,0]**: B (biasanya detik 27–57) lalu A (0–30).
Posisi sebenarnya dicetak sesuai cache. Kedua input disalin utuh dan seed/settings
dipertahankan; tidak ada gain/resampling tambahan, stitching, atau unduhan baru.
Run dan hasil lama dipertahankan. Perbandingan tersimpan di `sequence_diagnostics_*`.

- B bersih saat pertama, A rusak saat kedua: indikasi pengaruh urutan/state runtime.
- B rusak saat pertama, A bersih saat kedua: indikasi isi/konteks B atau generasinya.
- Keduanya bersih: sesi lama atau variasi inferensi mungkin berperan.
- Keduanya rusak: periksa sumber, parameter, dtype/decoder dan raw output.

Pola ini membantu menentukan eksperimen berikutnya; bukan bukti pasti penyebab.
MPS dapat bervariasi meski seed sama. Statistik sinyal tidak menilai musik secara
otomatis. Laporkan kondisi B saat urutan pertama, A saat kedua, dan raw modelnya.
Jika stem/raw tersimpan bersih tetapi lagu gabungan rusak, fokuskan pemeriksaan
pada stitching/mix. Mode lagu penuh belum dinyatakan bebas glitch.

## Eksperimen lagu penuh dengan konteks 30 detik

**Status:** glitch setelah potongan pertama masih dilaporkan. Gunakan notebook 07
di atas sebelum mengulang seluruh generasi notebook 06.

Hasil uji pengguna: chunk 2 sepanjang 10 detik tetap glitch meski diulang dalam
proses baru; generasi 30 detik pertama bersih. Ini mendukung percobaan konteks
lebih panjang, tetapi belum memastikan penyebab glitch atau kualitas seluruh lagu.

Pull kode terbaru, lalu buka
**[notebooks/06_mac_full_song_30s.ipynb](../notebooks/06_mac_full_song_30s.ipynb)**.
Rekaman tetap di `data/full_song/`; jalankan Restart Kernel → Run All.
Notebook memakai Base/MPS/seed 0/CPU offload, konteks 30 detik dan overlap nominal
3 detik. Lagu tepat 180 detik menghasilkan 7 chunk; chunk terakhir mengikuti akhir
lagu agar tetap punya konteks 30 detik. Model dimuat sekali selama proses.

Bobot lokal dipakai tanpa unduhan baru. Run `full_song30_*` terpisah dari hasil
10 detik, sehingga chunk yang glitch tidak dipakai kembali. Run yang terputus
dapat dilanjutkan dengan sumber dan pengaturan sama. Hasilnya `bass_full.wav`
dan `mix_full.wav`, dengan durasi sama seperti sumber hasil decode.

Dengarkan interior setiap chunk dan sambungan di interval 27–30, 54–57, 81–84
detik dan seterusnya. Overlap terakhir bisa lebih panjang. Generasi setiap chunk
masih mandiri sehingga karakter/pola bass dapat berubah. Keberhasilan satu klip
30 detik belum memvalidasi inferensi berulang sepanjang 3 menit. Arsitektur dan
bobot pretrained tetap sama; belum ada training baru.

## Glitch setelah sekitar detik 10 pada mode chunking

Ada laporan hasil nyata dengan glitch berat yang berlanjut setelah detik 10.
Mode notebook 03 masih eksperimen; tes timeline numerik bukan bukti kualitas
musikal. Jangan mengulang seluruh generasi hanya untuk mencoba crossfade lain.

Buka **[notebooks/04_seam_diagnostics.ipynb](../notebooks/04_seam_diagnostics.ipynb)**
untuk membaca cache run yang sudah ada dan mendengarkan chunk terpisah, raw model,
bass gabungan, serta mix. Tidak ada model dimuat, unduhan atau inferensi baru.
Cuplikan dan report disimpan di subfolder diagnostik baru; hasil lama dipertahankan.
Pastikan folder RUN yang tercetak adalah run dengan suara bermasalah.

Chunk kedua biasanya mewakili detik lagu 9–19. Crossfade pertama hanya 9–10;
bagian 10–18 seharusnya identik dengan interior chunk kedua. Diagnostik memeriksa
kesamaan sampel tersebut. Bila glitch sudah ada pada chunk kedua sebelum stitching,
perbaikan harus menyasar generasi/konversi, bukan menebak pengaturan crossfade.
Bila raw model bersih tetapi stem kanonis rusak, periksa konversi/clipping. Bila
chunk kanonis bersih tetapi gabungan rusak, periksa stitching/fase/timing. Statistik
peak/RMS/full-scale tidak otomatis menentukan kualitas atau penyebab glitch.

Untuk menelusuri lanjut, bagikan report dan/atau cuplikan bass sekitar detik 7–22
serta `chunk_001_before_stitch.wav` jika bersedia. Tidak perlu seluruh audio asli.
Belum ada klaim bahwa glitch ini sudah diperbaiki tanpa pemeriksaan audio nyata.

### Chunk 1 bersih tetapi berbeda karakter; chunk 2 glitch

Perbedaan karakter antar-generasi mandiri dapat terjadi meski input memakai seed
yang sama. Glitch yang sudah terdengar pada chunk terpisah perlu ditelusuri pada
generasi atau konversinya. Crossfade tidak memperbaiki interior chunk yang rusak.

Gunakan **[notebooks/05_context_comparison.ipynb](../notebooks/05_context_comparison.ipynb)**
setelah mendengarkan notebook 04. Pilih run bermasalah melalui `RUN` jika bukan
hasil terbaru. Restart Kernel → Run All menjalankan dua inferensi baru:

1. Ulang input chunk 2 yang sama persis, seed dan pengaturan sama, dalam proses baru.
2. Generasi 30 detik pertama sekaligus, tanpa stitching atau gain tambahan.

Bobot lokal dipakai kembali tanpa unduhan; hasil pembanding berada di folder baru.
Tes 30 detik membutuhkan memori/waktu lebih besar dan belum diuji pada Mac 24 GB.
Untuk memeriksa persiapan/rencana saja, set `DRY_RUN = True`.

Bila chunk baru bersih, selidiki state handler atau variasi inferensi; itu belum
membuktikan penyebab tertentu. Bila hanya konteks 30 detik yang bersih, konteks
pendek/input lokal mungkin berperan. Bila keduanya rusak, periksa audio mentah,
sumber, seed, decoder dan clipping sebelum memperpanjang generasi. Eksperimen ini
belum membuktikan bahwa inferensi 3 menit muat atau menghasilkan bass konsisten.

## Bass untuk lagu penuh sekitar 3 menit

**[notebooks/03_mac_full_song.ipynb](../notebooks/03_mac_full_song.ipynb)**
mempertahankan mode chunk 10 detik untuk reproduksi/diagnosis, memakai seluruh
rekaman hingga batas 10 menit POC. Setelah uji notebook 05 menunjukkan 30 detik
bersih, gunakan notebook 06 di atas. Instruksi mode lama:

1. Pull kode terbaru.
2. Taruh satu file rekaman sendiri di **`data/full_song/`**. Folder ini dipisah
   dari input notebook 02 agar rekaman uji lama tidak terpilih. WAV/MP3/M4A didukung.
3. Buka notebook 03, lalu Restart Kernel → Run All.
4. Hasil berada di folder run yang dicetak notebook: **`bass_full.wav`** dan
   **`mix_full.wav`**, dengan panjang sama seperti audio yang didecode.

Base/MPS/seed 0/CPU offload sudah diatur. Runtime dan bobot terverifikasi yang
sudah tersedia dipakai kembali; notebook ini tidak memasang atau mengunduh model.
Untuk rekaman tepat 180 detik dibuat 20 potongan 10 detik dengan overlap nominal
1 detik (potongan akhir mengikuti akhir lagu). Model dimuat sekali per proses.
Runtime total dapat mendekati banyak kali waktu generasi satu klip, tergantung
hardware, warm-up dan isi audio; belum ada benchmark 3 menit pada Mac pengguna.

Sumber/profil/manifest model yang sama memakai folder run yang sama. Chunk selesai
dipakai ulang setelah dicek hash; chunk gagal dicoba dalam folder attempt baru.
Hasil/cache yang berubah ditolak, bukan dipakai diam-diam. Jangan menjalankan dua
generasi bersamaan untuk run sama; lock OS melindungi folder itu. Hasil final yang
sudah selesai diverifikasi dan dipakai ulang. Run yang gagal tetap menyimpan
metadata dan attempt-nya. Mengubah seed/instrument menghasilkan run terpisah.

**Batas musikal:** setiap chunk digenerasi mandiri, bukan inferensi dengan konteks
lagu 3 menit sekaligus. Crossfade linear menjaga kontinuitas audio dan panjang
sampel, tetapi pola bass/groove, artikulasi dan fase dapat berbeda atau melemah
pada sambungan. Dengarkan harmoni dan transisi di sekitar 9,18,27 detik dan
seterusnya. Tidak ada jaminan lagu penuh mengikuti feel secara konsisten.

Pengujian menyiapkan audio sintetis 180 detik dan membuktikan round-trip timeline
20 chunk tanpa perubahan sampel untuk stem identitas. Runner juga diuji dengan
model palsu yang gagal pada chunk kedua lalu melanjutkan tanpa meregenerasi chunk
pertama. Tes tersebut tidak memvalidasi kualitas bass maupun inferensi MPS panjang.

## Langsung jalan untuk Mac Apple Silicon 24 GB

Buka **[notebooks/02_mac_24gb_full.ipynb](../notebooks/02_mac_24gb_full.ipynb)**.
Profil di `configs/mac-24gb.json` sudah berisi Base, MPS, bass, seed 0 dan CPU
offload. Memerlukan macOS 14 atau lebih baru untuk dependency MLX pada lockfile
upstream. Tidak perlu mengubah tiga flag instalasi/unduhan/generasi notebook lama.

1. Pull kode terbaru. Jika notebook lama berubah, simpan lewat Git stash dahulu.
   Notebook 02 adalah file baru, sehingga notebook 01 beserta konfigurasi lokalnya
   dapat tetap disimpan dalam stash.
2. Letakkan satu rekaman gitar/piano minimal 10 detik di `data/`. Folder ini
   dibuat oleh sel pertama; bisa dibuat sendiri di file browser JupyterLab.
   Satu file WAV/MP3/M4A dipilih otomatis, atau gunakan WAV bernama `input.wav`.
3. Jalankan **Restart Kernel → Run All** pada notebook 02.
4. Notebook memeriksa sumber dan tool, mempersiapkan klip, menginstal runtime
   Python 3.11 terpisah, memverifikasi MPS, mengunduh/verifikasi aset sekitar
   14.88 GB, lalu mencoba generasi dan menampilkan stem/mix.

Runtime dan bobot yang sudah ada dipakai kembali setelah diperiksa. Instalasi
pertama memerlukan jaringan dan ruang tambahan; bukan proses yang sudah selesai
hanya karena notebook berhasil dibuka. Tidak ada audio sintetis sebagai pengganti
hasil model. Sumber yang belum tersedia menghentikan notebook sebelum instalasi
atau unduhan. Parameter SOURCE/START/CONDITIONING tetap bisa diubah jika diperlukan.

`DRY_RUN=False` adalah default penuh. Mode True hanya untuk validasi pipeline dan
selalu menandai instalasi, unduhan dan inferensi sebagai tidak dijalankan.
Notebook ini dipersiapkan untuk Mac-mu; inferensi MPS nyata tetap belum diverifikasi
oleh agen Linux. Kirim traceback jika instalasi atau inferensi gagal.

## Mulai di komputer JupyterLab

```bash
git clone https://github.com/nenz24/StemsGenerating-Models.git
cd StemsGenerating-Models
jupyter lab
```

Untuk diagnostik aman/lintas platform, buka **[notebooks/01_quality_poc.ipynb](../notebooks/01_quality_poc.ipynb)** dan jalankan
sel dari atas. Jika sudah mempunyai checkout, jalankan `git pull --ff-only`
setelah menyimpan perubahan lokal. Jika JupyterLab belum terpasang, instal
`requirements-notebook.txt` di environment notebook sendiri.

1. Jalankan diagnostik. Tidak perlu tahu GPU sebelumnya: lihat `reports/hardware.json`.
2. Run All pertama menggunakan fixture 10 detik, tanpa instalasi ML, unduhan bobot,
   atau inferensi. Perlu `git`, `ffmpeg`, `ffprobe`, dan Python 3.11+.
3. Ganti `SOURCE` dengan rekaman gitar/piano sendiri, pilih bass atau drums dan
   seed 0/1/2. Bisa memakai ekspor WAV+JSON SongMap, atau audio biasa.
4. Instal runtime terpisah, unduh model terpilih, lalu aktifkan `GENERATE` lewat
   sakelar notebook. Sel-sel itu hanya berjalan saat kamu mengaktifkannya.
5. Dengarkan stem/mix dan bandingkan Base dengan XL pada sumber/seed sama.

## Mac Apple Silicon / Jupyter Homebrew

Jika diagnostik menunjukkan `Darwin / arm64`, jalankan di Terminal Mac:

```bash
brew install ffmpeg
cd /path/to/StemsGenerating-Models
git pull --ff-only
```

Simpan salinan notebook yang sudah diedit sebelum pull jika Git melaporkan
perubahan lokal. Restart kernel dan buka ulang notebook versi terbaru. Helper
mencari `ffmpeg`/`ffprobe` di PATH serta `/opt/homebrew/bin` dan `/usr/local/bin`.
Bila belum terpasang, notebook berhenti dengan petunjuk instalasi sebelum
persiapan audio, tanpa traceback subprocess yang panjang.

Python 3.15 bawaan JupyterLab tidak perlu diganti. Sel `INSTALL_RUNTIME` memakai
`uv` untuk menyiapkan **Python 3.11 terpisah** bagi dependency model. `DEVICE="auto"`
memilih MPS dari Python model pada Mac, CUDA pada NVIDIA, dan menolak fallback
CPU diam-diam. Diagnostik pada kernel yang belum memiliki PyTorch belum bisa
memastikan MPS; sel instalasi memeriksanya lagi di environment model.

Pada Mac mulai dengan `MODEL="acestep-v15-base"`, `SEEDS=[0]`, `OFFLOAD=True`.
Lihat RAM terpadu dalam laporan diagnostik dan ukur satu generasi sebelum XL.
Ruang disk yang besar tidak memastikan model muat di RAM. Jalur MPS memakai
PyTorch yang didukung API upstream; MLX DiT belum diaktifkan di adapter ini.
Metrik MPS mencatat **current** allocator/driver memory, bukan peak atau total
memori proses. Keberhasilan Mac tidak membuktikan Core AI/iPhone.

## Melihat progress di JupyterLab

Notebook meneruskan stdout/stderr subprocess melalui output Python kernel.
Unduhan menampilkan nomor file, nama dan ukuran. Progress transfer HF ditampilkan
jika client memberikannya; baris checksum menunjukkan byte file yang benar-benar
dibaca dan diverifikasi. Tahap menyalin dan total aset yang selesai diverifikasi
juga dicatat. Total aset terverifikasi bukan jumlah byte jaringan yang diunduh.

Jika child tidak menulis output selama 15 detik, heartbeat menunjukkan proses
masih hidup dan durasi tanpa output. Itu tidak membuktikan jaringan bergerak dan
bukan persentase unduhan. Exit nonzero tetap menjadi error notebook.

Untuk mengambil pembaruan saat notebook sudah dijalankan, simpan perubahan lokal
ke stash jika pull terhalang. Tunggu sel unduhan aktif selesai sebelum membuka
ulang notebook yang baru dan memulai proses berikutnya. File lengkap yang lolos
checksum dipakai kembali. Pemulihan transfer parsial mengikuti cache HF.

## Error matplotlib_inline saat mulai generasi

Jika semua 34 aset sudah terverifikasi tetapi impor backend gagal dengan
`MPLBACKEND=module://matplotlib_inline.backend_inline`, pull perbaikan terbaru.
Subprocess model sekarang memakai backend standar `Agg` dan cache konfigurasi
lokal `.runtime-cache/matplotlib`, sebelum dependency Lightning/TorchMetrics
mengimpor Matplotlib. Kernel Jupyter dan paketnya tetap terpisah.

Lakukan Restart Kernel → Run All pada notebook 02 untuk membuat folder run baru.
Folder hasil gagal dipertahankan untuk diagnosis; bobot lengkap yang sudah ada
akan diverifikasi/dipakai kembali, tanpa diunduh ulang. Pengujian mengulang
kegagalan impor dengan Matplotlib 3.10.8 dalam environment terpisah dan membuktikan
impor serta rendering PNG berhasil dengan Agg. Ini belum membuktikan seluruh
inferensi MPS pada Mac berhasil.

## Jika file Base tertahan berjam-jam

Nomor [32/34] dengan `acestep-v15-base/model.safetensors` berarti transfer bobot
Base sekitar 4.46 GiB belum selesai. Itu berbeda dari inferensi MPS. Heartbeat
menandakan proses hidup saja; tanpa log lain belum dapat dibedakan jaringan,
transfer Xet, atau penantian lock cache.

Downloader sekarang memakai jalur **HTTP resmi HF** secara default, dengan Xet
dinonaktifkan pada subprocess download, timeout metadata 20 detik dan HTTP read
60 detik. Timeout tersebut bukan batas waktu keseluruhan transfer. TLS dan
checksum tetap aktif. Cache HTTP terpisah menghindari lock/partial Xet yang
tertahan tanpa menghapusnya. File target yang sudah diverifikasi digunakan lagi;
file Base yang belum selesai dapat mulai diunduh dari awal pada cache HTTP baru.

Hentikan sel unduhan lama memakai **Interrupt Kernel** sebelum menjalankan
percobaan baru. Pull kode terbaru, lalu jalankan ulang sel unduhan notebook 02.
Perubahan default transport ada di script sehingga tidak perlu mengubah notebook.
Pilihan Xet tetap tersedia lewat `--transport xet` untuk eksperimen terpisah.

Untuk bukti tambahan, dari Terminal di repository jalankan:

```bash
python3 scripts/diagnose_download.py --interval 10
```

Ini hanya membaca metadata target/cache selama 10 detik dan memeriksa proses
yang membuka file lock bila `lsof` tersedia. Tidak membaca token/log autentikasi,
menghapus cache, atau menghentikan proses. Ukuran/mtime/alokasi disk bukan
pengukuran byte jaringan; lock yang ada belum berarti sedang dipegang. Kirim
outputnya jika retry masih tertahan. Jangan menghapus cache atau lock secara massal.

## Perbaikan permission uv / folder home

Jika instalasi lama gagal pada `~/.local/share/uv/python`, cukup pull kode terbaru
lalu jalankan ulang sel instalasi notebook 02. Checkout source dan `.bootstrap`
yang sudah ada dipakai kembali; tidak perlu sudo, chown, atau reinstall Jupyter.

Python unduhan uv sekarang disimpan di `.runtime-cache/uv/python`, cache uv/pip
juga di `.runtime-cache/`, dan environment model tetap di
`vendor/ACE-Step-1.5/.venv`. Cache unduhan model Hugging Face/Xet diarahkan ke
`.runtime-cache/huggingface/`. Folder cache diabaikan Git. HOME, token, proxy dan
CA trust yang diwariskan tidak diganti; checksum/TLS tetap diperiksa.

## Kandidat kualitas

- **XL Base:** dokumentasi upstream menempatkannya sebagai opsi kualitas lebih
  tinggi; gunakan sebagai kandidat utama jika resource mencukupi. Rekomendasi
  notebook: GPU NVIDIA dengan kelas 24 GB (laporan driver ≥24000 MiB VRAM). Ini belum pengukuran hardwaremu.
- **Base:** baseline pembanding yang lebih ringan. CPU offload tersedia.
- Keduanya: mode `lego`, 50 langkah, CFG 7, satu instrumen per job, planner mati.
  Angka kecepatan Turbo tidak dipakai untuk menaksir waktu lego.

Bobot yang benar-benar diunduh menurut metadata resmi: sekitar **14.88 GB untuk
Base** atau **30.05 GB untuk XL** beserta komponen utama upstream. Sediakan
headroom disk setidaknya dua kali ukuran unduhan ditambah 5 GiB. Komponen Turbo
serta LM ikut disediakan untuk memenuhi pemeriksaan initializer upstream; LM
planner tidak diaktifkan. Runtime/dependency ML memerlukan ruang tambahan.

Model pretrained dipin pada revisi HF dan source Git tertentu. Semua aset dicek
terhadap hash penerbit sebelum dipakai. Runtime memakai salinan kode model agar
sinkronisasi kode oleh upstream tidak mengubah aset unduhan asli.

## Bridge GitHub ↔ lokal

GitHub menyinkronkan **kode, notebook tanpa output, dan feedback kecil**. Ini tidak
memberi agen akses langsung ke filesystem lokal. Jangan push bobot, rekaman,
cache, `.env`, atau keluaran audio; semuanya sudah diabaikan oleh `.gitignore`.
Simpan input di `data/`, hasil di `runs/`. Tinjau perubahan sebelum push.

Untuk memberi informasi hardware, cukup isi berkas di `feedback/` berdasarkan
`reports/hardware.json`, commit dan push. Contoh format tersedia di
[feedback/README.md](../feedback/README.md). Setelah ada feedback atau traceback,
runner bisa disesuaikan dengan hardware JupyterLab yang sebenarnya.

## Status validasi

91 pengujian audio/CLI plan, helper, profil lengkap Mac, cache lokal, progress dan kontrak MPS/CUDA lulus di Linux tanpa GPU. Uji guidance memeriksa parameter
nyata yang dikirim ke adapter, default 7, plan 1 tanpa model, penolakan input/
baseline berubah, dan isolasi cache CFG 7 dari CFG 1. Kualitas CFG 1 belum diuji
dengan bobot nyata di cloud ini.
Pemeriksaan source roundtrip menguji
snapshot interval yang sama tanpa gain kedua, penolakan input berubah, CLI tanpa
model, dan adapter decoder bersama dengan model palsu. Bobot encoder nyata tidak
dijalankan pada tes cloud tersebut.
Pembanding decoder menguji snapshot
latent tanpa perubahan, jejak fallback/dtype, tiling/tail tanpa kehilangan sampel,
flag opt-in pada engine nyata dengan model palsu, dan ekspor WAV pembanding CPU.
CPU/MPS/MLX dengan bobot model nyata belum diuji di cloud ini.
Audit audio diuji dengan raw float
WAV sintetis yang melebihi batas amplitudo: clipping terukur dan pratinjau hanya
diattenuasi, dengan input lama tetap utuh. Ini bukan bukti penyebab glitch pengguna.
Termasuk round-trip audio
180 detik pada konteks 30 detik/overlap 3 detik, isolasi cache lama, dan resume
runner dengan model palsu pada konteks 10 serta 30 detik. Tes tersebut tidak
memvalidasi kualitas musik. Seluruh 9
sel kode notebook 01 juga dieksekusi tanpa error, tanpa mengunduh bobot
atau menghasilkan stem model. Ketujuh sel kode notebook 02 diuji dalam mode
dry-run yang ditandai jelas dengan persiapan audio nyata.
Seluruh empat sel kode notebook 05 juga dieksekusi dalam mode dry-run dengan
cache audio sintetis: input chunk identik dan rencana 30 detik tanpa inferensi.
Kelima sel kode notebook 06 lolos dry-run dengan audio sintetis 180 detik,
merencanakan 7 chunk tanpa unduhan atau inferensi model.
Keempat sel notebook 07 lolos mode cache-only dengan audio sintetis. Uji engine
memakai model palsu memeriksa urutan [1,0], satu handler, kesesuaian input/mix,
dan penyimpanan hasil pertama saat panggilan kedua gagal.
Rencana dependency
macOS 14 arm64/Python 3.11 berhasil diperiksa memakai uv 0.10.0 dan lockfile
upstream: 125 paket dapat direncanakan, tanpa memasang dependency ML tersebut. Checkout source upstream pada commit yang dipin
juga berhasil diverifikasi.
Pemilihan MPS dan pemanggilan API diuji dengan mock. Cloud ini tidak menjalankan
inferensi MPS/CUDA nyata atau instalasi dependency ML penuh. Pengguna melaporkan
generasi 30 detik pertama bersih, tetapi bagian selanjutnya pada mode lagu penuh
30 detik masih glitch. Default notebook tidak mengklaim keberhasilan generasi.
Tidak ada gerbang review lisensi untuk POC sesuai permintaan; catatan singkat ada
di [docs/model-notes.md](model-notes.md).

Pengujian:

```bash
python3 -m unittest discover -s Tests/StemEvaluationPython -v
```

STAGE tetap tersedia sebagai adapter riset CLI dari pekerjaan sebelumnya;
notebook pertama berfokus pada ACE-Step. Konversi Core AI, benchmark iPhone,
training/fine-tuning, dan sinkronisasi hasil otomatis belum diimplementasikan.
