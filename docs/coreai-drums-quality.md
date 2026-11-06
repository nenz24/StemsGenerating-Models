# Evaluasi glitch dan drum yang kurang terdengar

Run `3A1FD8B9` selesai 50/50. Klarifikasi pengguna: **groove drum sudah sesuai**,
tetapi banyak glitch menutupi suara drum. Penerimaan groove terpisah dari
**kualitas audio yang belum diterima**. Keluhan kurang terdengar merujuk pada
suara yang tertutup glitch, bukan bukti bahwa gain stem perlu dinaikkan.
Pertahankan preset, groove dan parameter generasi selama diagnosis artefak.
Keputusan lama untuk mengabaikan glitch bass ringan tidak berlaku untuk drum. Target lagu
penuh bass/drum/gitar tetap; drum perlu diterima sebelum melanjutkan tahap itu.

[Audit laporan dan WAV](../reviews/coreai-iphone17pro-drums-quality-rejection/README.md)
memisahkan hasil eksekusi dari kualitas. Drum mentah memiliki peak +2,65 dBFS;
Float32 menyimpannya tanpa pemotongan, tetapi playback dapat terdistorsi. Mix
memiliki sample peak −0,18 dBFS dan cocok persis dengan rumus mix yang ada.
Jadi clipping drum solo belum menjelaskan glitch mix. Sebanyak 87,54% energi
drum berada di 20–150 Hz. Ini pengukuran spektrum, bukan bukti audibilitas,
kesalahan instrumen atau penyebab glitch. Belum ada diagnosis penyebab pasti.

## Hasil terbaru: gain source0,7 belum menghilangkan glitch

[Laporan uji gain source](../reviews/coreai-drums-source-gain-result/README.md)
selesai 50/50 dengan sembilan pemeriksaan kontrol error nol. Input gain0,7
benar-benar digunakan; hash PCM kandidat yang dicatat juga cocok dengan hasil
perkalian PCM sumber terverifikasi. Conditioning identik dengan kontrol.
Pengguna menyatakan **"glicthnya ada lagi dipergantian pukulan"**. Tidak ada
penerimaan kualitas baru untuk kandidat generasi ini. Gain source saja belum
menyelesaikan masalah; jangan mengulang uji atau mencari gain/cap lain.

### Langkah aktif: satu acuan host30 detik dengan capture

[Alat acuan30 detik](../reviews/coreai-drums-reference30-plan/README.md)
memakai persiapan dan setting Colab yang sudah diterima: piano yang sama,
source32kHz PCM16 pada gain0,7, resampling handler ke48kHz, durasi/metadata30s,
`[Instrumental]`, tanpa kit reference, APG cap2,5/guidance7/50langkah. Ini
pemeriksaan jalur acuan, **bukan uji durasi saja**: beberapa perbedaan dari
preview10,24s dicocokkan sekaligus. CPU/CUDA RNG dan versi runtime tetap berbeda;
tidak ada klaim bit-exact atau numerical PASS dari hasil dengar.

Tutup Xcode. Dari folder repo di Terminal Mac:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/generate_coreai_drum_reference.py --execute \
  --out .runtime-cache/drums-reference30-v1 &&
open .runtime-cache/drums-reference30-v1
```

Saat pemilih file terbuka, pilih **`piano.mp3` asli** yang dipakai pada baseline
Colab, bukan output drum atau mix. Hash rekaman dan PCM acuan diverifikasi.
Bila file canonical30s yang tepat sudah tersedia, opsi `--prepared-source`
dapat dipakai sebagai pengganti `--source`; gain tidak diterapkan dua kali.
Tidak perlu mengunduh tautan audio cloud, memasang torchaudio atau mengulang
persiapan model. Bobot yang sudah ada diverifikasi lokal; tidak ada unduhan
checkpoint atau fallback native/accelerator.

Ini satu generasi **30 detik/50langkah pada CPU Mac**, lebih besar dari uji
10,24s sebelumnya. Tunggu sampai selesai. Output harus folder baru di luar
baseline/ModelCandidates. Source posterior/epsilon/mask, token/conditioning,
noise/schedule, dan conditional/unconditional/guided/momentum/latent pada
langkah1/5/10/25/50 disimpan untuk audit. Tidak ada reset ke gold, sweep, stitch,
EQ atau limiter. Model/cache/baseline BF16 dan batas abs/rel/RMSE tetap.

Setelah **Ready**, putar pada volume/output sama:

1. `01-colab-reference-headroom.wav` — drum Colab yang diterima.
2. `02-host-reference-headroom.wav` — hasil host baru30 detik.
3. `03-host-piano-drums-headroom.wav` — mix acuan0,5piano+0,5drum untuk menilai groove.

Kirim `report.json` dan nilai glitch serta groove kandidat02/03. WAV01/02 memakai
satu attenuasi bersama. Mix acuan ini tidak mengganti gain mix produksi iPhone.
Jika verifikasi atau inference gagal, report/error dipertahankan; jangan
mengganti referensi atau melonggarkan batas. Belum perlu Generate/build iPhone.
Kandidat30s tidak otomatis diterima atau diaktifkan menjadi model native.

## Hasil kontrol: uji VAE perkusi terdengar bersih

Pengguna mengirim [laporan roundtrip drum Colab](../reviews/coreai-drums-percussion-roundtrip-clean/README.md)
dan menyatakan **"sudah lebih baik dan glitch tidak terdengar"**. Identitas
eksperimen, arsip/drum sumber dan checkpoint cocok: satu encoder, satu decoder,
tanpa langkah generasi. Hasil ini berlaku pada pembanding perkusi terakhir;
pengguna tidak merinci penilaian 01/02 secara terpisah. Field otomatis
`listening_assessment: NOT EVALUATED` di report asli dipertahankan, sedangkan
feedback manusia dicatat terpisah. WAV Mac tidak diunggah pada pesan ini;
hash output adalah catatan report, belum dihitung ulang dari file tersebut.

Uji dengar codec perkusi selesai dan **tidak perlu diulang**. VAE FP32, baseline
BF16, cap produksi 2,5 dan toleransi tetap. Fokus berikutnya adalah conditioning
dan lintasan generasi terhadap acuan Colab. Rekonstruksi drum Colab yang bersih
belum menjadi hasil dengar baru untuk kandidat APG atau penerimaan drum iPhone.
Perilaku VAE pada latent lain juga belum dikesampingkan sepenuhnya.

[Input acuan 30 detik ditemukan](../reviews/coreai-drums-percussion-roundtrip-clean/source-preparation-audit.json)
dengan hash WAV persis sama dengan baseline yang diterima. Canonicalization
ulang saat ini menghasilkan PCM16 identik dengan input acuan, walaupun hash
WAV berbeda pada byte di luar PCM. Gunakan arsip input yang identik untuk
pemeriksaan acuan; jangan mengganti hash baseline. [Rencana pemeriksaan
generasi](../reviews/coreai-drums-percussion-roundtrip-clean/generation-reference-plan.json)
mengunci input, durasi/metadata, conditioning, RNG nyata dan parameter Colab.
Ini belum dieksekusi sebagai generasi baru atau uji isolasi durasi saja.
Jangan memperpanjang chunk app atau mengadopsi cap kandidat berdasarkan uji
codec ini. Uji berikutnya di host mengisolasi gain source sebelum reproduksi
acuan 30 detik yang mengubah beberapa bagian sekaligus.

## Riwayat gain source (selesai, glitch masih terdengar)

Uji ini sudah selesai. Command berikut disimpan untuk reproduksi; jangan diulang.

[Uji gain source](../reviews/coreai-drums-source-gain-plan/README.md) hanya
mengalikan PCM piano tersimpan dengan `0,7` **sebelum VAE encoder**, mengikuti
nilai gain tetap Colab. PCM/crop/resampler yang tersimpan dipertahankan; ini
bukan persiapan 32 kHz PCM16 atau reproduksi durasi 30 detik Colab. Seluruh
conditioning teks/lyric/timbre tetap, termasuk lirik kosong kontrol. Durasi
10,24 detik, epsilon source tersimpan, mask, noise/schedule diffusion, 50 langkah,
guidance 7 dan cap asli 2,5 tetap. Kandidat APG cap 1,46059 tidak dipakai dalam
uji ini agar perubahan input dapat dinilai sendiri. Baseline BF16 dan toleransi
tidak berubah; gain ini tidak mengubah volume playback atau gain mix.

Tutup Xcode, buka Terminal di folder repo Mac, lalu jalankan:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/compare_coreai_drum_source_gain.py \
  --diagnostic-trajectory --execute \
  --out .runtime-cache/drums-source-gain-v1 &&
open .runtime-cache/drums-source-gain-v1
```

Ini menghitung **satu kandidat baru 50 langkah pada CPU Mac**. Dua panggilan
source encoder (replay kontrol dan kandidat) mendahuluinya. Model/cache yang
sudah ada dipakai dengan verifikasi hash; tidak ada unduhan checkpoint, export
native, build app atau aktivasi profil. Folder output harus baru. Tanpa
`--execute`, script hanya menulis plan; gunakan folder baru setelah membuat plan.

Pasangan WAV yang diuji pada volume/output sama:

1. `01-control-source-gain1-headroom.wav`
2. `02-candidate-source-gain07-headroom.wav`

Keduanya berisi drum dengan satu attenuasi bersama. Pengguna sudah mengirim
report dan melaporkan glitch saat pergantian pukulan masih ada.
Ini kandidat yang benar-benar dihitung dari input berbeda, terpisah dari uji
VAE perkusi bersih yang sudah selesai. Tidak perlu menekan Generate di iPhone.

Empat gate source posterior mean/std/latent/context, conditioning/mask identik,
langkah pertama kontrol dan decoder harus lolos batas asli. Bila gagal, script
berhenti dan mempertahankan report; jangan melonggarkan gate atau mengulang
eksperimen yang sama. Provenance kontrol tetap `bounded_diagnostic`; keberhasilan
replay bukan full trajectory/native PASS. Kandidat belum diadopsi sebelum hasil
dengar dan pemeriksaan numerik berikutnya.

## Riwayat APG: drum lebih jelas, glitch saat pergantian pukulan

[Run kandidat APG](../reviews/coreai-drums-scaled-apg-partial-result/README.md)
selesai 50/50 dengan lima pemeriksaan kontrol error 0 dan conditioning identik.
Clipping kandidat aktif pada 48/50 iterasi. Pengguna mendengar drum lebih jelas,
tetapi glitch muncul saat pergantian pukulan sepanjang audio. Kandidat disimpan
sebagai kemajuan parsial; kualitas drum belum diterima penuh. Baseline BF16,
cap produksi 2,5 dan toleransi abs/rel/RMSE dipertahankan.

Run ini satu waveform 10,24 detik, tanpa stitching. Jangan menganggap kata
"transisi" sebagai sambungan chunk atau menerapkan crossfade. Telemetri APG
berindeks iterasi diffusion, bukan detik audio; angka itu tidak dapat melokasikan
glitch serangan drum. Tidak perlu Generate ulang atau menurunkan cap lagi dulu.

ZIP kandidat APG sudah diterima dan [diaudit](../reviews/coreai-drums-apg-waveform-audit/README.md).
Kelima file cocok; kandidat WAV adalah transform gain dari audio mentah secara
bit-exact. Replay latent kandidat pada VAE FP32 juga PASS pada batas asli
(normalized RMSE `1,75e-6`). Tidak ditemukan frame stereo nol/identik berturutan,
tetapi ini bukan detektor semua jenis glitch atau sertifikat kualitas.
Serangan tajam sekitar 3,568 detik muncul pada kontrol dan kandidat; statistik
transien tidak cukup untuk menyebutnya salah atau menghapus pukulan.

### Riwayat pembanding VAE perkusi (sudah selesai, terdengar bersih)

Piano bersih belum membuktikan kualitas rekonstruksi **perkusi**. Satu kontrol
baru memakai drum Colab 30 detik yang sudah diterima: resample WAV arsip
32 kHz PCM16 ke 48 kHz, ambil 10,24 detik pertama, lalu satu encoder dan satu
decoder VAE FP32. Tidak ada DiT/50 langkah, unduhan checkpoint, export native
atau perubahan preset. Ini rekonstruksi lossy dari WAV ekspor, bukan replay
latent Colab asli atau reproduksi full pipeline Colab.

Script ini sudah benar-benar dijalankan di CPU cloud dan kemudian di Mac.
Pengguna sudah mendengar hasil lokal tanpa glitch. Command berikut disimpan
untuk reproduksi, bukan permintaan mengulang. Dua WAV dibuat langsung di Mac
karena tautan audio cloud sebelumnya tidak dapat diunduh pengguna:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/listen_coreai_drum_roundtrip.py \
  --out .runtime-cache/coreai-drums-codec-listening-v1 &&
open .runtime-cache/coreai-drums-codec-listening-v1
```

Pasangan **`01-approved-drums.wav`** dan **`02-approved-drums-vae.wav`** berisi
drum saja, dengan satu gain attenuasi bersama. Hasil dengar terbaru tidak
melaporkan glitch pada uji ini; fokus investigasi kini generasi/conditioning.
Kesimpulan ini memakai feedback, bukan angka korelasi/SNR saja, dan belum
menentukan penyebab tunggal glitch generasi. Sumber arsip bukan
drum mentah sebelum VAE Colab dan roundtrip menambahkan kompresi lossy lagi.

Checkpoint dan arsip diperiksa SHA-256; folder output harus baru dan di luar
baseline/ModelCandidates. Gagal verifikasi berhenti dan mempertahankan report.
Baseline BF16, cap produksi 2,5, parameter generasi serta toleransi tetap.
Kandidat APG belum diadopsi atau dinyatakan selesai diperbaiki.

### Riwayat pengumpulan kandidat APG (sudah selesai)

Command berikut hanya disimpan sebagai catatan; ZIP sudah diterima:

```bash
test ! -e .runtime-cache/drums-apg-review-v1.zip &&
zip -j .runtime-cache/drums-apg-review-v1.zip \
  .runtime-cache/drums-apg-norm-v1/{report.json,01-control-original-apg-headroom.wav,02-candidate-scaled-apg-headroom.wav,candidate-audio.f32le,candidate-latent.f32le} &&
open -R .runtime-cache/drums-apg-review-v1.zip
```

Ini hanya membaca lima hasil
yang sudah ada dan membuat arsip baru; tidak memuat model atau menyertakan
checkpoint. Output lama tidak ditimpa. Jika folder hasil uji bernama berbeda,
sesuaikan hanya path folder itu.

Hash dan transform gain sudah diperiksa, serta serangan drum kandidat dan
kontrol sudah dibandingkan di sepanjang klip. Drum memang memiliki serangan tajam/broadband;
statistik spike saja tidak boleh dipakai untuk menghapus pukulan. Belum ada
dasar untuk de-clicker, EQ, limiter, fade, cap lain atau eksperimen model baru.

## Riwayat: kedua piano bersih dan uji APG (sudah selesai)

Pengguna sudah mendengarkan dua WAV lokal dan melaporkan **keduanya bersih,
hanya suara piano**. Itu memang hasil yang diharapkan dari rekonstruksi sumber.
[Laporan](../reviews/coreai-drums-source-clean-apg-plan/README.md) cocok dengan
input sebelumnya: satu decode, gain 1, tanpa encoder/DiT. VAE dipertahankan;
glitch drum masih ditelusuri pada conditioning/generasi bersama.

Uji berikutnya hanya mengubah **batas norm APG pada kandidat host**, dengan
piano, conditioning/lyrics, noise, durasi, 50 langkah, guidance 7 dan decoder
tetap. APG membatasi norm sepanjang waktu; untuk klip lebih pendek, batas RMS
per frame menjadi lebih besar pada cap 2,5 yang sama. Satu nilai kandidat
diturunkan dari referensi 30 detik yang diterima: `2,5*sqrt(256/750) ≈ 1,46059`.
Ini hipotesis terarah, belum bukti perbaikan dan bukan reproduksi konteks 30 detik.

**Baseline BF16 dan cap asli 2,5 tetap. Toleransi numerik abs/rel/RMSE tetap.**
Candidate cap adalah parameter generasi terpisah, bukan pelonggaran gate.
Conditioning kandidat harus identik byte dengan kontrol, dan replay kontrol
tetap harus lolos batas asli. Tidak ada limiter/EQ atau perubahan gain mix.

Command berikut disimpan untuk reproduksi; uji APG ini sudah selesai dan tidak
perlu diulang. Dari folder repo Mac yang sama:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/compare_coreai_drum_apg.py \
  --diagnostic-trajectory --execute \
  --out .runtime-cache/drums-apg-norm-v1 &&
open .runtime-cache/drums-apg-norm-v1
```

Ini menghitung **satu kandidat 50 langkah di CPU Mac**, dengan checkpoint/cache
yang sudah ada; bukan uji VAE satu decode lagi. Tidak ada download bobot baru,
export model, build app, atau aktivasi kandidat di iPhone. Output harus baru.
Tanpa `--execute` hanya plan dibuat; pakai nama folder baru untuk eksekusi setelah
plan. Bila gate/checkpoint gagal, simpan report/error dan berhenti.

Setelah selesai, dengarkan pada volume/output yang sama:

1. `01-control-original-apg-headroom.wav`
2. `02-candidate-scaled-apg-headroom.wav`

Keduanya berisi drum, dengan satu gain attenuasi bersama. Nilai apakah glitch
berkurang **dan** groove tetap sesuai, bukan hanya apakah lebih pelan.
Bagikan `report.json` dan hasil dengar. Report mencatat clipping dan RMS update
guidance selama kandidat; angka itu bukan sertifikat kualitas. Jika tidak
membantu, jangan ulang atau mencari cap lain otomatis. Kandidat belum diadopsi;
native parity/uji dengar iPhone tetap diperlukan bila hasil host berguna.

## Riwayat audit ZIP dan pengiriman pembanding piano (selesai)

[Audit waveform dan codec](../reviews/coreai-drums-waveform-codec-audit/README.md)
memverifikasi seluruh delapan file ZIP. Hash rekaman MP3 yang dicatat source
reference cocok dengan laporan iPhone/Colab; PCM uji saat ini juga cocok persis
dengan manifest. Perbedaan PCM antarrun tidak membuktikan salah rekaman.

Di environment ini dilakukan enam decoding VAE FP32 dari latent yang sudah
tersimpan, tanpa encoder atau DiT. Replay kandidat cocok dengan audio Mac pada
batas asli (normalized RMSE `2,52e-6`); decode bertahap vs penuh juga lolos
(`5,87e-7`). Mengganti ukuran decode belum didukung sebagai perbaikan glitch.
Perbedaan hash source model Colab/publisher ditelusuri: metode neural aktif dan
APG sama, dimensi conditioning sama, DCW otomatis nonaktif pada Base. Belum ada
penyebab tunggal atau perbaikan musikal yang terbukti.

**Klarifikasi akses awal:** kedua cuplikan cloud belum didengar karena tautannya
tidak dapat diunduh. Jawaban afirmatif saat itu ditarik sebagai bukti. Perintah
lokal berikut kemudian berhasil dan kedua piano sudah dilaporkan bersih.
Disimpan untuk reproduksi, bukan permintaan mengulang uji.

Buat dua WAV langsung di Mac agar dapat dibuka dari Finder. Dari folder repo:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/listen_coreai_source_roundtrip.py \
  --out .runtime-cache/coreai-source-listening-v1 &&
open .runtime-cache/coreai-source-listening-v1
```

Script memakai source bundle 10,24 detik yang sudah ada dan checkpoint VAE
lokal yang diverifikasi. Ini **satu decode VAE FP32**, tanpa encoder atau
50 langkah DiT, unduhan bobot, native export, build app atau perubahan selector.
Output harus folder baru di luar ModelCandidates. Bila checkpoint/gate gagal,
berhenti dan simpan error/report; jangan melonggarkan batas.

Setelah **Ready**, putar `01-piano-source.wav`, lalu
`02-piano-reconstruction.wav` pada volume/output yang sama. Laporkan apakah
sumber bersih dan apakah glitch muncul setelah rekonstruksi. Keduanya memakai
satu gain attenuasi bersama, tanpa EQ/limiter. File asal tidak ditimpa dan
diperiksa hash sebelum/sesudah. Nilai korelasi/SNR rekonstruksi lossy bukan
sertifikat kualitas musikal atau expected exact parity dengan piano asli.

## Riwayat hasil tag dan pengumpulan ZIP (sudah selesai)

[Laporan uji Mac](../reviews/coreai-drums-instrumental-lyrics-result/README.md)
selesai 50/50. Empat replay kontrol (conditioning, mask, langkah pertama dan
decoder) PASS dengan seluruh error 0. Pengguna mendengar glitch pada kontrol
lirik kosong **dan** kandidat `[Instrumental]`. Perubahan tag saja belum
menyelesaikan masalah; kandidat tidak diadopsi. Full trajectory parity baru
tidak dibuktikan dan provenance kontrol tetap `bounded_diagnostic`.

PCM uji ini berhash `836c4509…6356018`, berbeda dari PCM import iPhone sebelumnya
`c79163af…d57278` dan snapshot host yang dicatat di repo `9517cc69…10bb945`.
Ini bisa berasal dari perbedaan decode/preparation; bukan bukti salah rekaman.
Namun waveform dari run berbeda tidak boleh dipakai sebagai exact comparison.

Audio dan metadata **dari uji Mac yang sudah selesai** kini sudah diterima dan
diverifikasi. Instruksi berikut disimpan untuk reproduksi, bukan permintaan
mengulang pengumpulan. `report.json` berisi
angka/hash, bukan waveform. Tidak perlu Generate, build app, ekspor model,
perubahan gain/prompt atau eksperimen durasi baru untuk langkah ini.

Dari folder repo di Mac, kemas delapan file yang sudah tersedia:

```bash
test ! -e .runtime-cache/drums-audio-review-v1.zip &&
zip -j .runtime-cache/drums-audio-review-v1.zip \
  .runtime-cache/drums-instrumental-lyrics-v1/{report.json,control-empty-lyrics-headroom.wav,candidate-instrumental-headroom.wav,candidate-latent.f32le,candidate-audio.f32le} \
  prototypes/ios27-coreai/ModelCandidates/source-context-1024/{reference.json,audio.f32le,source_latents-reference.f32le}
```

Unggah `.runtime-cache/drums-audio-review-v1.zip`. Command hanya membaca hasil
lama dan membuat arsip baru; tidak memuat model atau menyertakan checkpoint.
Nama output yang sudah ada ditolak agar arsip sebelumnya tetap tersimpan.
Jika folder uji diberi nama berbeda, sesuaikan hanya path folder hasil.
Source bundle harus tetap yang dipakai uji tersebut. Saat meninjau arsip,
periksa hash reference/source PCM dan seluruh waveform/latent terhadap laporan
sebelum menghubungkan hasilnya. Jika tidak cocok, berhenti untuk mencocokkan
provenance, bukan menganggap perbedaan itu glitch model.

Piano PCM diperlukan untuk menilai input yang benar; dua WAV memakai attenuasi
bersama; latent tersimpan memungkinkan pemeriksaan codec kemudian tanpa
mengulang 50 langkah DiT; pemeriksaan codec tersebut kini sudah dilakukan.
Belum ada penyebab tunggal atau perbaikan musikal
yang terbukti. Baseline BF16, groove yang diterima dan toleransi dipertahankan.

## Riwayat: satu uji conditioning di Mac

Pengguna mendengar glitch pada **kedua referensi host**: original FP32 dan
partition-host. Keluhan ini direproduksi di luar iPhone dan pada jalur tanpa
partisi BF16; belum ada dasar untuk menyalahkan BF16 atau runtime native saja.
Tidak perlu mengulang Generate iPhone atau menaikkan gain sebagai tindak lanjut.

[Audit Colab](../reviews/coreai-drums-colab-conditioning-audit/README.md)
menemukan satu perbedaan input yang nyata: Colab yang diterima memakai
`lyrics="[Instrumental]"`, preview memakai lirik kosong. Handler LEGO yang
dipin meneruskan isi itu ke encoder. Caption dan instruction drum sama.
Ini hipotesis yang dapat diuji, bukan penyebab glitch yang sudah terbukti.

Eksperimen yang sudah selesai ini hanya mengganti isi lirik pada **host original FP32**.
Kontrol memakai waveform FP32 tersimpan; satu kandidat dihitung 50 langkah.
Piano/context, noise, schedule, timbre silence, caption, guidance, checkpoint,
durasi 10,24 detik dan toleransi tetap. Baseline BF16 dan app tidak diubah.
Kandidat belum tentu mempertahankan groove; nilai groove serta glitch saat
mendengarkan, dan jangan mengganti baseline sebelum hasilnya diterima.

Command berikut disimpan untuk reproduksi, bukan permintaan mengulang uji.
Tutup Xcode. Dari checkout Mac yang sama:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/compare_coreai_drum_lyrics.py \
  --diagnostic-trajectory --execute \
  --out .runtime-cache/drums-instrumental-lyrics-v1 &&
open .runtime-cache/drums-instrumental-lyrics-v1
```

Ini satu inferensi kandidat di CPU Mac dan memerlukan waktu/RAM; bukan sekadar
mengekspor WAV lama. Checkpoint yang sudah tersedia diverifikasi, tanpa unduhan
bobot baru. Source/config publisher kecil yang dipin dipakai dari cache atau
diverifikasi saat diambil. Tidak ada native re-export, build app, aktivasi
selector, penghapusan cache atau pencarian seed/durasi otomatis. Output harus
baru dan berada di luar ModelCandidates. Tanpa `--execute` hanya plan dibuat;
gunakan folder baru untuk eksekusi sesudah membuat plan.

Sebelum hasil dibandingkan, script harus mereproduksi conditioning kontrol,
langkah FP32 pertama, dan decoder dari latent kontrol pada batas asli. Bila
gate atau checkpoint gagal, script berhenti dan mempertahankan `report.json`;
jangan melonggarkan batas. Empat check kontrol ini bukan full trajectory parity
baru; provenance diagnostik kontrol dan kegagalan lama tetap dicatat.

Dengarkan pada volume/output yang sama:

1. `control-empty-lyrics-headroom.wav`
2. `candidate-instrumental-headroom.wav`

Keduanya memakai satu gain attenuasi bersama, tanpa EQ/limiter. Bagikan
`report.json` dan hasil dengar: glitch berkurang atau tetap, serta groove tetap
cocok atau berubah. Kandidat dan latent/conditioning disimpan terpisah. Script
tidak memberi label musical PASS dan tidak mengekspor kandidat menjadi model
iPhone. Kalau kandidat tetap buruk, jangan mengulangnya atau langsung mengubah
durasi: perbedaan input gain, konteks dan referensi kit dicatat dalam audit
untuk memilih uji berikutnya berdasarkan bukti.

## Riwayat pembanding playback

**Hasil uji pengguna:** setelah menekan **Bandingkan drum dengan headroom**,
glitch **tetap sama**. Attenuasi ini belum menyelesaikan masalah; tidak perlu
mengulang uji headroom drum. Hasil headroom mix belum dilaporkan. Temuan ini
tidak membuktikan semua jalur playback bebas masalah atau membuktikan BF16
sebagai penyebab. Saat itu langkah berikutnya adalah dua referensi host di bawah,
yang kini juga sudah dilaporkan masih glitch.

Instruksi berikut dipertahankan sebagai catatan cara menjalankan pembanding:

Tidak perlu preparation, export model, preflight atau generasi ulang. Tutup
Xcode; dari checkout Mac yang sama:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build **CoreAIProbe Reuse**, Release, ke iPhone seperti sebelumnya. Pertahankan
profil drum aktif. Jangan hapus app; hasil drum tersimpan dipulihkan saat dibuka.
Bandingkan tombol **Dengarkan drum** dengan **Bandingkan drum dengan headroom**;
ulangi untuk mix, pada output audio dan volume perangkat yang sama.

Pembanding membuat WAV sementara dengan satu gain linear yang hanya
menurunkan level bila peak melebihi −3 dBFS. Tidak ada limiter, EQ atau boost.
File asli diperiksa SHA-256 sebelum/sesudah dan tidak diganti. Hasil ini bukan
perbaikan musikal, bukan WAV baseline baru dan tidak mengubah level mix
tersimpan. Bila distorsi berkurang, playback berperan; bila tetap, kualitas
waveform/model masih perlu ditelusuri. Dua hal tersebut juga dapat bersamaan.
Jangan menaikkan volume untuk menyamakan loudness selama pemeriksaan headroom.

## Riwayat ekspor dua referensi host tanpa menjalankan model

Langkah ini sudah dilakukan: kedua referensi dilaporkan masih glitch.
Instruksi ekspor berikut disimpan untuk reproduksi, bukan permintaan mengulang.
Tidak perlu build iPhone baru atau Generate lagi untuk mengekspor/mendengarkan
referensi yang sudah tersedia. Dari checkout Mac yang sama, gunakan command
di bawah; pilih nama `--out` baru bila output sebelumnya sudah ada.

Bundle drum di Mac sudah menyimpan dua waveform dari persiapan sebelumnya:
original FP32 dan partition-host BF16-storage/FP32-compute. Ekspor keduanya ke
WAV listening menggunakan bundle yang sama; tidak ada download, inference,
native re-export atau aktivasi selector:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/export_coreai_listening_references.py \
  --instrument drums --diagnostic-trajectory \
  --out .runtime-cache/drums-host-listening-v1 &&
open .runtime-cache/drums-host-listening-v1
```

Verifikasi integritas tiga bundle tetap lengkap, termasuk hash native; ini
dapat memerlukan waktu baca disk. `--diagnostic-trajectory` hanya mengakui
provenance diagnostik bundle yang sudah ada. Kegagalan parity lintasan bebas
tetap gagal, bukan PASS baru. Jika verifikasi gagal, berhenti dan simpan error;
jangan otomatis re-export atau mengubah batas. Folder output harus baru dan
di luar ModelCandidates; folder lama ditolak agar tidak ditimpa. Untuk mengulang
ekspor pilih nama output baru.

Dengarkan `drums-original-fp32-headroom.wav` dan
`drums-partition-host-headroom.wav` pada volume/output yang sama. Keduanya
memakai **gain attenuasi bersama**, sehingga perbedaan level asli tetap ada.
Salinan `*-raw.wav` mempertahankan Float32 termasuk nilai >1; gunakan salinan
headroom untuk pemeriksaan awal. `piano-prepared.wav` menunjukkan input host.
`listening-report.json` mencatat hash, gain dan status numerik yang tetap sama.

| Yang terdengar | Fokus langkah berikutnya |
| --- | --- |
| Original FP32 juga buruk | Kualitas preset/model atau konteks pendek; bandingkan dengan POC Colab yang diterima sebelum mengubah preset |
| Original baik, partition-host buruk | Conditioning/partisi dan perbedaan lintasan numerik |
| Keduanya baik, iPhone buruk | Jalankan satu baseline iPhone dengan input host yang tepat, lalu audit native/parity/playback |

Ini petunjuk investigasi, bukan kesimpulan otomatis. **Input host dan import
iPhone saat ini memiliki hash PCM berbeda**, walaupun MP3 asal sama. Jangan
membandingkan waveform tersebut sebagai exact parity. Bila kedua referensi
host baik, tombol **Uji baseline piano 10,24 detik** yang sudah ada memakai PCM
host dan referensi yang cocok; itu baru memerlukan satu generasi. Jangan jalankan
ulang lebih dahulu sebelum mendengar referensi yang sudah tersedia.

Baseline BF16, checkpoint, original piano, noise/schedule, 50 APG Euler,
guidance 7, toleransi abs `1e-4`, rel `1e-3`, normalized RMSE `1e-3`, serta gain
mix piano `0,78` / stem `0,55` dipertahankan. Belum ada pembuktian perbaikan
glitch atau penerimaan kualitas audio drum dari perubahan diagnostik ini;
groove sudah dinilai sesuai oleh pengguna.
