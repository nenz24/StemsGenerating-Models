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

## Hasil terbaru: kedua piano bersih; satu kandidat APG di Mac

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

Dari folder repo Mac yang sama:

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
