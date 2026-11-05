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

## Hasil terbaru dan satu uji conditioning di Mac

Pengguna mendengar glitch pada **kedua referensi host**: original FP32 dan
partition-host. Keluhan ini direproduksi di luar iPhone dan pada jalur tanpa
partisi BF16; belum ada dasar untuk menyalahkan BF16 atau runtime native saja.
Tidak perlu mengulang Generate iPhone atau menaikkan gain sebagai tindak lanjut.

[Audit Colab](../reviews/coreai-drums-colab-conditioning-audit/README.md)
menemukan satu perbedaan input yang nyata: Colab yang diterima memakai
`lyrics="[Instrumental]"`, preview memakai lirik kosong. Handler LEGO yang
dipin meneruskan isi itu ke encoder. Caption dan instruction drum sama.
Ini hipotesis yang dapat diuji, bukan penyebab glitch yang sudah terbukti.

Eksperimen berikut hanya mengganti isi lirik pada **host original FP32**.
Kontrol memakai waveform FP32 tersimpan; satu kandidat dihitung 50 langkah.
Piano/context, noise, schedule, timbre silence, caption, guidance, checkpoint,
durasi 10,24 detik dan toleransi tetap. Baseline BF16 dan app tidak diubah.
Kandidat belum tentu mempertahankan groove; nilai groove serta glitch saat
mendengarkan, dan jangan mengganti baseline sebelum hasilnya diterima.

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
sebagai penyebab. Langkah aktif berikutnya adalah dua referensi host di bawah,
dengan preset dan batas numerik tetap.

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
