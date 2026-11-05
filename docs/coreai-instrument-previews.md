# Preview drum dan dua peran gitar di iPhone

Baseline bass 10,24 detik tetap disimpan dengan BF16, hitungan FP32, 50 langkah
APG Euler, guidance 7 dan toleransi asli. Workflow ini menambahkan preset drum,
gitar strumming dan gitar picking dari konfigurasi Colab yang sudah ada.
Tidak ada training atau pergantian checkpoint. Satu instrumen aktif per build.

| Preset CLI | Label app | Text / condition tokens | Hasil mentah |
| --- | --- | --- | --- |
| `drums` | drum | 111 / 123 | `drums.wav` |
| `guitar-strum` | gitar strumming | 106 / 118 | `guitar-strum.wav` |
| `guitar-pick` | gitar picking | 111 / 123 | `guitar-pick.wav` |

Jumlah token dipastikan memakai formatter publisher dan tokenizer yang dipin,
tanpa padding buatan atau pengubahan caption untuk memaksakan shape bass.
Lirik kosong tetap 11 token. App memiliki allowlist token, preset hash,
instruction dan nama profil tersendiri; baseline 5,12/10,24 detik tidak berubah.

## Mulai dari drum

Tutup Xcode. Dari checkout Mac yang sudah menyimpan tiga bundle bass 10,24 detik:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_instrument_preview.sh \
  --instrument drums --diagnostic-trajectory &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Tidak perlu memilih rekaman di Mac lagi: source-context yang sudah ada dipakai
untuk referensi host. Checkpoint/cache digunakan ulang; tidak dihapus. Persiapan
menghitung conditioning serta dua lintasan host 50 langkah dengan preset baru.
Tunggu **Ready**; ini lebih berat daripada update kode/build saja. Bila gagal,
simpan error dan jangan build seolah profil baru sudah diaktifkan. Bundle gagal
tidak menimpa bundle lama; selector hanya berubah setelah semua verifikasi lolos.

Pilih **CoreAIProbe Reuse**, Release, target iPhone. Cmd+R. Pastikan app menyebut
**Preset aktif: drum** dan 10,24 detik. Pilih piano yang sama, dengarkan input,
lalu tekan **Generate drum dari rekaman**. Bagikan laporan dan WAV piano/drum/mix.
Periksa groove, timing, instrumen bocor, fills dan headroom lewat uji dengar.
Kualitas preset baru belum diterima; lihat hasil perangkat terbaru berikut.

**Hasil drum terbaru:** 50/50 selesai dalam 85,824 detik, conditioning 6/6 PASS,
WAV tersedia, tetapi pengguna menolak kualitas: glitch di seluruh hasil dan
drum kurang terdengar. Ikuti [perbandingan kualitas/playback](coreai-drums-quality.md)
memakai WAV dan referensi host yang sudah ada. Tidak perlu preparation,
preflight atau generasi ulang untuk pemeriksaan awal ini. Run pertama sebelumnya
gagal bind `blocks-16-17` (0/50); [preflight](coreai-native-preflight.md) tetap
tersedia untuk kegagalan native. Penyebab keberhasilan run terbaru belum diketahui.

`--diagnostic-trajectory` adalah permintaan eksplisit eksperimen yang sama
mekanismenya dengan bass panjang: empat identical-input full-denoiser dan 50
identical-input APG checks harus lolos batas asli; kegagalan lintasan bebas tetap
NOT PASSED. Bukan pelonggaran toleransi atau fallback otomatis. Untuk strict
trajectory acceptance, jalankan tanpa flag. Jangan mengubah referensi/batas jika
gate gagal. Pada input Files baru, output parity tetap NOT EVALUATED (0/0),
sedangkan enam conditioning checks preset tetap diperiksa dengan referensinya.

## Gitar setelah evaluasi drum

Ganti `--instrument drums` menjadi `--instrument guitar-strum`, lalu kemudian
`--instrument guitar-pick`; flag diagnostic tetap eksplisit bila diperlukan.
Masing-masing membuat hasil tersendiri pada timeline piano yang sama. Kedua
peran ditujukan untuk nantinya digabung sebagai satu stem gitar, tetapi preview
ini **belum** menggabungkan layer atau menghasilkan arrangement tiga stem.

ConditionEncoder masih memakai publisher silence untuk timbre, seperti bass.
Referensi audio gitar beku dari workflow Colab belum dipindahkan; caption sama
tidak membuktikan kesamaan tone/style atau konsistensi full song. Uji dengar
peran dan porting referensi timbre tetap pekerjaan tersendiri. Jangan memakai
preset picked-v2 yang sebelumnya ditolak sebagai pengganti pick-layer-v3.

## Reuse yang diverifikasi

Source encoder dipakai bersama. Exporter memverifikasi bundle asal lengkap
sebelum menyalin native payload. Hanya stage dengan shape, binding, precision
dan checkpoint yang cocok dipakai ulang; referensi setiap preset dihitung baru.

| Target ← sumber yang tersedia | Conditioning reuse / export | DiT+APG+decoder reuse / export |
| --- | ---: | ---: |
| drum ← bass panjang | 10 / 17 | 3 / 13 |
| strumming ← bass panjang | 27 / 0 | 16 / 0 |
| picking ← drum | 27 / 0 | 16 / 0 |

Jika sumber dengan geometri sama belum tersedia, driver memilih bundle lain
yang sudah ada; hanya stage cocok yang disalin. Metadata `native_reuse`
mencatat manifest asal dan daftar aset yang dipakai ulang. Ini perencanaan
kompatibilitas, bukan bukti runtime native. Tidak ada rename aset berbeda shape.
Qwen tetap 28 layer, ACE ConditionEncoder 8 lyric+4 timbre, DiT 24 layer.

`active-preview.json` memilih source + conditioning + sampler satu profil.
Build tidak menyalin seluruh bank instrumen. Semua bundle di Mac, original
piano, bass/WAV lama dan cache tetap disimpan. Hasil baru memakai direktori UUID
dan nama stem berbeda; app memulihkan hasil instrumen aktif saja. Mix piano
tetap 0,78 dan stem 0,55, dengan gain bersama bila peak >0,98; WAV mentah tidak
dinormalisasi. Pembanding playback headroom opsional hanya membuat salinan
sementara; tidak mengubah WAV/gain mix tersimpan. Glitch bass ringan sebelumnya
diabaikan atas permintaan pengguna; glitch drum terbaru belum diterima.

Untuk mengaktifkan ulang bundle instrumen yang sudah lengkap tanpa export:

```bash
bash scripts/prepare_coreai_instrument_preview.sh \
  --instrument drums --activate-only --diagnostic-trajectory
```

Untuk kembali ke baseline bass yang sudah ada:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_preview_window.py --seconds 10.24 \
  --activate-only --diagnostic-trajectory
```

Rebuild setelah pergantian selector. AOT/default compute tidak diaktifkan;
source cpuOnly diminta, placement aktual belum diukur. First-step scope tunggal,
retention maksimal 15 handles mulai step 2, targeted bind-repair satu budget,
progres bounded dan detached CPU copies tetap. Guard 2000 MiB bukan exact peak.

## Bukti dan batas validasi

[Audit kontrak/tokenisasi](../reviews/coreai-instrument-preview-contracts/README.md)
memuat tokenisasi aktual dan rencana reuse. 135 tes host lulus, dua tes optional
export tools dilewati; termasuk delapan tes baru isolasi/gate/reuse/selector.
Torch CPU 2.8.0, Transformers 4.57.3 dan Diffusers 0.37.0 dipakai untuk regresi.
Pemeriksaan sintaks Python/shell dan whitespace lulus. Swift SDK Apple tidak
tersedia di Linux. Native export model penuh, build Apple, parity perangkat,
latency dan kualitas musikal preset baru belum diverifikasi. Full-song chunking,
overlap, resume dan ekspor tiga stem sepanjang lagu masih tahap berikutnya.
