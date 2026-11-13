# Review glitch perpindahan akor pada bass/gitar30s

Laporan runtime saja tidak menyimpan waveform atau latent. Run yang selesai
dan resource gate yang lolos tidak membuktikan audio bebas glitch. Untuk
membedakan hasil latent generasi dari proses decode, ambil output existing,
tanpa menjalankan Generate ulang atau mengubah sampler/prompt/baseline.

## Ekspor hasil yang sudah tersimpan

Pada perangkat iPhone yang sama, download **container CoreAI Probe terbaru**
melalui pengelola perangkat Xcode: **CoreAI Probe → App Container → Download…**.
Simpan folder hasil download. Container lama sebelum preview bass/gitar tidak
memiliki run ini. Helper memeriksa UUID dan akan berhenti bila salah folder.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
python3 scripts/package_coreai_instrument_transition_review.py \
  --choose-container &&
open -R .runtime-cache/bass-guitar-transition-review-v1.zip
```

Pilih folder container yang baru di-download ketika dialog muncul. `--source`
juga menerima `.xcappdata`, folder container, `Documents`, `InstrumentReference30`
atau folder induk unduhan. Pencarian read-only dibatasi8 tingkat/4096 direktori,
tidak mengikuti symlink dan melewati Library/cache. Folder unduhan yang memiliki
wrapper tambahan atau beberapa container dapat diperiksa langsung:

```bash
python3 scripts/package_coreai_instrument_transition_review.py \
  --source "$HOME/Downloads" --list-runs
```

Daftar menampilkan letak `InstrumentReference30`, UUID bass/gitar yang tersedia
dan kelengkapan nama file, tanpa membuat ZIP atau menilai kualitas/numerik.
Jika run tidak ditemukan saat packaging, daftar yang sama disertakan dalam
pesan error; ini bukan bukti bahwa data sudah dihapus dari iPhone.
Untuk packaging dari folder Downloads, ganti `--choose-container` dengan
`--source "$HOME/Downloads"` dan sertakan UUID yang diminta.

Default mengambil pointer terakhir bass dan gitar secara terpisah; untuk
mengunci run tertentu, isi `--bass-run` serta `--guitar-run` dengan UUID dari
laporan masing-masing. Jangan mengetik path contoh atau membuat generation
baru hanya untuk mendapatkan file yang sebenarnya sudah tersimpan.

Salinan unduhan dengan UUID sama hanya dipilih otomatis jika seluruh berkas
yang diperlukan identik secara byte. Salinan yang berbeda/tidak lengkap atau
pointer terbaru dengan UUID berbeda menyebabkan helper berhenti dan mencetak
lokasinya; tidak ada pemilihan run lain sebagai pengganti.

Helper hanya memakai stdlib Python. Tidak memuat model, mengekspor bobot,
menghapus cache atau mengubah file di container. Output existing dipertahankan;
pilih `--out` baru jika perlu. ZIP harus berada di luar container/models/
baselines dan berukuran maksimal32MiB. Tidak perlu build ulang app untuk ini.

ZIP berisi kedua report asli,16 array existing (lima latent checkpoint,
final latent/momentum, raw stereo audio per instrumen), dua cuplikan WAV
22–30 detik dan satu manifest. Berkas audio/latent ini bersifat privat; unggah
untuk review dan jangan commit ZIP, isinya atau report perangkat.

## Pengikatan audio terhadap laporan

Final latent dan checkpoint50 harus cocok dengan SHA256 pada laporan. WAV
solo existing juga harus cocok dengan SHA256 laporan. Raw planar Float32
dikalikan gain FP32 asli kemudian ditulis ulang sesuai format WAV perangkat;
SHA256 hasilnya wajib persis sama. Ini memverifikasi hubungan raw audio dengan
audio yang didengarkan, tanpa inverse-gain approximation atau toleransi baru.

Cuplikan akhir hanyalah slice raw audio pada gain dengar asli, tanpa filter,
crossfade, limiter atau normalisasi tambahan. Hash checkpoint sparse lain
merupakan hash byte saat ekspor, bukan provenance independen dari laporan.
Keluhan glitch tidak mengubah numerical parity bass/gitar yang belum
dievaluasi atau gate drum sebelumnya yang belum lolos4/6.

## Pemeriksaan lanjutan pada audio yang sama

Decode final latent native pada CPU FP32 dengan decoder VAE asli, dalam bentuk
full750 dan enam window256 yang sama. Bandingkan raw waveform pada batas asli
abs1e-4/rel1e-3/nRMSE1e-3; simpan kegagalan dan scope tiap perbandingan.
Periksa lima sambungan5.12/10.24/15.36/20.48/25.6 detik serta bagian akhir.
Sambungan fixed decoder tidak identik dengan pergantian akor musik.

Jika artefak tetap terdengar pada kedua decode dari latent yang sama, bukti
lebih mendukung pemeriksaan hasil generasi/conditioning berikutnya. Jika
berbeda, isolasi decode/backend pada input yang sama. Belum ada klaim penyebab
atau perbaikan sebelum raw output dan uji dengar tersebut tersedia. BF16 DiT,
50 langkah APG, source gain sekali, preset dan toleransi tetap dipertahankan.
