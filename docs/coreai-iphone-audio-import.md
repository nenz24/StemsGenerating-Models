# Impor rekaman piano dari Files di iPhone

Tahap ini menambahkan input baru ke pipeline bass pendek yang sudah berjalan.
Impor, decode/resampling, encoder sumber, conditioning, 50 langkah APG dan decoder
berjalan di iPhone. [Laporan perangkat pertama](../reviews/coreai-iphone17pro-audio-import/README.md)
mencatat generasi dari MP3 stereo 44,1 kHz selesai: 50/50 langkah, 65,88 detik,
footprint maksimum proses tersampel 472,71 MiB. Waktu dan sampler generasi ini
tidak mencakup tahap penyalinan/decode/resampling/hash yang sudah selesai
sebelumnya; kualitas audio input baru masih menunggu pemeriksaan WAV/uji dengar.

## Jalankan

Tutup Xcode, lalu dari checkout di Mac:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih scheme **CoreAIProbe Reuse**, target iPhone, lalu Cmd+R. Model yang sudah
ada dipakai kembali; tidak perlu menjalankan skrip persiapan/export model lagi.
Profile resource `auto` tetap memilih source CPU dan mengecualikan aset AOT serta
probe lama yang tidak dipakai. Perubahan ini tidak menambahkan bobot model.

1. Tekan **Pilih rekaman piano**, lalu pilih audio di Files. Untuk uji awal,
   pakai `piano.mp3` yang sudah dipakai pada baseline.
2. Tekan **Dengarkan bagian piano** untuk memeriksa bagian awal yang terpilih.
3. Tekan **Generate bass dari rekaman** dan tunggu sampai 50/50 langkah serta
   `WAV generated on device: YES`.
4. Dengarkan bass dan piano+bass. Bagikan laporan generasi, kedua WAV, piano
   5,12 detik dan info rekaman untuk pemeriksaan.

Audio harus mono atau stereo, minimal 5,12 detik dan dapat dibaca AVAudioFile
(misalnya WAV, MP3, M4A atau AIFF yang didukung sistem). App selalu mengambil
5,12 detik awal. Audio mono disalin ke kedua kanal; sample rate dikonversi ke
48 kHz, Float32 planar. Tidak ada normalisasi input. Audio pendek, lebih dari
dua kanal, kegagalan decode atau nilai nonfinite ditolak sebelum inference.
Jika memakai provider cloud, Files mungkin perlu mengunduh file sebelum impor;
inference tidak memakai server.

## Input dan hasil

App menyalin file asli melalui security-scoped URL dan NSFileCoordinator ke
`Documents/piano-import-<UUID>/original.<extension>`. File provider tidak diubah.
Decoder hanya menampung jendela pendek dengan buffer 4096 frame; file asli
penuh disalin dan di-hash secara streaming. Setiap impor berhasil disimpan
terpisah dan memakai ruang disk sesuai ukuran file asli. Impor gagal hanya
menghapus folder baru miliknya sendiri.

`piano-preview.wav` dan `import.json` mencatat PCM yang benar-benar diteruskan
ke encoder. Hasil generasi tersimpan dalam `Documents/bass-preview-<UUID>/`:

- `bass.wav`: output decoder asli.
- `piano-with-bass.wav`: PCM input yang sama, dicampur dengan bass pada gain
  piano 0,78 dan bass 0,55; gain bersama diturunkan bila peak melebihi 0,98.
- `piano.wav`: tepat 245760 frame stereo yang dipakai sebagai input.
- `input.json`: identitas impor, SHA-256 asli/PCM/WAV, format asli dan konversi.
- `report.txt`: mode input, kemajuan, hitungan call, waktu dan footprint tersampel.

Hasil terakhir tetap dapat dipulihkan saat membuka ulang app. Pilihan input
untuk generasi baru perlu dipilih lagi setelah app dibuka ulang. Tidak ada
mikrofon, pemilihan offset, chunking lagu panjang atau prompt bebas pada tahap ini.

## Pembandingan yang berlaku

Rekaman baru tidak punya trajectory/waveform referensi host yang cocok. App
melewati empat output-reference encoder sumber dan sepuluh pembandingan output
sampling/decoder untuk mode impor, tanpa memuatnya sebagai input model.
Laporan menyatakan **numerical parity NOT EVALUATED**, dengan output checks
`0/0`; ini bukan kegagalan generasi maupun numerical PASS.

Shape, dtype, finite values, identitas model/kontrak, hash PCM, epsilon, mask,
noise dan schedule tetap diperiksa. Conditioning teks/timbre masih memakai
preset bass yang sama sehingga enam pembandingannya tetap berlaku. Epsilon
fixture dan initial noise fixture dipakai kembali; RNG dan tokenisasi prompt
bebas di iPhone belum diimplementasikan.

**Uji baseline piano 5,12 detik** tetap memakai PCM host lama dan seluruh gate
referensinya, termasuk toleransi yang tidak diubah. AVAudioConverter dapat
menghasilkan PCM berbeda dari decoder/resampler Mac meskipun file asli sama;
mode impor tidak boleh mengklaim hash output atau numerical parity baseline.
Kecocokan nada/groove pada input baru harus didengar.

## Validasi

Sebanyak 78 uji regresi host lulus pada 2 November 2026, mencakup kontrak model,
APG, referensi, staging resource dan helper update proyek. Struktur proyek, empat
source Swift dan scheme Release/reuse juga berhasil diperiksa. Workspace
Linux tidak memiliki Swift compiler/Apple SDK. Laporan pengguna kemudian
mengonfirmasi build Release dan penyelesaian impor MP3 44,1 kHz stereo hingga
WAV pada iPhone. Hash input konsisten dengan metadata; WAV aktual dan playback
run ini belum diperiksa. Peak bass yang dilaporkan 1,037155 memerlukan pemeriksaan
headroom pada WAV Float32, tanpa langsung menyimpulkan hard clipping.
Selanjutnya periksa WAV dan uji dengar, rekaman berbeda, mono, stereo 48 kHz,
penolakan audio pendek dan pembatalan picker. Tidak diperlukan perubahan
checkpoint untuk pemeriksaan tersebut.
