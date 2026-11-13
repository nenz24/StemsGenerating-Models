# Conditioning bass, drum dan gitar untuk window30s yang sama

Helper ini menyiapkan tiga cache host dengan bentuk `[1,127,2048]`, untuk
memakai ulang sampler750-frame yang ada. Tidak ada perubahan bobot BF16 DiT,
FP32 compute/VAE, abs1e-4/rel1e-3/nRMSE1e-3, atau baseline audio. Cache ini
belum menjalankan lagu penuh atau mengaktifkan profil iPhone.

## Jalankan pada Mac dengan checkpoint yang sudah ada

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_full_song_conditioning.py --execute &&
open -R .runtime-cache/full-song-conditioning-v1/review.zip
```

Capture acuan tetap `.runtime-cache/drums-reference30-v3`. Checkpoint base,
Qwen dan silence harus sudah ada di vendor/cache lokal dengan checksum yang
dipatok. Source publisher/formatter dan config/tokenizer juga harus sudah
tersimpan dari persiapan sebelumnya. Berkas yang hilang atau berubah menyebabkan
helper berhenti; tidak ada download/repair. Lokasi alternatif menerima
`--capture`, `--publisher-dir`, `--qwen-cache`, `--tokenizer`,
`--base-checkpoint`, `--qwen-checkpoint` dan `--silence-latent`.
Output yang sudah ada dipertahankan; gunakan `--out` baru untuk percobaan lain.

Tanpa `--execute`, mode default hanya mencetak plan memakai stdlib.
`--inspect-tokens` memverifikasi formatter/tokenizer lokal serta panjang token,
tanpa memuat model atau menulis output. Ini bukan hasil neural conditioning.

## Preset dan penghitungan

Drum memakai prompt acuan30s persis, termasuk `[Instrumental]`, metadata,
whitespace dan end markers. Bass mengambil caption `electric-bass.json`;
gitar mengambil `clean-electric-guitar-strum-layer.json`. Keduanya menjadi
preset30s baru dengan kalimat tambahan `No percussion or vocals.`. Identitas
preset baru dan hash preset sumber dicatat terpisah; preset lama tidak diubah.

Tokenizer yang dipatok harus menghasilkan111 token teks dan15 token lirik
secara alami untuk setiap instrumen. Satu token timbre dari publisher silence
memberi127 token packed, semuanya valid. Tidak ada padding, pemotongan atau
kit reference. Ukuran yang sama memungkinkan pemakaian bentuk input native
yang ada; kualitas bass/gitar masih perlu didengarkan setelah generation.

Qwen dimuat sekali untuk tiga text forward, dengan lookup tabel embedding
lirik yang sama. Qwen dilepas sebelum satu ConditionEncoder dimuat untuk tiga
forward. Semua perhitungan host FP32 CPU. Tidak ada DiT, source encoder,
decoder, export native, penghapusan cache atau penggantian selector model.

Tujuh keluaran drum dibandingkan terhadap capture asli: text/lyric IDs,
text/lyric embeddings, publisher silence, packed condition dan mask. ID dan
mask juga wajib persis sama. Seluruh perbandingan harus lolos toleransi asli
sebelum `ready_for_staging` menjadi true. Hasil yang gagal tetap disimpan
bersama report gagal; tidak dipakai mengganti acuan. Gate ini hanya conditioning;
status sampler penuh sebelumnya tetap4/6 dan belum lolos.

`review.zip` berisi report serta21 array conditioning, tanpa bobot model,
audio pengguna atau trajectory sampler. Simpan sebagai artefak privat dan
unggah untuk audit lanjutan; jangan commit ZIP, array atau report lokal.
`ready_for_native_execution` tetap false. Integrasi context seluruh lagu,
noise global, overlap-add, uji dengar sambungan serta pengukuran resource dan
thermal seluruh lagu masih harus dilakukan.

Lihat [persiapan lagu penuh](coreai-full-song-preparation.md),
[source encoder berjendela](coreai-source-window-encoder.md),
[panduan performa](coreai-bass-preview-performance.md) dan
[precision A/B/A](../reviews/coreai-iphone17pro-precision-aba/README.md).
Jangan mengekstrapolasi waktu komponen ke waktu lagu penuh atau memperbesar
seluruh sampler menjadi FP32 berdasarkan hasil conditioning.

Tahap berikutnya tersedia pada [preview tiga instrumen30s](coreai-instrument-preview30.md),
memakai cache ini dan sampler existing yang sama di iPhone untuk uji dengar
bass/gitar, sebelum integrasi lagu penuh.
