# Tahap 2: conditioning dari rekaman dan instrumen tujuan

Dua uji utama tahap 1 sudah [PASS di iPhone](../reviews/coreai-iphone17pro-retained-loop/README.md).
Keputusan prototipe adalah mempertahankan model/fungsi per asset selama sampling,
lalu melepas pool Swift sebelum komponen berikutnya. Tahap aktif berikutnya
adalah mengganti condition/context sintetis dengan conditioning yang berasal
dari rekaman pengguna dan instrumen yang diminta.

Sasaran pertama adalah jalur **piano → electric bass** pada klip pendek, memakai
[preset bass CUDA](../configs/electric-bass.json) sebagai acuan. Gitar sebagai sumber dan tiga stem
menyusul setelah jalur pertama bekerja. Target seluruh preprocessing dan
inference tetap lokal di iPhone/CoreAI.

Source publisher Base yang dipin (`modeling_acestep_v15_base.py`, manifest
`configs/assets-base.json`) memperlihatkan bahwa `prepare_condition` memakai:

1. Hidden state teks, lyric dan audio referensi melalui condition encoder;
   encoder mencakup `text_projector`, `lyric_encoder`, `timbre_encoder`, packing
   sequence dan attention mask.
2. Source latents, silence latents, chunk masks dan status cover. Cabang
   tokenizer/detokenizer atau hints yang sudah tersedia harus ditelusuri sesuai
   jalur `lego` yang dipakai baseline, sebelum memilih cabang yang akan dipindah.
3. Penyusunan context dengan concatenation source latents dan chunk masks,
   beserta condition/mask untuk denoiser. Latent hasil cover hints dipilih oleh
   `is_covers`; nilainya belum boleh ditebak untuk task pengiring.

Manifest aset mencantumkan **Qwen3-Embedding-0.6B** untuk jalur embedding teks.
Pemilihan prompt/format token, empty-lyric/reference handling, masking dan
posterior encoder audio harus sesuai jalur CUDA yang sudah diterima. VAE encoder
mean 1 detik yang lolos belum membuktikan kesetaraan seluruh preprocessing atau
posterior sampling pada jalur tersebut.

Pekerjaan implementasi tahap ini adalah menelusuri preprocessing `lego` yang
dipin, membuat fixture dari rekaman dan prompt nyata pada host, memindahkan
komponen conditioning yang diperlukan, lalu membandingkan hasil aktual di
iPhone. Nilai teks/audio siap pakai di fixture dapat menjadi referensi, tetapi
belum membuktikan conditioning mandiri dari rekaman baru di perangkat. Tidak
boleh mengganti komponen yang belum dipindah dengan tensor acak lalu menyebutnya
generasi dari rekaman.

Bentuk condition/mask dan panjang latent nyata mungkin berbeda dari kasus
sintetis 25 frame/64 condition token. Kontrak ekspor perlu disesuaikan dengan
jalur nyata sebelum integrasi sampling. Komponen dimuat secara berurutan sesuai
masa pakainya; menambahkan footprint dari laporan terpisah bukan pengukuran
peak gabungan pipeline. Keputusan menahan pool denoiser tidak otomatis berarti
menahan semua model conditioning dan codec bersamaan.

**Bagian audio kini diimplementasikan:** [audio-context piano](coreai-source-context.md)
memakai encoder posterior dengan sample eksplisit dan mask helper `lego` asli,
128 latent frame dari rekaman nyata. Ekspor native, host parity dan
[uji iPhone](../reviews/coreai-iphone17pro-source-context/README.md) berhasil:
3 run dan seluruh 12 output lolos. Bagian audio selesai untuk fixture ini;
tidak perlu mengulang uji yang sama sekarang. Embedding teks dan ConditionEncoder
belum diimplementasikan. Uji ini memvalidasi PCM yang dikanonisasi pada host,
belum raw-recording import/resampling langsung di iPhone. Setelah conditioning
benar, tahap 3 merangkainya dengan sampling/guidance dan decoder menjadi WAV
pendek yang bisa didengar. Tidak ada uji tambahan tahap 1 yang diminta.
