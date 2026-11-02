# Ekspor host: preview bass 10,24 detik

Ekspor source native sudah dieksekusi di Linux memakai checkpoint asli yang
terverifikasi, audio piano 10,24 detik dan prompt bass dengan durasi yang sesuai.
Ini bukan bukti build Apple atau inference CoreAI di iPhone. Target berikutnya
adalah run perangkat melalui [panduan](../../docs/coreai-bass-preview-1024.md).

Profil memakai satu jendela 491520 frame stereo 48 kHz, 256 latent frame,
106 token teks / 11 lyric / satu timbre, conditioning 118 token dan hidden
128 token untuk dua cabang guidance. Seluruh 24 layer denoiser, 50 langkah APG
serta decoder penuh dipertahankan; tidak ada penyambungan klip independen.

| Pemeriksaan host | Hasil |
| --- | --- |
| Distribution sampling / weight-norm folding sumber | Exact |
| Conditioning partition dan BF16 storage vs FP32 penuh | Ketiga output exact |
| APG pada input identik vs publisher FP64 | 50/50 strict PASS |
| Normalized RMSE latent akhir vs referensi penuh | 0,000446334 |
| Normalized RMSE waveform vs referensi penuh | 0,000373360 |
| Gate pointwise lintasan / waveform | NOT PASSED |
| Status ekspor | `bounded_diagnostic` |

Batas abs 1e-4, rel 1e-3 dan normalized RMSE 1e-3 tidak diubah. Semua 50 latent
comparison dan waveform finite serta berada di bawah batas normalized RMSE,
namun sebagian sample melanggar batas pointwise. Tidak ada klaim overall
numerical PASS. Referensi tetap memakai conditioning FP32 penuh, denoiser
FP32 asli dan APG publisher FP64; input referensi tidak diganti menjadi output
kandidat untuk meloloskan gate.

Percobaan FP16 storage sebelumnya ditolak (normalized RMSE latent akhir sekitar
0,0013275); proyeksi FP64 saja juga ditolak (sekitar 0,0011704). Angka tersebut
tersimpan di `rejected-fp16-experiments.json`. Pemeriksaan dua matrix checkpoint
menemukan 551 dan 1350 nilai yang berubah pada cast FP16, dengan perubahan
maksimum 2,98e-8; BF16 mempertahankan nilai aslinya secara exact. BF16 memiliki
rentang eksponen lebih luas. Profil panjang menggunakan BF16 storage, dengan
cast FP32 sebelum projection dan FP32 APG semula. Bentuk input/output tetap
Float32. BF16 berlaku pada conditioning/denoiser; bobot VAE tetap FP32.
Ini tidak membuktikan semua perbedaan numerik hilang atau dukungan
runtime iPhone sudah teruji.

Sebanyak **44 aset native** diekspor: satu encoder sumber, 27 conditioning,
14 shard denoiser, satu APG/Euler dan satu decoder. Ketiga bundle diverifikasi
ulang dan selector lokal diaktifkan melalui driver persiapan. Dry-run staging
memilih `bass-source-1024`, **163 file / 6,261 GB**, mengecualikan 11,441 GB aset
lokal lain. Tidak ada salinan 6 GB yang dibuat oleh dry-run. Ukuran ini adalah
resource build di workspace, bukan total data/cache/RAM di iPhone.

Sebanyak **85 uji regresi host lulus**, termasuk penolakan kontrak campuran,
selector gagal yang mempertahankan output lama, staging satu profil dan
pelestarian bobot kecil BF16 dengan hitungan FP32. Project/scheme juga diperiksa
secara struktural. Workspace tidak memiliki Swift compiler/Apple SDK; build,
specialization BF16 pada CPU iPhone, memori/waktu native dan uji dengar masih
belum diverifikasi. Profil pendek yang sudah diterima tetap disimpan.

`reference.json`, `source-reference.json` dan `conditioning-reference.json`
adalah salinan manifest hasil ekspor aktual, berisi hash aset/biner lokal.
`summary.json` mencatat hasil, lingkup dan batas bukti. Audio, bobot serta
bundle model tetap lokal dan diabaikan Git.
