# Uji encoder pretrained pada iPhone

Decoder asli sudah lulus parity untuk
[0,2 detik](../reviews/coreai-iphone17pro-pretrained-decoder-02s/README.md) dan
[1 detik](../reviews/coreai-iphone17pro-pretrained-decoder-1s/README.md).
Tahap ini memeriksa arah sebaliknya: **audio → latent** memakai encoder asli
dari checkpoint VAE yang sama. Ini belum menjalankan diffusion generator.

## Jalankan di Mac

Dari folder repository:

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_decoder.sh --component encoder --frames 25 &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip persiapan yang sama kini mendukung `--component encoder`, sehingga memakai
environment Python dan cache checkpoint yang sudah dipakai decoder. VAE yang
tersedia diverifikasi dan digunakan ulang; jika cache tidak ada, hanya VAE
337 MB yang diunduh. Tidak memerlukan GPU CUDA atau training baru.

Hasil disimpan terpisah di
`prototypes/ios27-coreai/ModelCandidates/encoder-pretrained-1s/`.
Kedua paket decoder tetap tersimpan. Eksekusi ulang memverifikasi paket yang
sudah lengkap; output yang rusak atau sudah ada tidak ditimpa. Paket besar,
cache dan staging diabaikan Git. Repo menyimpan kode dan manifest validasi.
Menambahkan profil encoder menambah sekitar 338 MB resource lokal; jika ketiga
profil disalin ke aplikasi, total resource model sekitar 1 GB. Ini ukuran paket
pengujian, bukan ukuran produk akhir atau penggunaan RAM saat inference.

## Jalankan di iPhone

Pilih iPhone 17 Pro di Xcode → **Cmd+R** → **Uji encoder 1 detik**.
Sebaiknya jalankan encoder terlebih dahulu setelah membuka ulang aplikasi agar
baseline footprint tidak tercampur dengan pengujian model sebelumnya. Cache
CoreAI tetap dapat memengaruhi waktu load. Signing Team tetap memakai pengaturan
lokal yang sudah ada. Kirim laporan lengkap dari tombol **Bagikan laporan**.

Paket harus disiapkan sebelum build aplikasi. `git pull` saja tidak memasukkan
bobot besar ke resource Xcode. Source Swift profil encoder belum dikompilasi
dengan Apple SDK di workspace Linux ini.

## Kontrak numerik

- Input Float32 `[1, 2, 48000]`: 1 detik audio stereo, 48 kHz, multitone sintetis
  tetap dengan envelope halus; belum menggunakan rekaman piano Anda.
- Output Float32 `[1, 64, 25]`: posterior mean/mode encoder, tanpa sampling acak.
- Referensi PyTorch CPU FP32; ukuran dan SHA-256 kedua binary diperiksa aplikasi.
- Toleransi tetap: absolute `1e-4`, relative `1e-3`, normalized RMSE `1e-3`.
- Tiga run, waktu load dan run+read-back, serta footprint seluruh proses
  tersampel setiap 10 ms. CPU-only diminta; hardware aktual tidak diukur.

[Ekspor encoder berhasil di Linux](../reviews/coreai-pretrained-encoder-export-1s/README.md),
dengan checkpoint penuh terverifikasi dan hasil identik sebelum/sesudah folding
weight normalization. Hasil iPhone masih diperlukan untuk memverifikasi parity,
latensi dan memori encoder. Kelulusan tes ini kelak belum membuktikan kualitas
rekonstruksi, sinkronisasi atau generasi tiga instrumen; tahap generasi masih
memerlukan conditioning dan model generatif.
