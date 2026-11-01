# Uji satu forward denoiser lengkap di iPhone

Profil ini menjalankan **seluruh 24 layer denoiser**, termasuk input projection,
dua timestep embedding, conditioning projection, normalisasi dan output projection.
Ini langkah berikutnya setelah dua blok lolos parity: satu prediksi velocity
lengkap pada latent sintetis 25 frame. Belum ada loop diffusion, CFG, rekaman
piano, conditioning teks/instrumen atau waveform pengiring.

## Jalankan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_denoiser.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Skrip memakai kembali environment dan checkpoint Base dari cache. Jika cache
tidak ada, checkpoint publisher berukuran 4,79 GB. Seluruh SHA-256 diperiksa,
lalu hanya tensor `decoder.*` dimuat secara strict: **1.575.458.880 parameter**.
Config dan source publisher tetap dipin dan diperiksa hash-nya.

Paket lokal baru sekitar **3,15 GB** berada di
`prototypes/ios27-coreai/ModelCandidates/denoiser-pretrained/`.
Semua baseline lama tetap tersimpan; jika semuanya tersedia, Xcode menyalin
sekitar **5,18 GB** resource model pengujian. Persiapan dan penyalinan lebih
besar daripada probe dua blok. Paket lengkap yang sudah ada diverifikasi saat
skrip diulang; tidak ditimpa. Bobot, cache dan native asset besar diabaikan Git.

Bobot disimpan FP16 untuk ukuran file, dengan perhitungan dan input/output
FP32. Buffer analitik RoPE tetap FP32; proyeksi linear, Conv1d dan ConvTranspose1d
menggunakan bobot yang diubah ke FP32 saat dipakai. Ini bukan optimasi RAM yang
terbukti: pada [uji dua blok sebelumnya](../reviews/coreai-iphone17pro-pretrained-dit-storage16/README.md),
parity lolos tetapi footprint tersampel lebih tinggi daripada profil FP32.

## Jalankan di iPhone

Pilih iPhone 17 Pro → **Cmd+R** → **Uji denoiser lengkap**. Jalankan profil ini
terlebih dahulu setelah membuka ulang aplikasi. Kirim laporan lengkap, termasuk
dua status parity, waktu load/run dan footprint tersampel. Build Swift terbaru
belum dikompilasi dengan Apple SDK di workspace Linux ini.

Probe menyimpan tahap terakhir secara best effort ke Documents aplikasi sebelum
validasi, load dan setiap inference. Jika aplikasi terhenti tanpa laporan akhir,
buka kembali aplikasi untuk melihat laporan/tahap terakhir yang tersimpan.
Tahap tersebut hanya menunjukkan tempat proses berhenti; penyebab terminasi
tetap memerlukan log perangkat dan tidak otomatis berarti kehabisan RAM.

## Kontrak numerik

| Input Float32 | Shape | Isi uji |
| --- | --- | --- |
| noise | [1, 25, 64] | Noise latent seed 0 |
| context | [1, 25, 128] | Context sintetis |
| condition | [1, 64, 2048] | Embedding conditioning sintetis |
| timestep | [1] | 0,5 |
| timestep_r | [1] | 0,25 |

Output `velocity`: Float32 `[1, 25, 64]`. Layout denoiser `[batch, frame, channel]`
berbeda dari layout VAE `[batch, channel, frame]`. 25 frame setara jendela latent
1 detik pada 25 Hz, tetapi keluaran velocity ini bukan audio yang bisa didengar.
Source asli melakukan padding 25→26 frame sebelum patching dan crop kembali
ke 25 frame. Mask eager, RoPE dan timestep dihitung di dalam model. Cache KV,
output attention dan early exit dimatikan; semua 24 layer benar-benar dilewati.

Tiga run masing-masing wajib lolos dua pembandingan:

1. CoreAI terhadap referensi PyTorch untuk bobot tersimpan FP16/hitungan FP32.
2. CoreAI terhadap referensi dari seluruh bobot FP32 asli pada input yang sama.

Ukuran dan SHA-256 lima input serta kedua referensi diperiksa aplikasi. Batas
tetap `abs(error) <= 1e-4 + 1e-3 * abs(reference)`, normalized RMSE <= `1e-3`.
Exporter berhenti jika perubahan penyimpanan bobot gagal batas baseline itu.

[Ekspor Linux lengkap berhasil](../reviews/coreai-pretrained-full-denoiser-export/README.md).
Pemuatan, dua gerbang parity, latensi dan memori denoiser lengkap di iPhone belum
terverifikasi. Bobot dense FP32 denoiser saja berukuran 6,30 GB; angka itu bukan
peak RAM terukur. Gerbang ini menguji kelayakan seluruh denoiser, tanpa menganggap
timing atau footprint dua blok dapat diekstrapolasi ke seluruh pipeline.
