# Uji penyimpanan bobot FP16 dengan hitungan FP32

[Dua blok DiT FP32 dengan konteks 160/64 token sudah PASS di iPhone](../reviews/coreai-iphone17pro-pretrained-dit-context/README.md),
rata-rata dua run hangat 31,10 ms. Tahap ini memakai bobot yang disimpan sebagai
FP16, lalu diubah ke FP32 saat digunakan dalam proyeksi linear. Aktivasi, mask,
input/output dan perhitungan tetap FP32. Ini menguji pengurangan ukuran asset
sambil menjaga hasil baseline; bukan klaim pengurangan RAM atau waktu inference.

[Hasil iPhone sudah PASS untuk kedua referensi](../reviews/coreai-iphone17pro-pretrained-dit-storage16/README.md):
rata-rata dua run hangat 32,46 ms, footprint maksimum tersampel 638,2 MiB.
Penghematan RAM belum terbukti; laporan profil FP32 sebelumnya mencatat 174,1 MiB.
Langkah berikutnya adalah [satu forward denoiser lengkap](coreai-full-denoiser.md).
Panduan di bawah tetap tersedia untuk mengulang profil dua blok.

## Jalankan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_dit_block.sh --profile storage16 &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Checkpoint Base dari cache dipakai ulang dan seluruh SHA-256 publisher diperiksa.
Jika cache tidak tersedia, unduhannya tetap 4,79 GB. Tidak memerlukan training
atau GPU CUDA. Native asset baru sekitar **252 MB**, dibanding sekitar **504 MB**
untuk dua blok FP32. Paket lengkap dengan delapan input dan dua referensi sekitar
**257 MB**. Semua paket lama tetap disimpan; direktori baru adalah
`prototypes/ios27-coreai/ModelCandidates/dit-storage16-pretrained/`.
Paket yang sudah ada diverifikasi saat skrip diulang, tanpa menimpanya.

## Jalankan di iPhone

Pilih iPhone 17 Pro → **Cmd+R** → **Uji bobot FP16**. Jalankan profil ini
terlebih dahulu setelah membuka ulang aplikasi; cache CoreAI tetap memengaruhi
load. Kirim laporan lengkap. Swift terbaru belum dikompilasi dengan Apple SDK
di workspace Linux ini; ekspor dan pembandingan PyTorch sudah diuji di Linux.

Setiap run harus lolos dua pemeriksaan dengan toleransi yang sama:

1. Keluaran CoreAI dibanding referensi PyTorch untuk bobot tersimpan FP16 dengan
   hitungan FP32: `Pretrained dit-storage16 parity: PASS`.
2. Keluaran yang sama dibanding referensi dari bobot FP32 asli pada input yang
   sama: `Original FP32 baseline parity: PASS`.

Kedua referensi disimpan sebagai Float32 little-endian, diperiksa ukuran dan
SHA-256 sebelum digunakan. Batas setiap elemen tetap `abs(error) <= 1e-4 +
1e-3 * abs(reference)` dan normalized RMSE <= `1e-3`. Batas tidak dilonggarkan.
Delapan shape input, blok 0→1, mask lokal/penuh, CPU-only dan konteks 160/64
tetap sama dengan profil konteks FP32.

## Mengapa hitungan tetap FP32

Audit PyTorch pada input sintetis tetap menunjukkan bahwa aktivasi FP16 menghasilkan
normalized RMSE sekitar 0,001138 terhadap baseline FP32, dengan sekitar 50,8%
elemen melewati batas gabungan di atas. Audit ini dicatat di manifest, lalu
profil yang diekspor tetap memakai hitungan FP32. Kualitas musik dengan aktivasi
FP16 belum diuji; hasil audit ini hanya berlaku pada kasus numerik tersebut.

Untuk bobot tersimpan FP16 dengan hitungan FP32, audit yang sama menghasilkan
normalized RMSE sekitar 4,57e-7 dan seluruh elemen lolos batas. Exporter berhenti
jika pembandingan baseline ini gagal. Perubahan mencakup pembulatan penyimpanan
bobot; nilainya tidak dianggap identik bit per bit dengan FP32.

Konversi bobot ke FP32 dapat terjadi saat specialization atau eksekusi, sehingga
ukuran file setengah tidak membuktikan RAM ikut menjadi setengah. Probe tetap
melaporkan footprint seluruh proses tersampel setiap 10 ms dan penempatan hardware
aktual belum diukur. Full denoiser, loop diffusion dan generasi instrumen lengkap
tetap memerlukan pengujian berikutnya.
