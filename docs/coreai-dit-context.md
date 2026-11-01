# Uji dua blok DiT dengan konteks panjang

[Blok 0 dengan 16 hidden / 8 condition token sudah PASS di iPhone](../reviews/coreai-iphone17pro-pretrained-dit-block/README.md).
Rata-rata dua run hangat 4,51 ms. Input pendek itu belum melewati radius
attention lokal 128 token atau menjalankan blok full-attention berikutnya.

Profil ini memakai **160 hidden / 64 condition token**, lalu menjalankan blok
pretrained **0 (local) → 1 (full)**. Precision tetap Float32 dan kebijakan
perangkat tetap CPU-only. Input aktivasi masih sintetis; ini belum generasi musik.

[Hasil iPhone profil ini sudah PASS](../reviews/coreai-iphone17pro-pretrained-dit-context/README.md):
rata-rata dua run hangat 31,10 ms. Tahap berikutnya adalah
[penyimpanan bobot FP16 dengan hitungan FP32](coreai-dit-storage16.md).
Langkah di bawah tetap tersedia untuk mengulang baseline konteks FP32.

## Jalankan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_dit_block.sh --profile context &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Checkpoint Base dan environment dari uji sebelumnya dipakai ulang; tidak perlu
mengunduh checkpoint lagi jika cache masih tersedia. Jika tidak ditemukan,
ukuran unduhan Base tetap 4,79 GB. Seluruh SHA-256 publisher diperiksa sebelum
bobot digunakan. Hanya dua blok yang dimuat, tanpa training atau GPU CUDA.

Paket baru disimpan terpisah di
`prototypes/ios27-coreai/ModelCandidates/dit-context-pretrained/`.
Profil pendek dan ketiga paket codec tetap tersimpan. File model dan data besar
diabaikan Git; paket yang sudah lengkap diverifikasi saat skrip diulang.
Paket lokal tambahan sekitar 507 MB. Semua resource pengujian yang tersedia
akan disalin Xcode; ukuran tersebut belum menggambarkan paket produk akhir.

## Jalankan di iPhone

Pilih iPhone 17 Pro → **Cmd+R** → **Uji DiT konteks panjang**.
Jalankan profil ini terlebih dahulu setelah membuka ulang aplikasi agar baseline
footprint lebih mudah dibaca. Cache CoreAI tetap dapat memengaruhi waktu load.
Bagikan laporan lengkap. Source Swift terbaru belum dikompilasi dengan Apple
SDK di workspace Linux; pengguna telah menjalankan profil FP32 ini di iPhone,
sementara perubahan terbaru untuk profil bobot FP16 masih perlu diuji.

## Kontrak dan cakupan

| Input Float32 | Shape |
| --- | --- |
| hidden | [1, 160, 2048] |
| condition | [1, 64, 2048] |
| temb | [1, 6, 2048] |
| rope_cos / rope_sin | [1, 160, 128] |
| self_mask / local_mask | [1, 1, 160, 160] |
| cross_mask | [1, 1, 160, 64] |

Output `hidden_out`: Float32 `[1, 160, 2048]`. Delapan input tetap menjadi
argumen runtime. Kedua blok memakai conditioning, modulasi timestep dan RoPE
yang sama, dengan bobot masing-masing dari checkpoint asli. Cache KV dimatikan.

Mask self-attention dibuat memakai fungsi publisher yang diverifikasi. Blok 0
menggunakan mask bidirectional `abs(query-key) <= 128`, ditambah padding pada
key terakhir. Blok 1 memakai mask penuh dengan padding yang sama. Pada query 0,
key 128 masih diizinkan oleh mask lokal dan key 129 diblokir, sementara mask
penuh tetap mengizinkan key 129. Cross-attention memask key conditioning terakhir.
Semua mask aditif memakai nilai Float32 hingga yang finite.

Toleransi tidak diubah: setiap elemen mengikuti `abs(error) <= 1e-4 +
1e-3 * abs(reference)`, dan normalized RMSE <= `1e-3`. Tiga run mengukur
run+read-back; footprint seluruh proses tersampel tiap 10 ms, bukan peak persis
atau seluruh memori perangkat. Penempatan hardware aktual tidak diukur.

Profil ini memperluas cakupan mask, panjang sekuens, conditioning dan akumulasi
error dua blok. Perbandingan waktu dengan satu blok pendek bukan benchmark
scaling terkontrol karena jumlah blok, panjang input dan pola mask berbeda.
Tidak ada klaim durasi audio dari tensor sintetis ini. Full denoiser, conditioning
dari piano/teks, seluruh loop sampling, anggaran precision/memori dan kualitas
musik tetap memerlukan pengujian tersendiri.
