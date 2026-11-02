# Tahap 2b: conditioning instrumen bass

Audio-context piano sudah [PASS di iPhone](../reviews/coreai-iphone17pro-source-context/README.md).
Berikutnya: lookup embedding token, seluruh 28 layer Qwen3, text projector,
delapan layer lyric encoder, empat layer timbre encoder, normalisasi dan packing.
Semua perhitungan neural ini diekspor ke CoreAI; output bukan embedding siap pakai
yang hanya dibaca dari fixture.

## Jalankan di Mac

```bash
git pull --ff-only &&
bash scripts/prepare_coreai_bass_conditioning.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

**Tidak ada dialog pilih rekaman pada langkah ini.** Caption mengikuti preset
`configs/electric-bass.json`; instruksinya `Generate the bass track based on the
audio context:`. Format prompt/metadata/lyrics memakai helper runtime yang dipin.
Klip 5,12 detik diformat helper menjadi `duration: 5 seconds`; BPM/key/time
signature tidak diisi. Lirik kosong dan bahasa `unknown` mengikuti parameter
default lego tanpa LM. Ada 105 token teks dan 11 token lyric, seluruhnya valid
tanpa padding untuk satu contoh.

Cache Python dan checkpoint Base sebelumnya dipakai ulang. **Qwen3-Embedding-0.6B
sekitar 1,19 GB** perlu diunduh bila belum ada; tokenizer dan silence latent
publisher juga diperlukan. Seluruh checksum diverifikasi. Ekspor host yang
sudah dijalankan menghasilkan **27 aset native sekitar 2,42 GB**. Mac mengekspor
aset lokalnya sendiri; byte/ukuran native dapat berbeda antarplatform. Bundle
codec, audio-context dan denoiser sebelumnya dipertahankan.

Tunggu terminal menampilkan **Ready**, build **Cmd+R** ke iPhone, lalu tekan
**Uji conditioning bass** dan bagikan laporan lengkap. Targetnya satu lintasan
dengan enam pembandingan terhadap dua referensi:

```text
Bass text/ConditionEncoder dual parity: PASS (all 40 layers, six checks)
Completed conditioning stages: 27/27; parity checks: 6/6
```

Tidak perlu mengulang audio-context atau tahap 1. Jika `missingPreparedBundle`,
pastikan skrip selesai dengan Ready. Jika masih terjadi setelah persiapan
sukses, lakukan Clean Build Folder lalu Cmd+R. Jika persiapan gagal, kirim
output terminalnya.

## Kontrak dan validasi

Qwen memakai causal attention dari implementasi aslinya. ACE lyric/timbre
memakai bidirectional attention, sliding radius 128 atau full attention sesuai
urutan layer. RoPE/mask per aset sesuai panjang token uji. Tidak ada KV cache
atau decoding autoregresif. Bobot disimpan FP16 dengan hitungan FP32, tanpa
klaim penghematan RAM runtime.

Tidak ada referensi timbre pengguna pada preset ini. Jalur asli memakai 750
frame pertama silence latent publisher setelah transpose `[B,C,T]` menjadi
`[B,T,C]`. Timbre encoder tetap dijalankan penuh. Pada satu referensi dan semua
token valid, unpack/packing menjadi reshape dan concat **lyric → timbre → text**.
Spesialisasi ini dibandingkan di host dengan ConditionEncoder asli, termasuk
packing/unpacking aslinya.

Embedding Qwen `[1,105,1024]`, condition `[1,117,2048]` dan mask Float32 all-one
`[1,117]` masing-masing dibandingkan terhadap hasil pipeline FP16-storage dan
model FP32 utuh. Batas tetap `abs(error) <= 1e-4 + 1e-3 * abs(reference)` dan
normalized RMSE `<=1e-3`. Mask wajib tepat all-one. Token dikirim sebagai Float32
yang merepresentasikan integer secara tepat, divalidasi, lalu dikonversi ke
integer oleh graph lookup embedding.

Model dimuat berurutan; output disalin ke scalar CPU baru sebelum model/fungsi
Swift dilepas. Eviction cache backend tidak dijamin. Guard footprint 2000 MiB
sebelum setiap aset bukan jaminan peak OS. Ukuran native bukan ukuran RAM.

## Status dan batas

[Ekspor dan host parity sudah berhasil](../reviews/coreai-bass-conditioning-export/README.md)
dengan checkpoint nyata. Build Apple dan parity perangkat untuk kode baru
masih menunggu; Linux tidak memiliki Apple SDK.

**Tokenisasi masih di Mac untuk preset tetap.** Lookup embedding dan seluruh
neural inference berjalan di CoreAI di iPhone saat uji. Prompt bebas dan
tokenizer langsung di iPhone belum diimplementasikan. Host references hanya
untuk pembandingan. Batch dengan padding, lirik lain atau timbre reference
lain belum didukung spesialisasi ini.

Uji ini belum menggabungkan audio-context, conditioning, denoiser, CFG dan
decoder, serta belum menghasilkan bass. Condition 117 token dan source-context
128 frame memerlukan penyesuaian kontrak denoiser sebelum integrasi; fixture
sintetis lama 64 token/25 frame tidak langsung dipakai. Setelah gate ini lolos,
pekerjaan berikutnya adalah merangkainya menjadi WAV pendek pada tahap 3.
