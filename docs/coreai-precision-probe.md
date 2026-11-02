# Uji biaya bobot DiT BF16 versus FP32

[Baseline 10,24 detik](../reviews/coreai-iphone17pro-audio-import-1024/README.md)
diterima melalui uji dengar, tetapi butuh 772,415 detik. Backend-await mengambil
96,41% waktu. Sebanyak 720 reuse hits tercatat; langkah 3–50 tetap rata-rata
15,191 detik. Ini belum menjelaskan penyebab internal backend.

Eksperimen berikutnya membandingkan satu pasangan layer pada shape yang sama
dengan sampler panjang: CFG batch 2, 128 hidden token, 118 condition token.
Kandidat A menyalin **aset BF16 baseline yang persis sama**. Kandidat B memakai
parameter BF16 yang diperluas ke FP32 sebelum ekspor, dengan nilai learned
parameter harus identik terhadap checkpoint FP32 asli. Kedua jalur menghitung
dalam FP32. Ini tidak mengurangi presisi, langkah, guidance atau mengganti model.
Baseline generasi tetap disimpan; eksperimen belum diaktifkan sebagai sampler.

## Jalankan sekali di Mac dan iPhone

Tutup Xcode, lalu dari folder checkout:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_precision_probe.sh &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Tidak ada pemilihan piano atau perhitungan dua lintasan 50 langkah. Checkpoint
dan aset BF16 profil panjang dipakai ulang. Hanya satu pasangan layer FP32
baru diekspor. Bundle yang sudah lengkap diverifikasi saat perintah diulang;
bundle gagal/berubah tidak ditimpa. Tunggu **Ready**.

Pilih scheme **CoreAIProbe Reuse**, target iPhone, lalu Cmd+R. Tekan
**Uji biaya bobot BF16 vs FP32** sekali. Bagikan laporan dari tombol **Bagikan
laporan** di bawah teks hasil. Tombol **Bagikan laporan generasi** di dekat WAV
lama membagikan laporan baseline, bukan hasil eksperimen ini.

Hanya profil `dit-precision-1024` masuk build: sekitar **761,91 MB / 726,61 MiB**
resource pada ekspor host ini, 16 file. Encoder, Qwen/ConditionEncoder, sisa
22 layer, APG, decoder dan seluruh probe lama tidak ikut. Checkpoint/bundle
generasi tetap berada di Mac. Bobot dan aktivasi uji diabaikan Git.

## Apa yang diukur

Urutan adalah **BF16-A1 → FP32-B → BF16-A2**. Setiap pass membuka satu model dan
mengikat satu fungsi, lalu memanggil fungsi tersebut empat kali dengan delapan
input sintetis yang identik. Tidak ada warmup tersembunyi. Scope Swift dilepas
sebelum pass berikutnya; ini bukan jaminan eviction cache native. Total target
3 opens, 3 binds, 9 reuse hits, **12 panggilan komponen**. Tidak ada generasi WAV.

Semua output wajib finite dan lolos referensi pasangan layer FP32 asli pada
batas abs `1e-4`, rel `1e-3`, normalized RMSE `1e-3`. Host wajib membuktikan
parameter identik dan error output **nol** untuk BF16 versus asli dan FP32
yang diperluas versus BF16. Aset BF16 diperiksa hash terhadap manifest sumber;
kedua aset/tensor diperiksa lagi di app.

Laporan memisahkan panggilan pertama dan rata-rata panggilan 2–4. Termasuk
model-open, bind, validation, backend-await, detached CPU copy, thermal state,
low-power mode dan sampled process footprint. Backend-await adalah interval
wall API, bukan kernel time. Ratio komponen tidak diklaim sebagai percepatan
seluruh sampler. A/B/A membantu melihat perubahan antar-pass, tetapi tidak
mengontrol cold cache, suhu, daya, residency atau semua efek urutan.

FP32 memperbesar aset pasangan layer sekitar dua kali, sehingga compiler dan
memori tetap perlu diperiksa di iPhone. Guard 2000 MiB sebelum pemuatan/call dan
sesudah bind bukan jaminan transient peak native. Tidak ada penghapusan cache,
retry otomatis, AOT/default compute, atau perubahan pada baseline audio.

Jika FP32 lebih cepat dan lolos parity, baru pertimbangkan ekspor sampler
bertahap FP32 dengan anggaran memori/ukuran serta validasi lintasan terpisah.
Jika keduanya lambat, eksperimen tidak membuktikan bahwa konversi BF16 adalah
penyebab. Jangan menurunkan 50 langkah atau melonggarkan gate untuk menamai
hasil lebih cepat.

## Kembali ke generasi 10,24 detik

Untuk bundle Mac yang berstatus `trajectory_drift_diagnostic` seperti baseline
yang sudah diterima, aktifkan lagi bundle lama tanpa ekspor ulang:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_preview_window.py --seconds 10.24 \
  --activate-only --diagnostic-trajectory
```

Lalu Stop/Cmd+R. Untuk bundle yang lolos gate biasa tanpa diagnostic request,
gunakan perintah yang sama tanpa `--diagnostic-trajectory`.

## Bukti host

[Ekspor dan pemeriksaan aktual](../reviews/coreai-precision-probe-export/README.md)
berjalan dengan checkpoint pretrained asli dan exporter resmi. Hasil perangkat
masih pending; ekspor di Linux tidak membuktikan build Swift, native parity,
latency, atau penggunaan accelerator pada iPhone.
