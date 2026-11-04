# Performa preview bass CoreAI

Tiga laporan perangkat kini tersedia untuk audio yang sama:

| Pengukuran | Awal, satu scope | Ulang, satu scope | Release + reuse |
| --- | ---: | ---: | ---: |
| Durasi audio | 5,12 s | 5,12 s | 5,12 s |
| Waktu pipeline | 85,95 s | 755,29 s | **59,24 s** |
| Sampled maximum proses app | 417,19 MiB | 436,47 MiB | **450,42 MiB** |
| Sampling | 50/50 | 50/50 | 50/50 |
| Build flags tercatat | Tidak tersedia | Unrecorded | **Release / -O / DEBUG OFF** |
| Dua waveform vs referensi Mac | PASS | PASS | PASS |

[Laporan reuse lengkap](../reviews/coreai-iphone17pro-bass-preview-reuse/README.md)
mengonfirmasi 15 opens, 15 binds, 735 reuse hits dan 750 sampling calls, lalu
pool dilepas sebelum decoder. Hash dua WAV persis sama dengan baseline uji
dengar yang disetujui. Delapan output comparison PASS, dua latent terakhir
NOT PASSED; host `bounded_diagnostic` menjaga overall gate NOT PASSED.

Waktu 59,24 s sekitar 12,75 kali lebih cepat daripada run 755,29 s yang diamati,
dan 1,45 kali daripada run awal 85,95 s. Ini perbandingan antar-run, bukan
controlled cold-cache/thermal benchmark atau bukti pengaruh Release terpisah.
Untuk audio 5,12 s, rasio wall/audio masih 11,57: belum real-time.

Backend-await run reuse adalah 37,14 s; copy output 7,71 s. Step pertama
15,07 s, langkah 2–50 rata-rata 651,59 ms. Backend-await adalah interval API,
bukan kernel time. Penyebab internal seluruh perbedaan antar-run belum
terisolasi. Angka memori adalah sampled app process, bukan exact peak atau
total model/device RAM, dan bukan jaminan skalabilitas full song.

## Jalur prototype yang dipilih

**Profil 10,24 detik sudah menyelesaikan satu run**, dengan [laporan dan audit
WAV](../reviews/coreai-iphone17pro-audio-import-1024/README.md): 772,415 detik,
footprint maksimum tersampel 1.276,660 MiB, 30 opens/30 binds/720 hits/750 calls.
Langkah pertama memakai scope tunggal per aset, lalu reuse mulai langkah kedua.
Langkah 3–50 tetap rata-rata 15,191 detik; backend-await 744,691 detik atau
96,41% total pipeline. Reuse handles tercatat tetapi tidak membuktikan percepatan
backend pada profil ini. Penyebab internal, thermal dan cache state belum
diisolasi. Klip diterima melalui uji dengar, bukan sebagai anggaran performa.
Recovery cache 0/1 berarti jalur repair belum teruji oleh run yang selesai ini.
[Uji pasangan layer BF16 versus FP32](coreai-precision-probe.md) sudah
[menyelesaikan A/B/A](../reviews/coreai-iphone17pro-precision-aba/README.md).
Mean berulang BF16 55,964 → FP32 53,533 → BF16 53,336 ms: tidak ada keuntungan
FP32 yang konsisten dalam eksperimen ini. Strict parity tetap NOT PASSED,
0/12; semua panggilan mencatat satu titik pelanggaran yang sama. Jangan
memperbesar sampler ke FP32 berdasarkan hasil ini. Fokus diagnosis berikutnya
adalah ukuran pool dan interleaving aset pada input tetap. Working set/native
weight movement masih hipotesis, belum penyebab yang dibuktikan. Uji komponen
ini tidak mereproduksi kondisi full generation 772 detik.

[Probe interleaving BF16](coreai-asset-interleaving-plan.md) kini menyiapkan
perbandingan input tetap: single, pool-only, grouped dan interleaved. Mulai
dengan dua pasangan layer, 140 call per run. [Run K=2 pertama](../reviews/coreai-iphone17pro-interleaving-k2/README.md)
selesai: interleaved/grouped +6,81% dan +3,18%, tetapi kontrol single-after
melambat +13,38%. Thermal fair sepanjang run; strict parity NOT PASSED 40/140.
Hasil belum mengisolasi switching atau menjelaskan pipeline penuh. Ulangi
K=2 dengan urutan terbalik dari thermal nominal, tanpa export ulang.
Benchmark ini tidak mengganti sampler atau toleransi.

[Run K=2 urutan terbalik](../reviews/coreai-iphone17pro-interleaving-k2-reversed/README.md)
kemudian selesai: interleaved/grouped menjadi −4,95% dan −6,82%, sementara
kontrol single-after melambat +15,06%. Thermal tetap fair. Keuntungan timing
berbalik mengikuti urutan mode; belum ada penalti switching K=2 yang konsisten.
Parity tetap NOT PASSED 40/140 dan hash output per aset cocok di kedua run.
Lengkapi kontrol ketiga grouped-first dari thermal nominal sebelum memutuskan
pool lebih besar. Tidak perlu preparation atau ekspor ulang untuk kontrol ini.

[Kontrol ketiga](../reviews/coreai-iphone17pro-interleaving-k2-nominal/README.md)
nominal sepanjang 140 call, tetapi single-after tetap +13,51%; interleaved
+8,20%/+4,37%. Pelaporan live sebelumnya menyalin/menulis teks yang terus
bertambah. Caller diagnostik kini membatasi progres live dan menyimpan laporan
lengkap di batas mode/akhir; penyebab drift belum dibuktikan. Update kode/build
saja, ulangi K=2 grouped-first dari nominal sebelum memperbesar pool. Aset,
baseline generasi, copies dan toleransi tetap; cari penanda
`bounded-live-progress-v1` pada laporan baru.

[Run K=2 bounded-progress](../reviews/coreai-iphone17pro-interleaving-k2-bounded/README.md)
selesai dengan thermal nominal: drift kontrol +0,92%, interleaved/grouped
+1,28%/−1,32%, sampled maksimum 206,721 MiB. Ini mendukung adanya confound
pelaporan, bukan bukti penyebab tunggal atau percepatan full-loop. Parity tetap
NOT PASSED 40/140. Berikutnya K=4, 220 call, memakai `--pairs 4 --expand-existing`;
K=2 diarsipkan setelah bundle baru lolos verifikasi. Tidak ada export bobot native.

[K=4 selesai](../reviews/coreai-iphone17pro-interleaving-k4/README.md): 220 calls,
14 opens/binds, nominal sepanjang run. Mean berulang per pasangan 51–54 ms;
interleaved/grouped +1,22%/−0,09%/−2,15%/−2,80%, drift kontrol +1,68%.
Sampled maksimum 202,502 MiB bukan total RAM native. Parity NOT PASSED 120/220:
00–01 tetap gagal, tiga pasangan lain lolos. Belum ada perlambatan besar K=4
atau penjelasan full-generation. Berikutnya K=8 dengan driver yang sama:
`--pairs 8 --expand-existing`, 380 calls; K=4 diarsipkan, BF16/batas tetap.

[K=8 selesai](../reviews/coreai-iphone17pro-interleaving-k8/README.md): 380 calls,
26 opens/binds, nominal sepanjang run. Mean berulang tetap 52–56 ms;
switching +0,93–3,51%, drift kontrol +4,47%. Belum mereproduksi perlambatan
besar pipeline penuh. Parity NOT PASSED 240/380: selain 00–01, pasangan baru
14–15 gagal (6 elemen/output), konsisten di grouped/interleaved. Berikutnya
lengkapi sweep pasangan K=12 dengan `--pairs 12 --expand-existing`: 540 calls,
38 opens/binds. Memori proses tersampel 97,677 MiB bukan bukti residency atau
penurunan total RAM model. Tidak ada perubahan BF16 atau toleransi.

[K=12 selesai](../reviews/coreai-iphone17pro-interleaving-k12/README.md): 540 calls,
38 opens/binds, nominal; mean per pasangan 52–56 ms, drift kontrol +5,64%,
switching +0,48–6,04%. Parity NOT PASSED 400/540; tetap hanya 00–01 dan 14–15
yang gagal. Sweep pool pasangan selesai tanpa mereproduksi perlambatan besar.
Berikutnya [diagnosis prefix input aktual](coreai-actual-prefix-probe.md): tiga
langkah pertama dari schedule 50 yang sama, front/tail/APG dan feedback aktual,
45 sampling calls, tanpa decoder/WAV. Default generasi tetap 50 langkah.

[Prefix aktual selesai](../reviews/coreai-iphone17pro-actual-prefix-3/README.md):
input/PCM cocok dengan baseline, 45 calls/30 opens/binds/15 hits, nominal.
Step 1/2 sekitar 14,9 s, tetapi step 3 **1,069 s** dengan backend 760,580 ms;
tail/APG sangat kecil. Ini satu warm step, bukan bukti 50 langkah lebih cepat.
Progres sampling/decoder panjang pada jalur produksi kini dibatasi juga, dengan
penanda `generation-bounded-v1` dan delta API per step. Berikutnya satu run
generasi **50/50** pada piano yang sama, tanpa preparation/export ulang; bandingkan
laporan dan hash WAV baseline. Semua model, dtype, gate dan jumlah langkah tetap.

[Full run bounded selesai; keluhan glitch ringan diabaikan atas permintaan pengguna](../reviews/coreai-iphone17pro-generation-bounded-glitch-review/README.md):
50/50 dalam 96,438 s, mean step 3–50 1,058 s; 750 sampling + satu decoder,
30 opens/binds/720 hits. Ketiga WAV unggahan cocok persis dengan hash baseline;
PCM piano dan rekonstruksi mix kini diverifikasi independen. Pengguna mendengar
glitch pada bass solo dan mix di app, lalu meminta mengabaikannya karena ringan.
Investigasi dihentikan atas arahan pengguna; keluhan ini tidak menghambat
preview drum/gitar berikutnya. Tidak ada DSP atau perubahan sampler untuk itu.
Runtime lebih cepat dan kesamaan WAV tidak mengubah status numerical gate.

**Source CPU memakai reuse secara default saat fresh launch tanpa override**,
termasuk membuka app Release dari ikon iPhone. AOT yang sengaja disertakan
masih memakai mode AOT satu-scope; hasil CPU ini tidak memvalidasinya. Default
baru hanya mengubah pemilihan lifecycle ke jalur yang sudah diuji pada klip ini.
Model, guidance 7, 50 langkah APG, dtype, referensi dan batas numerik tetap sama.

Pool lazy berisi maksimal 14 shard DiT + satu APG model/fungsi, dilepas sebelum
decoder dan saat gagal. Encoder piano dan Qwen tidak masuk pool. Guard proses
2000 MiB sebelum stage dan sesudah bind tetap aktif. Retention dapat menaikkan
live memory; guard tidak menjamin transient compiler peak/native eviction.
Untuk 5,12 detik tidak ada retry baru. Profil 10,24 detik mencatat satu budget
recovery khusus BNNS cache bind sebelum inference; [syarat dan accounting](coreai-bass-preview-1024.md)
tetap eksplisit. Ini tidak mengulang langkah atau mengganti metode sampling.

## Scheme dan penggunaan berikutnya

- **CoreAIProbe Reuse**: Release, `COREAI_PREVIEW_REUSE=1`; jalur rekomendasi.
- **CoreAIProbe Performance**: Release, override `0`; kontrol satu-scope.
- **CoreAIProbe**: Debug, override `0`; kontrol satu-scope.

Override hanya berlaku untuk proses yang diluncurkan scheme tersebut. Fresh
launch tanpa override memilih reuse untuk source CPU. Membuka ikon saat proses
lama masih hidup tidak mengubah environment proses itu.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih **CoreAIProbe Reuse** untuk pengembangan berikutnya. Tidak perlu generate
ulang untuk menyimpan hasil yang sudah diterima ini. Bila mengukur perubahan
selanjutnya, gunakan input yang sama dan **Bagikan laporan generasi**, termasuk
hash WAV, flag build dan ringkasan API timing. Base `CoreAIProbe-Info.plist` dan
DEBUG fallback kini terbukti bekerja pada report perangkat. SDK/build untuk
pemilihan default terbaru masih perlu verifikasi Mac; jalur reuse sendiri
sudah teramati selesai.

Baseline musikal 5,12 detik dan baseline runtime pendek ini tidak menggantikan
baseline CUDA 90 detik. Cold starts/repeatability, input rekaman baru langsung
di iPhone, durasi lebih panjang, instrumen lain dan full song tetap belum
tercakup. Jangan mengekstrapolasi waktu/memori klip ini secara linear ke lagu
penuh atau mengklaim numerical PASS dari persetujuan pendengaran.
