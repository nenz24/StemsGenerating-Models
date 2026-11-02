# Perbandingan satu aset dan banyak aset bergantian

Rancangan berdasarkan main `d0b88c0f7f39ca40d5d537befd3cb53749d1c899`.
Caller, preparation dan staging diagnostik tersedia; build Apple dan run
perangkat kini memiliki [satu laporan K=2](../reviews/coreai-iphone17pro-interleaving-k2/README.md).
Run menyelesaikan 140 call; strict parity tetap NOT PASSED 40/140. Switching
+3,18–6,81% lebih kecil daripada drift kontrol +13,38%; thermal fair. Langkah
berikutnya adalah K=2 urutan terbalik dari nominal, tanpa preparation ulang.
Baseline BF16 dan batas numerik dipertahankan.

[Run urutan terbalik](../reviews/coreai-iphone17pro-interleaving-k2-reversed/README.md)
juga selesai: switching −4,95–6,82% (lebih cepat), kontrol single-after +15,06%,
thermal fair. Efek timing berbalik dengan urutan; penalti switching K=2 belum
konsisten. Berikutnya kontrol ketiga grouped-first dari thermal nominal,
menggunakan bundle/build yang sama. Tidak ada perubahan model atau toleransi.

[Kontrol nominal ketiga](../reviews/coreai-iphone17pro-interleaving-k2-nominal/README.md)
tetap menunjukkan drift kontrol +13,51%. Caller kini memakai progres live
pendek dan breadcrumb native terpisah agar pekerjaan pelaporan tidak bertambah
seiring call. Laporan lengkap disimpan sebelum inferensi, di batas mode dan
saat selesai/error tertangkap; crash dapat kehilangan detail mode yang sedang
berjalan. Perubahan instrumentasi belum diuji di Apple. Update kode/build
tanpa preparation, ulangi K=2 grouped-first dari nominal, dan tandai hasil
sebagai cohort `bounded-live-progress-v1` sebelum memutuskan K=4.

[Kalibrasi bounded-progress](../reviews/coreai-iphone17pro-interleaving-k2-bounded/README.md)
telah selesai: nominal, drift kontrol +0,92% dan switching +1,28%/−1,32%.
Keputusan terbaru adalah lanjut K=4 dengan instrumentasi yang sama; gunakan
`bash scripts/prepare_coreai_interleaving_probe.sh --pairs 4 --expand-existing`.
Bundle K=2 diarsipkan setelah bundle K=4 diverifikasi. Target K=4 adalah
220 calls, 14 opens/binds dan 220 prebound hits; belum ada hasil perangkat K=4.

[Hasil K=4](../reviews/coreai-iphone17pro-interleaving-k4/README.md) kini tersedia:
220 calls nominal, drift kontrol +1,68%, mean per pasangan 51–54 ms, tanpa
perlambatan switching besar yang teramati. Strict parity tetap NOT PASSED
120/220. Langkah terbaru K=8 memakai `--pairs 8 --expand-existing`; target
380 calls/26 opens/26 binds/380 prebound hits. K=4 diarsipkan setelah bundle
baru diverifikasi. Instrumentasi dan batas tetap, belum ada hasil K=8.

[K=8 kini selesai](../reviews/coreai-iphone17pro-interleaving-k8/README.md):
380 calls nominal, mean per pasangan 52–56 ms, switching +0,93–3,51% dan drift
kontrol +4,47%. Parity NOT PASSED 240/380; 14–15 juga gagal pointwise di kedua
mode, tanpa perubahan hash output. Berikutnya K=12, 540 calls/38 opens/binds,
memakai `--pairs 12 --expand-existing`. Sweep ini tetap bukan full-loop;
front/tail/APG dan actual input/feedback belum diuji oleh fixture komponen.

## Persiapan dan menjalankan

Gunakan checkout yang sudah memuat perubahan ini di Mac. Bundle
`bass-preview-1024` dan `dit-precision-1024` yang sudah diverifikasi harus ada;
preparation tidak mengekspor ulang aset atau membuat FP32 sampler baru.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
bash scripts/prepare_coreai_interleaving_probe.sh --pairs 2
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Tunggu Ready. Pilih CoreAIProbe Reuse / Release dan iPhone, Cmd+R. Tekan
**Uji 2 aset BF16 — berkelompok dahulu**. Setelah selesai, bagikan laporan
dari bagian hasil diagnostik. Lakukan tiga pengulangan dengan urutan mode
diseimbangkan memakai tombol **bergantian dahulu** pada pengulangan kedua.
Biarkan thermal kembali ke kondisi awal di antara run. Tiap run K=2 menjalankan
140 call; tanpa warmup atau feedback output. Jika run berhenti, simpan laporan.

Naikkan ukuran hanya sesudah run kecil ditinjau. `--pairs 4/8/12` dengan
`--expand-existing` mempersiapkan bundle baru terlebih dahulu, memverifikasi
kesamaan seluruh payload bersama dan manifest sumber, lalu mengarsipkan bundle
diagnostik lama ke `.runtime-cache/coreai-interleaving-archives/`. Jika install
baru gagal dan tujuan masih kosong, bundle lama dipulihkan. Persiapan gagal
tidak mengubah bundle lama/selector. Tanpa flag eksplisit, ukuran berbeda
ditolak. Folder output terpisah juga dapat dipakai. Folder aktivasi harus bernama
`dit-interleaving-1024`. Tombol hanya menampilkan ukuran yang ada di bundle.
Resource build berisi probe saja; bundle generasi di Mac tetap disimpan.
Untuk kembali ke baseline generasi, gunakan perintah aktivasi ulang pada
[panduan precision](coreai-precision-probe.md#kembali-ke-generasi-1024-detik).

### Verifikasi implementasi di workspace Linux

Enam tes baru lulus: distribusi kerja/switching dan accounting, geometri long,
korupsi dan selector atomik, precision/toleransi/identitas sumber, duplikasi
aset, isolasi staging dan preservasi baseline. Tes regresi tambahan memberi
24 kelulusan; dua tes tidak dapat berjalan karena `torch` dan
`openstep_parser` belum terpasang. Pemeriksaan sintaks Python, shell dan
`git diff --check` lulus. Fixture tes adalah payload nonexecutable.

Bundle BF16 panjang dan precision fixture belum tersedia di workspace ini;
preparation host aktual belum dijalankan. Apple SDK/Swift compiler tidak
tersedia, sehingga build dan native runtime masih memerlukan Mac/iPhone.
Tidak ada angka performa baru atau klaim numerical PASS dari implementasi ini.

## Dasar keputusan

[Precision A/B/A](https://github.com/nenz24/StemsGenerating-Models/blob/d0b88c0/reviews/coreai-iphone17pro-precision-aba/README.md)
mencatat repeated mean BF16 A1 55,964 ms, expanded FP32 B 53,533 ms dan BF16 A2
53,336 ms. Tidak ada keuntungan FP32 konsisten. Strict parity NOT PASSED 0/12:
satu titik melampaui bound pada semua call. RMSE kecil tidak membatalkan
kegagalan pointwise. Tidak ada bukti full tensor bit-identical dari ringkasan itu.

[Panduan performa](https://github.com/nenz24/StemsGenerating-Models/blob/d0b88c0/docs/coreai-bass-preview-performance.md)
mencatat pipeline 10,24 detik memerlukan 772,415 s; backend-await 744,691 s
(96,41%), sampled maximum 1.276,660 MiB. Reuse hits tidak membuktikan percepatan.
Working set, perpindahan bobot native dan pergantian aset masih hipotesis.

## Kontrak tetap

- Gunakan aset pasangan layer BF16 asli profil 10,24 detik dengan hash sumber
  terverifikasi. Jangan menduplikasi satu aset dengan nama berbeda untuk
  mengklaim banyak aset berbeda. Jangan ekspor sampler FP32 baru.
- Learned weights BF16 storage, perhitungan/input/output FP32; VAE tetap FP32.
  Geometry CFG batch 2, hidden 128 token, condition 118 token.
- Delapan input sintetis tetap sesuai kontrak precision probe. Setiap pasangan
  layer memiliki referensi original FP32 host sendiri pada input identik.
  Output suatu call tidak menjadi input call berikutnya.
- Abs 1e-4, rel 1e-3, normalized RMSE 1e-3 tidak berubah. Semua call diperiksa.
  Finite parity failures boleh dicatat sambil meneruskan jadwal diagnostik,
  tetap NOT PASSED. Hash/shape/type/nonfinite/native error dan guard memori
  menghentikan run; tidak ada retry/purge/warmup tersembunyi.
- Sampler produksi, checkpoint, seed, guidance 7, shift 1 dan 50 langkah APG
  tetap. Benchmark komponen tidak menghasilkan WAV atau mengganti aktivasi
  baseline generasi yang sudah diterima melalui uji dengar.

## Matriks yang memisahkan pool dan switching

Gunakan pasangan layer berurutan: A=00–01, B=02–03, hingga L=22–23.
Pool diuji bertahap K=1,2,4,8,12. Berhenti bila guard gagal; K=12 belum
merupakan keseluruhan pipeline karena front/tail/APG tidak termasuk.
Untuk setiap K>1 jalankan:

| Mode | Handle tertahan | Urutan inference | Tujuan |
| --- | --- | --- | --- |
| Single-before | A | A × 20 | Kontrol awal A |
| Pool-only | K | A × 20 | Biaya A dengan handle lain hanya dibuka/bind |
| Grouped | K | A × 20, B × 20, … | Kontrol jumlah call dan kerja yang sama |
| Interleaved | K | (A,B,…K) × 20 | Efek switching dengan pool/jumlah call sama |
| Single-after | A | A × 20 | Kontrol drift sesudah beban |

Setiap mode memakai scope model/fungsi sendiri, dilepas setelah mode berakhir.
Dalam grouped dan interleaved, seluruh K model/fungsi dibuka/bind dalam urutan
yang sama sebelum inference. Pool-only tidak memanggil aset lain; binding
tidak membuktikan bobot sudah resident. Panggilan pertama setiap aset tetap
tercatat dan dianalisis terpisah. Jangan menamai pelepasan Swift sebagai native
eviction. Tidak ada inference pemanasan tambahan.

Seluruh mode melakukan prebind sebelum inference, termasuk single. Target
single 1 open/1 bind/20 calls/20 prebound pool hits; pool-only K opens/K binds,
20 calls/20 hits; grouped/interleaved K opens/K binds/20K calls/20K hits.
Hit berarti lookup handle yang sudah ada, termasuk inference pertama; bind
saja tidak menambah hit. Calls 2–20 tetap dipisah untuk statistik berulang.

## Pelaporan dan pengulangan

Gunakan Release/-O/DEBUG OFF, cpuOnly yang sama, foreground, device/OS/build,
fixture dan hash bundle sama. Catat low-power, pengisian, thermal sebelum dan
sesudah mode/call. Guard proses 2000 MiB sebelum load/call dan sesudah bind
tetap; ini tidak menjamin transient peak native aman. Muat K secara bertahap.

Jalankan minimal tiga pengulangan per K, menyeimbangkan urutan grouped dan
interleaved. Kembali ke kondisi thermal awal sebelum pengulangan; jangan
menghapus run gagal. Fresh launch bukan bukti compiler cache dingin.

Simpan per call: run/mode/K/urutan, asset hash, fixture hash, open/bind,
validation/backend-await/output-copy/other/total ms, thermal dan sampled
footprint. Simpan finite/parity, jumlah pelanggaran, worst bound ratio dan
indeks/actual/reference/bound, max abs error serta normalized RMSE. Bila
memungkinkan simpan hash seluruh output untuk membandingkan mode; scalar
ringkasan yang sama tidak membuktikan seluruh tensor identik.

Ringkas first-call dan mean/median/p95/min/max calls 2–20 per aset per run.
Bandingkan A terhadap dirinya sendiri: pool-only/single untuk efek pool;
interleaved/grouped untuk efek switching pada kerja/pool identik. Bandingkan
juga masing-masing B…K terhadap kontrol grouped-nya. Total waktu saja tidak
adil untuk single versus interleaved karena jumlah call berbeda. Tampilkan
sebaran antar-run dan strict parity terpisah dari timing collection.

Backend-await adalah wall interval API, bukan kernel time; sampled process
maximum bukan exact peak/total RAM. Slowdown terulang hanya mendukung pengaruh
pola kerja yang diuji, belum membuktikan mekanisme cache atau native residency.
Rasio timing dengan parity gagal tetap diagnostik, bukan penerimaan sampler.

## Urutan menuju lagu penuh di iPhone

1. Build dan jalankan bundle/caller diagnostik terpisah. Host memeriksa
   hash/kontrak dan schedule/count accounting; build Apple dan run iPhone
   tetap diperlukan, Linux tidak mengemulasi CoreAI.
2. Gunakan hasil matriks untuk memilih eksperimen lifecycle berikutnya, lalu
   ulangi pipeline bass 10,24 detik dengan gate dan baseline yang sama.
3. Naikkan durasi atau uji chunk/context dengan penilaian sambungan, ritme,
   tonalitas, konsistensi antarseksi dan referensi yang cocok.
4. Validasi drum dan gitar secara terpisah, lalu arrangement bass+drum+gitar:
   sinkronisasi stem, mixing/headroom, memori kumulatif, thermal dan recovery.
5. Penerimaan akhir memakai satu lagu penuh yang benar-benar dihasilkan di
   iPhone. Laporkan waktu total, memori, kelengkapan stem dan uji dengar;
   jangan mengekstrapolasi performa klip secara linear atau menyamakan
   persetujuan musikal dengan numerical PASS.
