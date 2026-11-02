# AOT bass preview untuk iPhone h18p

## Status perangkat dan jalur yang dipakai sekarang

Metal Toolchain **27B47** berhasil mengompilasi seluruh 16 aset h18p di Mac
pengguna, total 3,830 GB. [Log kompilasi](../reviews/coreai-aot-toolchain-27b47/compile-report.txt)
terpisah dari bukti eksekusi iPhone.

[Laporan dan backtrace iPhone terbaru](../reviews/coreai-iphone17pro-aot-metal-crash/README.md)
lolos checksum AOT, source-context piano dan full Qwen/ConditionEncoder.
Proses kemudian crash `EXC_BAD_ACCESS` pada `llvm-worker-0`, saat memasuki aset
AOT pertama. Frame teratas berada di `MetalPerformanceShadersGraph`,
`MPSGraphAICodeCompilerDelegate.getInitializedAICodeBytecodeWithPayloadPrefix:delegateId:`.
Belum ada sampling step selesai atau bass WAV. Ini menunjukkan kegagalan
kompilasi native Metal/MPS; belum membuktikan GPU inference atau RAM habis.

Build default **`COREAI_MODEL_PROFILE=auto` sekarang memilih source + `.cpuOnly`**,
meskipun AOT tersedia. Paket AOT tidak ikut app default dan tetap disimpan di
Mac. Pengelolaan satu scope model/fungsi per aset dan guard footprint 2000 MiB
tetap dipakai. Ini menghindari permintaan default yang memicu jalur compiler
Metal pada percobaan tersebut, tetapi bukan bukti masalah memori BNNS source
sebelumnya telah teratasi. Komponen encoder/conditioning tetap inference nyata,
bukan embedding/audio yang diganti fixture host.

[Console CPU berikutnya dan konfirmasi pengguna](../reviews/coreai-iphone17pro-bass-preview-cpu-complete/README.md)
mencatat seluruh 50 putaran denoiser/APG serta decoder selesai; app tetap terbuka
dan stem bass bisa diputar. Ini satu keberhasilan eksekusi, bukan jaminan
repeatability/peak memory atau strict numerical PASS. Untuk menilai hasil itu,
ambil laporan akhir dan WAV yang sudah ada, tanpa regeneration.

Dari repository, setelah menutup Xcode:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Cmd+R, lalu **Generate bass 5,12 detik** sekali. Tidak perlu mengulang preparation,
AOT atau gate komponen terpisah. Laporan harus menunjukkan:

```text
Sampling/decoder assets: source .aimodel; requested policy cpuOnly; device compilation required
Native phase: model-open/specialization; asset: front.aimodel; cpuOnly requested: true
```

Penanda native ditulis ke saved report dan console sebelum load, bind,
backend-await, readback dan pelepasan scope Swift. UUID dan waktu mulai
membedakan laporan baru dari laporan tersimpan. `EXC_BAD_ACCESS` tidak dapat
ditangkap sebagai Swift error. Jika debugger berhenti pada native crash,
`bt 30` di console LLDB menampilkan hingga 30 frame thread yang dipilih.

## Paket AOT untuk diagnosis eksplisit

Profil **`bass-aot`** masih tersedia untuk diagnosis AOT. Profil ini diketahui
crash pada percobaan pengguna di atas; keberhasilan inference belum terbukti.
[Resource staging](coreai-app-resources.md) membawa source-context, conditioning,
16 aset compiled, serta manifest source preview dan 12 fixture/reference
numeriknya. Salinan 16 model source sampling/decoder dan probe historis tidak
ikut profil AOT. Tidak ada pergantian mode runtime diam-diam bila AOT rusak.

Opsi CLI berasal dari [help pengguna](../reviews/coreai-aot-toolchain-27b47/compile-help.txt):

```bash
xcrun coreai-build compile path/to/front.aimodel \
  --output path/to/output/ \
  --platform iOS --min-deployment-version 27.0 \
  --preferred-compute none --architecture h18p
```

`none` berarti tanpa preferensi, bukan CPU-only. CLI ini hanya menawarkan
`gpu`, `neural-engine`, `none`; **tidak menyediakan opsi CPU-only**. Sesuai
[panduan Apple](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time),
loader AOT eksplisit tetap memakai `.default` agar sesuai compiler `none`.
Memaksakan `.cpuOnly` pada paket compiled ini belum tervalidasi dan tidak dipakai
sebagai mitigasi. Permintaan `.cpuOnly` build default memakai aset source.
Compute unit aktual tetap belum diukur; stack kompilasi MPS tidak membuktikan
inference di GPU selesai.

Untuk menyiapkan atau melanjutkan paket diagnosis pada Mac lain:

```bash
bash scripts/prepare_coreai_bass_preview_aot.sh
```

Skrip memerlukan source preview yang sudah ada, tanpa ekspor PyTorch atau
penghitungan ulang 50 langkah referensi. Enam belas kompilasi dijalankan
berurutan untuk satu arsitektur h18p. Receipt tiap aset mengikat source,
compiler/help/Xcode dan arsitektur. Kompilasi lengkap yang valid dipakai ulang;
parsial dipertahankan di `.runtime-cache`. Source, hasil lama dan checkpoint
cache tidak ditimpa. Linked payload compiler dibuat mandiri sebelum packaging.
Paket final diterbitkan hanya setelah semua hasil valid.

Bundle berada di `ModelCandidates/bass-preview-512-aot/`, diabaikan Git. Loader
memeriksa arsitektur, source-reference identity, 16 nama aset dan seluruh hash.
Hash dibaca per 4 MiB dengan autorelease pool per potongan; progres 1/16 sampai
16/16 dicatat. Manifest source dipertahankan byte demi byte walau model source
sampling/decoder tidak ikut app AOT. Native placement/eviction, loader baru dan
mitigasi CPU default tetap memerlukan validasi perangkat; Linux hanya menguji
argumen, packaging, integritas dan referensi host. Toleransi numerik serta
status `bounded_diagnostic` tidak diubah.
