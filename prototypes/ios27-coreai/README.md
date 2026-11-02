# CoreAI device and model probes

Initial target: iPhone 17 Pro, iOS 27.0, Xcode 27.0 / 27A266a. These Swift
sources use APIs declared in the six public interfaces supplied by the user.
The user has built and run the device probe on an iPhone with iOS 27.0.1:
**Float32 NDArray round-trip PASS**, CoreAI architecture `h18p`, reported compute
kinds CPU/GPU/Neural Engine. See the [device report](../../reviews/coreai-iphone17pro-device-probe/README.md).
The subsequent [native model report](../../reviews/coreai-iphone17pro-native-affine/README.md)
confirms successful asset validation, loading and three exact affine inference
runs. This workspace is Linux with no Apple SDK, so device runs are user-supplied
evidence, not independent reproductions here.
No stub CoreAI implementation is used.
The folder includes a native Xcode project and shared `CoreAIProbe` scheme.

Satu [run bass preview source CPU-request lengkap](../../reviews/coreai-iphone17pro-bass-preview-cpu-complete/README.md)
sudah tercatat pada console perangkat: 50 putaran denoiser/APG dan decoder
selesai, app tetap terbuka dan pengguna menerima nada/feel bass. Laporan lengkap
mencatat 85,95 s untuk 5,12 s audio dan sampled process maximum 417,19 MiB;
waveform Mac PASS, dua latent akhir NOT PASSED. Full-song/repeatability dan
exact peak/total RAM belum tervalidasi.

**Lihat laporan generasi** dan **Bagikan laporan generasi** memakai `report.txt`
yang tersimpan bersama dua WAV. Ini tetap benar meskipun laporan yang tampil
sudah berubah karena menjalankan probe conditioning. Saat app dibuka kembali,
pasangan WAV/report lengkap terbaru dimuat untuk playback/share tanpa model
inference; audio ini diberi label tersimpan. Progress/error terakhir tetap
tersedia secara terpisah. Helper pemulihan/UI baru menunggu validasi build Apple.

## Pengukuran performa berikutnya

Run kedua membutuhkan 755,29 s, dengan hash WAV sama dan 96,7% waktu di
backend-await. Build flags masih unrecorded. Project sekarang memakai base
`CoreAIProbe-Info.plist` dan compile-time DEBUG fallback untuk mencatatnya.

Pilih scheme **CoreAIProbe Reuse** (Release), Cmd+R, Generate satu kali, lalu
**Bagikan laporan generasi**. Scheme mengaktifkan pool maksimal 15 sampling
model/fungsi untuk 50 langkah; pool dilepas sebelum decoder. Model, preset,
referensi dan toleransi tidak berubah. Guard footprint tetap 2000 MiB, tanpa
jaminan compiler peak/native eviction. Mode ordinary **CoreAIProbe** tetap
Debug satu-scope; **CoreAIProbe Performance** tetap Release satu-scope.
[Instruksi dan batas pengukuran](../../docs/coreai-bass-preview-performance.md).

## Resource build

Default `COREAI_MODEL_PROFILE=auto` membawa pipeline bass source dengan runtime
`.cpuOnly`, menyusul [crash compiler Metal/MPS AOT/default](../../reviews/coreai-iphone17pro-aot-metal-crash/README.md).
AOT dan probe historis tidak ikut app default. Model asli/cache/AOT Mac tetap
tersimpan; file yang tidak berubah tidak perlu disalin ulang.
[Pilihan profil dan ukuran](../../docs/coreai-app-resources.md). Profil `bass-aot`
memilih AOT secara eksplisit untuk diagnosis, dan `all` membawa seluruh resource
serta tombol diagnosis lama.

## Run the device probe

Di Terminal Mac, jalankan dari folder repository:

```bash
git pull --ff-only
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

1. Simpan Apple Team ID dalam konfigurasi lokal, satu kali:

   ```bash
   python3 scripts/configure_coreai_signing.py --team TEAMIDANDA
   ```

   Ganti `TEAMIDANDA` dengan Team ID Apple berisi 10 huruf kapital/angka.
   Debug dan Release membaca `Signing.xcconfig`, yang menyertakan file lokal
   `Signing.local.xcconfig` bila tersedia. File lokal ini diabaikan Git, sehingga
   pengaturan akun tidak perlu mengubah `project.pbxproj` setiap update.
   Jika sudah memilih Team melalui UI Xcode, lihat pemindahan/pemulihan di bawah.
2. Sambungkan iPhone 17 Pro, buka kunci, dan pilih **Trust/Percayai** jika diminta.
   Pilih scheme **CoreAIProbe** dan iPhone tersebut di toolbar Xcode, lalu tekan
   **Run** atau **Cmd+R**. Deployment target sudah disetel ke iOS 27.0.
   Jika diminta, aktifkan **Settings > Privacy & Security > Developer Mode**
   di iPhone dan ikuti instruksinya.
3. Tap **Jalankan pemeriksaan** and share the resulting text. Report any compile,
   link or launch failure as a failure; successful import alone is insufficient.

This allocates a four-element Float32 NDArray, checks its type/shape and reads
the values back. It lists the compute kinds reported as available and the CoreAI
architecture string. It needs no microphone permission, model, network or GPU
download. A Simulator result is labelled and is not an iPhone measurement.
Reported compute availability does not prove a model ran on that accelerator.
The device probe does not call model inference or measure audio generation.

## Pull tertahan perubahan lokal project.pbxproj

Jika `git pull --ff-only` menampilkan **Your local changes would be overwritten**
untuk proyek CoreAI, tutup Xcode terlebih dahulu, lalu jalankan dari repository:

```bash
git fetch origin main &&
coreai_update_script="$(mktemp /tmp/stems-coreai-update.XXXXXX)" &&
git show origin/main:scripts/update_coreai_project.py > "$coreai_update_script" &&
python3 "$coreai_update_script" &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

`fetch` bisa berjalan meskipun perubahan lokal menghalangi pull. Skrip membuat
backup kecil di folder sejajar repository: pbxproj lokal dan staged, patch proyek,
dan konfigurasi signing lokal bila ada. **Folder model, working copy AOT dan
cache tidak diarsipkan.** Team eksplisit yang unik di proyek dipindahkan ke
`Signing.local.xcconfig` yang diabaikan Git; pengaturan lain di include tetap ada.
Tanpa Team eksplisit, include lama dipertahankan byte demi byte.

Hanya pbxproj dipulihkan dari HEAD, lalu branch main di-fast-forward ke hasil
fetch. Perubahan proyek selain Team tersedia di backup, bukan dicampur otomatis
ke proyek terbaru. File kerja lain dan stash tidak dipulihkan/dihapus. Skrip
berhenti sebelum restore jika ada konflik, merge/rebase aktif, branch berbeda,
histori divergen, Team ambigu, atau perubahan file lain bertumpang tindih dengan
update. Ini berbeda dari pemulihan konflik autostash di bagian berikut.

Untuk update berikutnya yang kembali tertahan oleh perubahan proyek Xcode:

```bash
git fetch origin main && python3 scripts/update_coreai_project.py
```

## Konflik setelah Applying autostash

Jika log sudah menyatakan fast-forward, versi repository sebenarnya sudah masuk;
konflik dapat muncul sesudahnya ketika Git mengembalikan pengaturan Xcode lokal.
Tutup Xcode agar file proyek tidak ditulis ulang saat pemulihan. Dari repository:

```bash
git fetch origin main &&
recovery_script="$(mktemp /tmp/stems-coreai-recover.XXXXXX)" &&
git show origin/main:scripts/recover_coreai_autostash.sh > "$recovery_script" &&
bash "$recovery_script" &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

`fetch` dan `git show` dapat dijalankan meskipun working tree berkonflik. Skrip
mengharuskan satu-satunya file konflik adalah proyek CoreAI dan branch dapat
fast-forward ke `origin/main`. Jika file lain berkonflik, skrip berhenti dan
menampilkan status untuk diperiksa.

Sebelum memulihkan proyek, skrip membuat backup di folder sejajar repository:
arsip seluruh folder prototype, versi lokal/upstream proyek dari index konflik,
patch working/staged dan daftar stash. Team ID diambil dari versi lokal yang
tersimpan dan divalidasi sebelum working tree diubah. Jika Team tidak ditemukan
atau ambigu, skrip berhenti setelah backup dan mempertahankan konflik.

Skrip kemudian memulihkan **hanya** file proyek dari HEAD, menyimpan Team dalam
konfigurasi lokal, dan melakukan pull fast-forward dengan autostash dimatikan.
Stash asli tetap ada. Pengaturan proyek lokal lainnya tetap tersedia dalam
backup/stash untuk ditinjau, bukan otomatis dicampurkan lagi. File lain di
working tree tidak dipulihkan oleh skrip.

Setelah Team berpindah ke konfigurasi lokal, update normal memakai
`git pull --ff-only`. Jika menahan pull karena perubahan tracked lain, periksa
perubahan tersebut; jangan mengaplikasikan ulang seluruh autostash lama.

## Compile the model caller

`CoreAIModelProbe.swift` is already included in the target to type-check the
caller. It does not run automatically. It accepts a valid local asset and an explicitly named
function with fixed-shape NDArray inputs and no state/interleaving. It rejects
unsupported contracts instead of inventing source-to-audio preprocessing.
It can enumerate actual function names through `probe.model.functionNames`.
The caller uses `SpecializationOptions.default` unless a preference is supplied;
`preferredComputeUnitKind` is a preference, not proof of actual placement.

This caller is not an exporter, model or audio generator. An API such as
`AIModel(contentsOf:options:)` does not imply safetensors or PyTorch files are
accepted. No ACE-Step conversion route or CoreAI-native export format has been
established. The supplied CoreAICompiler and CoreAICommon public interfaces
contain imports but no compiler/graph-construction declarations.

## Uji model native berikutnya

Dokumentasi resmi sudah terbaca. Format sumber adalah **`.aimodel`**, hasil AOT
**`.aimodelc`**. Ini memperbaiki ejaan `.aimodels` yang sebelumnya diikuti dari
koreksi pengguna sebelum dokumentasi dapat diakses.

Aset `ModelCandidates/affine.aimodel` sudah diekspor dengan **coreai-torch 0.4.3**.
Hash native dibuat otomatis oleh exporter resmi. Input Float32 [1, 4] bernama
`x`, output `y`, fungsi `main`, rumus `y = 2*x + 0.25`. Referensi PyTorch ada di
`affine-native-reference.json`. Tidak ada konversi/unduhan bobot ACE-Step.

Source aplikasi menambahkan tombol **Uji model native**. Tombol itu memuat aset
native melalui CoreAI dengan CPU-only, memeriksa nilai terhadap referensi dalam
3 run, dan mencatat waktu pemuatan terpisah dari run pertama serta 2 run hangat.
CPU-only adalah kebijakan yang diminta; penempatan akselerator tidak diukur.
Pengguna telah menjalankan versi ini pada iPhone: validasi aset true, model
dimuat, 3 run parity PASS, error absolut 0. Rata-rata dua run hangat sekitar
0.770 ms, khusus model numerik empat elemen ini.

Prototype menyalin sumber portable menggunakan folder resource yang sudah ada
untuk menguji pemuatan langsung. Ini belum merupakan integrasi build-time AOT.
Untuk integrasi Xcode standar/AOT, dokumentasi Apple mensyaratkan Metal Toolchain.
Lihat [jalur ekspor resmi dan cara reproduksi](../../docs/coreai-native-export.md).

Paket legacy `.mlpackage` tetap disimpan sebagai arsip percobaan yang ditolak
(`Missing hash file`), bukan sebagai input tombol baru. Inference model native kecil telah lulus;
kecocokan operator ACE-Step dan kualitas musik masih belum terverifikasi.

## Optional command-line build on the Mac

From the repository root, this compiles and links without device signing or
installing an app:

```bash
xcodebuild \
  -project prototypes/ios27-coreai/CoreAIProbe.xcodeproj \
  -scheme CoreAIProbe \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/CoreAIProbeBuild \
  CODE_SIGNING_ALLOWED=NO build
```

The project links the SDK's public CoreAI umbrella framework. SwiftUI/UIKit
use normal SDK autolinking. No CocoaPods, package fetch, project generator,
model download, audio assets or custom build script is required. Xcode build
products and user-specific workspace files are excluded from Git.

## Decoder audio pretrained

[Paket uji decoder](../../docs/coreai-pretrained-decoder.md) sudah disiapkan:
checkpoint asli diverifikasi seluruh SHA-256, reference PyTorch FP32 dibuat,
dan ekspor native penuh berhasil diuji di Linux. Native decoder sekitar 338 MB
bersifat lokal dan diabaikan Git.

[Profil 0,2 detik sudah PASS di iPhone](../../reviews/coreai-iphone17pro-pretrained-decoder-02s/README.md):
tiga run cocok, rata-rata dua run hangat 33,88 ms, footprint proses maksimum
tersampel 792,3 MiB (bukan peak persis atau alokasi model saja).

[Profil 1 detik juga PASS](../../reviews/coreai-iphone17pro-pretrained-decoder-1s/README.md):
rata-rata dua run hangat 159,77 ms, footprint proses maksimum tersampel 849,3 MiB.
Encoder posterior mean 1 detik juga sudah
[PASS di iPhone](../../reviews/coreai-iphone17pro-pretrained-encoder-1s/README.md),
dengan rata-rata dua run hangat 131,33 ms. Footprint proses tersampel 97 MiB
bukan alokasi model terisolasi atau bukti encoder hanya membutuhkan RAM sebesar itu.
Skrip memakai cache bila tersedia.
Paket akhir yang sudah ada diverifikasi dan dipakai ulang; versi rusak tidak
ditimpa otomatis. Paket 1 detik dibuat terpisah; tombol **Uji decoder pretrained**
tetap memakai baseline 0,2 detik. Kedua paket memakai latent seed tetap, bukan
rekaman piano atau generasi pengiring. Ekspor encoder 1 detik juga berhasil di
Linux. Encoder memakai audio multitone sintetis dan menghasilkan posterior mean
latent; [panduan encoder](../../docs/coreai-pretrained-encoder.md) menjelaskan kontrak.
Pengujian ini belum menjalankan diffusion generator lengkap.

## Blok diffusion transformer pretrained

[Satu blok pendek sudah PASS di iPhone](../../reviews/coreai-iphone17pro-pretrained-dit-block/README.md),
rata-rata dua run hangat 4,51 ms. Skrip memverifikasi checkpoint Base lengkap
(unduhan 4,79 GB bila belum ada), lalu mengekspor hanya `decoder.layers.0.*`
dengan self/cross-attention dan MLP asli. Native block sekitar 252 MB;
semua paket codec tetap tersimpan. Tujuh input runtime mencakup hidden states,
conditioning, timestep modulation, RoPE dan masks.

[Panduan blok DiT](../../docs/coreai-pretrained-dit-block.md) menjelaskan batas
uji 16 token sintetis ini. Belum ada generasi instrumen
lengkap, loop diffusion atau pengukuran end-to-end di iPhone.

Berikutnya jalankan `bash scripts/prepare_coreai_dit_block.sh --profile context`,
build ulang, lalu tekan **Uji DiT konteks panjang**. Profil terpisah ini memakai
blok 0 (local radius 128) → 1 (full), 160 hidden / 64 conditioning token,
dengan mask publisher yang berbeda. Native bundle tambahan sekitar 507 MB,
checkpoint Base diambil dari cache, profil pendek dan codec tetap disimpan.
[Panduan konteks panjang](../../docs/coreai-dit-context.md) menjelaskan kontraknya.
[Profil konteks FP32 sudah PASS di iPhone](../../reviews/coreai-iphone17pro-pretrained-dit-context/README.md),
rata-rata dua run hangat 31,10 ms.

Berikutnya jalankan `bash scripts/prepare_coreai_dit_block.sh --profile storage16`,
build ulang, lalu tekan **Uji bobot FP16**. Profil baru menyimpan parameter sebagai
FP16, tetapi hitungan dan input/output tetap FP32. Native asset sekitar 252 MB;
paket lengkap dengan referensi baseline FP32 tambahan sekitar 257 MB. App wajib
meluluskan pembandingan dengan kedua referensi. Baseline konteks FP32 tetap disimpan.
[Panduan bobot FP16](../../docs/coreai-dit-storage16.md) menjelaskan audit precision
dan batas pengukuran RAM.
[Hasil berikutnya sudah PASS pada kedua referensi](../../reviews/coreai-iphone17pro-pretrained-dit-storage16/README.md),
dua run hangat rata-rata 32,46 ms, tetapi footprint tersampel 638,2 MiB tidak
menunjukkan penghematan RAM.

## Denoiser lengkap, satu forward

Jalur monolitik gagal pada loading/specializing dengan `EXC_RESOURCE`, limit
3376 MB, dan kini dinonaktifkan. Tekan Stop (■) jika Xcode paused pada exception.
Jalankan `bash scripts/prepare_coreai_denoiser.sh`, build ulang, lalu tekan
**Uji denoiser bertahap**. Semua 24 layer serta proyeksi/pad/crop asli dijalankan
melalui 14 asset berurutan. Masing-masing asset sekitar 252 MB atau kurang;
total paket sekitar 3,15 GB memakai FP16 stored weights/FP32 math.
Checkpoint Base dipakai ulang. Paket monolitik disimpan di cache di luar folder
resource aplikasi, dan keenam baseline komponen tetap tersedia.

[Panduan denoiser lengkap](../../docs/coreai-full-denoiser.md) menjelaskan kontrak
dan batas memorinya. Tahap validasi/load/inference disimpan secara best effort;
setelah aplikasi dibuka ulang, laporan denoiser terakhir dapat muncul kembali.
Laporan lama diberi label dan tidak otomatis menjalankan model. Output tiap
bagian disalin menjadi tensor CPU mandiri sebelum scope model berakhir; cache
internal CoreAI belum tentu dievikt. Guard berhenti sebelum bagian berikutnya
jika footprint proses mencapai 2000 MiB, tanpa menjamin peak di dalam tahap.
Ekspor dan parity seluruh partisi Linux berhasil.
[Dua laporan iPhone](../../reviews/coreai-iphone17pro-sharded-denoiser/README.md)
mencatat kegagalan BNNS pada bagian 13, kemudian PASS seluruh 14 bagian dan kedua
referensi setelah tombol ditekan ulang dalam aplikasi yang tetap terbuka.
Run PASS memerlukan 3,79 detik. Footprint proses tersampel 40,22 MiB bukan seluruh
kebutuhan RAM model; cache dan penyebab kegagalan pertama belum terukur.

Setelah `git pull --ff-only`, build ulang dan tekan **Uji kestabilan 3x**.
Paket model yang sama dipakai; persiapan/unduhan ulang tidak diperlukan bila sudah
tersedia. Ketiga percobaan disimpan bersama dan PASS hanya jika ketiganya lolos.
Tidak ada penghapusan cache atau restart antar percobaan.
[Wrapper sudah dilaporkan PASS 3/3 di iPhone](../../reviews/coreai-iphone17pro-sharded-denoiser-stability/README.md):
4,68 / 3,46 / 2,04 detik, dengan kedua pembandingan lolos tiap percobaan.
Latensi masih berfluktuasi per bagian dan footprint proses tersampel meningkat;
kebutuhan RAM total dan plateau jangka panjang belum terukur. Hasil ini belum uji
cold launch, loop diffusion atau generasi pengiring dari piano.

## Loop denoising empat langkah

Tombol **Uji loop denoising 4 langkah** sekarang tersedia. Pull dan build ulang;
tidak perlu ekspor model atau unduh checkpoint ulang jika paket 14 asset sudah
ada. Referensi kecil di `ModelCandidates/denoiser-loop-4/` ikut GitHub.
Jadwal `1 → 0,75 → 0,5 → 0,25 → 0`, Euler ODE, guidance=1, shift=1, timestep_r=timestep.
Setiap langkah memakai latent hasil aktual langkah sebelumnya dan memeriksa
velocity serta latent terhadap kedua referensi. Harus selesai 4/4 langkah dan
56/56 tahap. [Hasil iPhone sudah PASS](../../reviews/coreai-iphone17pro-denoiser-loop/README.md):
seluruh 16 pembandingan lolos, total 53,00 detik, normalized RMSE latent akhir
terhadap FP32 asli `7,0143e-6`. Mayoritas biaya ada di interval run+copy pasangan
layer; interval itu juga mencakup binding fungsi, sehingga penyebab waktunya
belum terisolasi. Pengukuran fase tersebut menjadi langkah berikutnya.
[Panduan dan batas pengujian](../../docs/coreai-denoiser-loop.md).

Tombol **Profil waktu loop 4 langkah** menambahkan timer untuk pembukaan model,
binding fungsi, validasi input, await backend, copy output dan sisa overhead.
Laporan merangkum seluruh fase dan interval di luar tahap model. Model, input,
referensi dan semua pemeriksaan parity tetap sama; tidak ada unduhan atau ekspor
baru. [Hasil iPhone sudah PASS](../../reviews/coreai-iphone17pro-denoiser-loop-timing/README.md):
51,88 detik total, dengan 50,75 detik di `backend-await` (99,36% waktu tahap
terukur); opening/binding/copy hanya sekitar 0,10–0,12 detik masing-masing.
Penyebab di dalam `function.run` belum terpisah antara inisialisasi dan eksekusi
steady-state. Langkah berikutnya adalah uji satu pasangan layer/fungsi yang
dipertahankan untuk beberapa run. Belum diimplementasikan dalam pencatatan ini.
[Panduan profiling](../../docs/coreai-loop-timing.md).

Tombol **Uji reuse satu pasangan layer** mempertahankan model layers 0+1 dan
fungsi yang dimuat sekali, lalu mengukur empat call dengan input tetap. Setiap
call wajib lolos terhadap dua referensi host yang ikut GitHub; tidak perlu
menyiapkan atau mengekspor model ulang. [Uji ini sudah PASS di iPhone](../../reviews/coreai-iphone17pro-retained-pair/README.md),
1042,53 ms pertama versus rata-rata 8,52 ms pada tiga run ulang; semua referensi lolos.
[Panduan reuse](../../docs/coreai-pair-reuse.md) dan
[tahapan menuju generasi](../../docs/iphone-generation-roadmap.md).

**Uji loop 24 layer dengan reuse** menyimpan setiap model/fungsi pada penggunaan
pertama untuk dipakai lagi di langkah berikutnya. Targetnya 14 open, 14 bind,
42 hit reuse, 56 call dan seluruh 16 pembandingan lolos. Footprint diperiksa
sebelum setiap tahap dan backend. [Mode ini sudah PASS di iPhone](../../reviews/coreai-iphone17pro-retained-loop/README.md):
18,52 detik total, langkah 3/4 sekitar 0,50/0,53 detik pada run itu.
Kedua uji utama tahap 1 selesai untuk keputusan prototipe. Berikutnya conditioning
rekaman; kelayakan lagu panjang tetap terbuka. [Panduan uji kedua](../../docs/coreai-retained-loop.md).

## Audio-context dari rekaman piano

Jalankan `bash scripts/prepare_coreai_source_context.sh` pada Mac dan pilih
rekaman piano (minimal 5,12 detik). Skrip memakai checkpoint VAE cache dan
mengekspor graph posterior/audio-context baru tanpa menimpa bundle lama.
Jika sudah ada, bundle diverifikasi dan dipakai ulang. Build lalu tekan
**Uji audio-context piano**; targetnya tiga run dan empat output per run lolos.

Outputnya mean/std posterior, sampled source latent dan context `lego` dari
audio nyata. PCM/epsilon/mask dan referensi dibuat pada Mac lalu ikut bundle
lokal ignored, tidak ikut GitHub. Ini belum import rekaman langsung di iPhone,
embedding teks, ConditionEncoder atau generasi bass. [Uji iPhone sudah PASS](../../reviews/coreai-iphone17pro-source-context/README.md):
3/3 run dan 12/12 output, rata-rata backend 1,037 detik, footprint maksimum
tersampel 398,21 MiB. Bagian audio tahap 2 selesai untuk fixture ini; lanjut
conditioning teks/instrumen. [Panduan](../../docs/coreai-source-context.md).

## Conditioning instrumen bass

Jalankan `bash scripts/prepare_coreai_bass_conditioning.sh`, tunggu Ready,
build Cmd+R lalu tekan **Uji conditioning bass**. Tidak ada pilihan rekaman
pada langkah ini. Qwen 0,6B diunduh jika belum ada (sekitar 1,19 GB); Base cache
dipakai ulang. Ekspor menghasilkan 27 aset sekitar 2,42 GB dan mempertahankan
bundle lain. Target: 27/27 tahap dan enam pembandingan PASS.

Lookup token, semua 28 layer Qwen dan ConditionEncoder lengkap (8 lyric +
4 timbre layer, projection, packing) dijalankan di CoreAI. [Uji iPhone sudah PASS](../../reviews/coreai-iphone17pro-bass-conditioning/README.md):
27/27 tahap, 6/6 pembandingan, total 9,46 detik dan footprint maksimum tersampel
783,99 MiB. Kedua gate conditioning selesai secara terpisah untuk fixture ini;
berikutnya integrasi sampling/decoder pada tahap 3. Tokenisasi prompt bass tetap di Mac,
belum ada prompt bebas di iPhone. Tidak ada sampler/CFG atau suara bass pada
gate ini. [Panduan](../../docs/coreai-bass-conditioning.md).

## Generasi WAV bass pendek

Jalankan `bash scripts/prepare_coreai_bass_preview.sh`, tunggu Ready, lalu build
Cmd+R dan tekan **Generate bass 5,12 detik**. Bundle piano dan conditioning bass
sebelumnya dipakai kembali; tidak ada file picker. Persiapan menghitung dua
lintasan referensi 50 langkah dan mengekspor 16 aset sekitar 3,49 GB.

Pipeline menjalankan encoder sumber dan seluruh conditioning secara nyata,
50 APG-guided Euler steps, lalu decoder penuh. Setelah selesai, gunakan tombol
dengar/bagikan WAV bass, campuran piano+bass dan laporan. Target: 50/50 langkah,
10 pembandingan dan WAV generated YES. WAV finite tetap bisa disimpan sebagai
diagnostik saat pembandingan numerik tidak lolos; laporan tidak menandainya PASS.
Ekspor/gate host lolos; build Apple dan pipeline iPhone belum diuji.
[Panduan dan batas fixture](../../docs/coreai-bass-preview.md).

## AOT untuk h18p

Setelah Metal Toolchain terpasang, jalankan
`bash scripts/prepare_coreai_bass_preview_aot.sh`. Skrip memakai source preview
yang ada dan mengompilasi 16 aset khusus `h18p`, dengan resume per aset dan
hash paket. Setelah Ready, Stop/Cmd+R lalu gunakan tombol generasi yang sama.
Aplikasi mencatat pemakaian `.aimodelc` dan policy sampling/decoder `default`
(compiler `--preferred-compute none`); source/Qwen tetap `cpuOnly`. CLI yang
diberikan pengguna tidak menyediakan opsi preferensi CPU. AOT/caller baru
belum divalidasi pada Mac/iPhone dan masih dapat memerlukan specialization.
[Panduan](../../docs/coreai-bass-preview-aot.md).
