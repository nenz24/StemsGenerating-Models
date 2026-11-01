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
cold launch, loop diffusion atau generasi pengiring dari piano. Gate selanjutnya
adalah loop multi-timestep pendek dengan referensi latent per langkah; belum
diimplementasikan. Tidak ada perubahan model atau kode probe dalam pencatatan
laporan ini.
