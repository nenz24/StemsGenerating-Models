# CoreAI device and model probes

Initial target: iPhone 17 Pro, iOS 27.0, Xcode 27.0 / 27A266a. These Swift
sources use APIs declared in the six public interfaces supplied by the user.
The user has built and run the device probe on an iPhone with iOS 27.0.1:
**Float32 NDArray round-trip PASS**, CoreAI architecture `h18p`, reported compute
kinds CPU/GPU/Neural Engine. See the [device report](../../reviews/coreai-iphone17pro-device-probe/README.md).
No model was loaded or run. This workspace is Linux with no Apple SDK, so that
run is user-supplied evidence, not an independent reproduction here.
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

## Uji model numerik berikutnya

Setelah pull, buka project yang sama, **Run**, lalu tekan **Uji model numerik**.
Bagikan seluruh laporan, termasuk jika gagal memuat model atau compile.

Model kandidat `ModelCandidates/affine.mlpackage` menghitung `y = 2*x + 0.25`:
input `[0, 0.25, -0.5, 1]` menghasilkan referensi `[0.25, 0.75, -0.75, 2.25]`.
Paket dan referensi berukuran sekitar 2 KB. Kandidat diekspor dengan
**Core ML Tools 9.0** sebagai ML Program dengan minimum target iOS 18. Ini
**uji kompatibilitas format lama**, bukan klaim bahwa format itu sudah didukung
CoreAI atau jalur konversi ACE-Step sudah bekerja. Aplikasi tetap `import CoreAI`
dan memanggil `AIModel`/`InferenceFunction.run`; tidak ada runtime fallback.

Folder kandidat disalin sebagai resource biasa. Xcode tidak diminta melakukan
Core ML code generation atau membuat wrapper `MLModel`. Pemeriksaan sumber
`AIModelAsset.isValid` dicatat, lalu loader CoreAI diberi kesempatan menerima atau
menolak sumber tersebut. Loader tidak diganti jika gagal. Pemanggil model generik
tetap memvalidasi aset secara default; hanya eksperimen kompatibilitas ini yang
meneruskan sumber langsung ke loader setelah mencatat hasil validasi.

Uji meminta kebijakan `.cpuOnly`, memeriksa parity dalam tiga run, dan mencatat
waktu pemuatan/specialization terpisah dari run pertama serta dua run hangat.
Waktu run termasuk membaca output. Ini bukan pengukuran peak RAM atau bukti
penempatan akselerator; model kecil ini juga tidak mewakili biaya generasi musik.
Numerical probe baru belum dikompilasi/dijalankan dengan SDK Apple di workspace
Linux ini. Hasil tensor awal pengguna hanya berlaku pada versi probe sebelumnya.

Developer dapat mengekspor ulang ke **folder baru** dengan:

```bash
python3 -m venv /tmp/coreai-export-env
/tmp/coreai-export-env/bin/python -m pip install 'coremltools==9.0' 'numpy<2.3'
/tmp/coreai-export-env/bin/python scripts/export_coreai_candidate.py --out /tmp/coreai-candidate
```

Pengguna aplikasi tidak perlu memasang exporter atau mengunduh model.

Before audio tests, establish the official export tool and create a tiny known
model asset with reference inputs/outputs. Load it on device, run the function,
compare values to the reference, and measure load/specialization and warmed
inference separately. Then attempt one audio component and a short one-instrument
trial. End-to-end ACE-Step compatibility, quantization, operator support, peak
RAM, generation latency and thermal behaviour remain unverified.

API evidence and limitations: [SDK review](../../docs/coreai-sdk-probe.md).

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
