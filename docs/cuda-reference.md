# Referensi CUDA untuk B yang glitch

Tujuan pertama adalah satu hasil CUDA yang dapat dibandingkan dengan baseline
MPS. CFG 1 sudah dicoba di Mac dan lebih buruk. Sweep seed 0–4/CFG 3,5,7,
metadata SongMap, input native/48 kHz dan konteks panjang menunggu hasil ini.

## Ekspor di Mac

```bash
python3 scripts/cuda_reference.py --export
```

Exporter mengambil pembanding decoder terbaru yang selesai. Pilih run eksplisit
jika perlu:

```bash
python3 scripts/cuda_reference.py --export --run /path/to/decoder_comparison_xxx
```

Baseline harus Base, CFG 7, seed/profil konsisten, 50 langkah dan B 30 detik.
Input/hasil yang berubah hash atau konfigurasi baseline berbeda ditolak. Exporter
memakai `git archive HEAD`, sehingga source yang dipaketkan adalah commit yang
sudah di-pull, bukan perubahan lokal yang belum di-commit. Paket menyertakan
kode/notebook agar server tidak memerlukan akses GitHub privat untuk bridge ini.
Source ACE upstream tetap di-clone pada SHA pinned ketika memasang runtime.

ZIP tidak berisi bobot. Unduhannya di server baru sekitar 13.86 GiB untuk aset
model, di luar paket runtime/CUDA. Cache/temp dan salinan aset memerlukan ruang
lebih dari ukuran unduhan. Notebook mencetak estimasi disk sebelum pemasangan.

## Google Colab tanpa server pribadi

Setelah pull dan ekspor di Mac, buka [Google Colab](https://colab.research.google.com/).
Pilih **File → Upload notebook**, lalu pilih `notebooks/09_colab_reference.ipynb`
dari checkout Mac. Pilih **Runtime → Change runtime type → GPU**, misalnya T4
jika tersedia. GPU mengikuti ketersediaan akun; notebook tidak membeli paket GPU.

Run all dan upload **satu** `cuda_reference_*.zip` hasil exporter ketika sel pertama
meminta file. Tidak perlu clone repo privat, token GitHub, terminal, atau mount
Google Drive. Notebook memeriksa GPU sebelum instalasi besar, memverifikasi isi/
hash paket dan menentukan folder source dari hasil ekstraksi, bukan dari asumsi
current working directory `/content` atau folder `notebooks`.

Jika ffmpeg/ffprobe belum ada, notebook memasangnya pada VM Colab. Model tetap
memakai runtime Python 3.11 terisolasi; PyTorch bawaan kernel tidak diubah.
Bootstrap memakai `pip` kernel untuk memasang `uv==0.10.0` ke virtualenv kosong
dengan `pip --python`, tanpa bergantung pada `ensurepip` Python bawaan Colab.
Sel setup selalu menjalankan `uv sync --frozen` agar instalasi yang terputus
dapat dilanjutkan; paket yang sudah sesuai dipakai ulang.
Upload paket yang sama pada VM yang sama memakai folder/runtime/cache yang sama,
sementara hasil inferensi berada di folder run baru. Source yang berubah di
folder hasil ekstraksi ditolak, bukan ditimpa. `BUNDLE_ZIP` dapat diisi dengan
path ZIP yang sudah diupload untuk melewati tombol upload saat mengulang sel.

Sel terakhir menawarkan download `cuda_result.zip`. Sesi Colab sementara:
download hasil sebelum sesi berakhir. VM baru perlu menyediakan bobot lagi;
notebook tidak otomatis menghubungkan Google Drive untuk menyimpannya.
Untuk pemeriksaan paket/plan saja, `DRY_RUN = True` melewati instalasi, unduhan,
inferensi dan download browser. Mode ini bukan hasil GPU nyata.

### Memulihkan sesi lama yang gagal pada `ensurepip`

Jika paket lama gagal saat membuat `.bootstrap` dengan Python 3.13 Colab,
tambahkan dan jalankan sel ini di bawah sel yang gagal. Gunakan sesi yang sama
agar variabel `ROOT` dan helper notebook masih tersedia:

```python
bootstrap_python = ROOT / ".bootstrap/bin/python"
run_logged([
    sys.executable, "-m", "pip", "--python", str(bootstrap_python),
    "install", "pip", "uv==0.10.0",
], env=runtime_environment(), check=True)
run_logged([
    sys.executable, str(ROOT / "scripts/setup_runtime.py"), "--install",
], env=runtime_environment(), check=True)
print(json.dumps(check_cuda(runtime_python()), indent=2))
```

Ini memasang `pip` dan `uv` hanya ke bootstrap, sehingga installer lama dapat
melanjutkan tanpa `ensurepip`. Source paket, input B dan parameter inferensi tidak
diubah. Setelah setup berhasil, lanjutkan sel unduh/verifikasi bobot; tidak perlu
upload atau ekspor ulang untuk sesi ini. Untuk sesi baru, pull perbaikan di Mac,
ekspor ZIP baru dan upload notebook/ZIP yang baru.

## Server Linux CUDA milik sendiri (alternatif)

Upload ZIP, lalu ekstrak di Terminal ke folder baru (jangan menimpa percobaan lain):

```bash
python3 -m zipfile -e cuda_reference_xxx.zip cuda-reference
```

Buka `cuda-reference/source/notebooks/08_cuda_reference.ipynb` dan Run All.
Notebook memverifikasi hash paket sebelum setup, memeriksa `nvidia-smi`, memasang
runtime pinned dengan sync yang dapat diulang, dan memeriksa `torch.cuda.is_available()` di
runtime tersebut sebelum mengunduh bobot. Wheel Linux x86_64 memakai PyTorch
2.10.0+cu128; driver/GPU server harus kompatibel. Tidak ada instalasi paket sistem
atau penyewaan server otomatis. Sediakan ffmpeg/ffprobe melalui pengelola OS
jika belum ada. Kernel Jupyter tidak harus memiliki PyTorch yang cocok.

Aset yang sudah tersedia pada
`source/vendor/ACE-Step-1.5/checkpoints/` diverifikasi dan dipakai ulang. Tidak
menghapus cache atau bobot lama. CUDA yang tidak tersedia, OOM, ketidakcocokan
parameter, dan output tidak valid tetap menghasilkan error, bukan klaim sukses.

Jika server sudah mempunyai checkout bridge/runtime/aset yang cocok, dapat
memakai direktori `reference/` dari paket dengan CLI:

```bash
python3 scripts/cuda_reference.py --reference /path/to/unpacked/reference --out runs/cuda_plan
python3 scripts/cuda_reference.py --reference /path/to/unpacked/reference --out runs/cuda_real --execute
```

Tanpa `--execute`, hanya plan dibuat. Untuk instalasi/aset lihat sel notebook.
Gunakan direktori output baru; runner tidak menimpa hasil percobaan sebelumnya.

## Apa yang dipertahankan

Audio B disalin utuh, tanpa gain kedua atau resampling baru. Kondisi audio-only,
caption/instruction, seed, Base, CFG 7, 50 langkah, dan panjang 30 detik sama.
Paket mencatat revisi bridge/upstream serta manifest dan hash input/baseline.
Runner menolak perubahan parameter generasi di luar lokasi file `src_audio`.

Perubahan platform juga mengubah dtype VAE/DiT dan stream RNG. Dtype aktual dan
panggilan decoder dicatat dalam capture. Seed sama tidak berarti tensor noise
CUDA dan MPS identik. Hasil bersih/lebih buruk pada satu device adalah petunjuk,
bukan bukti tunggal bug platform; pengujian lanjutan harus menjaga faktor ini.

BPM/key/time signature belum ditambahkan pada baseline. Menambahkannya sekarang
akan membuat hasil tidak dapat dibandingkan sebagai pembanding device saja.
Tidak ada klaim bahwa metadata yang kosong pasti menyebabkan glitch.

## Hasil

Bandingkan `mps_baseline_listen.wav` dan `cuda_listen.wav` dengan gain 0.7 pada
perangkat/volume yang sama. Output `cuda_reference.json`, `session.json`, result
job dan capture merekam hardware, versi, parameter, dtype dan hash hasil.
`cuda_result.zip` hanya menyertakan report, dua audio, dan latent capture kecil;
bukan folder runtime/checkpoint yang besar. Jika sel generasi gagal, report
terakhir tetap ada; sel ekspor dapat dijalankan sendiri untuk mengemas artefak
yang tersedia.

Jika CUDA bersih, selidiki perbedaan runtime/precision/seed sebelum menjalankan
lagu penuh. Jika CUDA juga rusak, evaluasi konfigurasi/input dan kesesuaian model;
jangan menganggap ini sudah membuktikan bug MPS. Sweep atau input 60–180 detik
baru masuk sesudah satu hasil CUDA ini diperiksa.
