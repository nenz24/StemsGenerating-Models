# StemsGenerating-Models

POC lokal di JupyterLab: rekaman gitar/piano → bass atau drum pengiring memakai
model pretrained ACE-Step. Mulai dari evaluasi kualitas, kemudian putuskan apakah
fine-tuning diperlukan. Belum ada model baru yang dilatih atau konversi iPhone.

## Mulai di komputer JupyterLab

```bash
git clone https://github.com/nenz24/StemsGenerating-Models.git
cd StemsGenerating-Models
jupyter lab
```

Buka **[notebooks/01_quality_poc.ipynb](notebooks/01_quality_poc.ipynb)** dan jalankan
sel dari atas. Jika sudah mempunyai checkout, jalankan `git pull --ff-only`
setelah menyimpan perubahan lokal. Jika JupyterLab belum terpasang, instal
`requirements-notebook.txt` di environment notebook sendiri.

1. Jalankan diagnostik. Tidak perlu tahu GPU sebelumnya: lihat `reports/hardware.json`.
2. Run All pertama menggunakan fixture 10 detik, tanpa instalasi ML, unduhan bobot,
   atau inferensi. Perlu `git`, `ffmpeg`, `ffprobe`, dan Python 3.11+.
3. Ganti `SOURCE` dengan rekaman gitar/piano sendiri, pilih bass atau drums dan
   seed 0/1/2. Bisa memakai ekspor WAV+JSON SongMap, atau audio biasa.
4. Instal runtime terpisah, unduh model terpilih, lalu aktifkan `GENERATE` lewat
   sakelar notebook. Sel-sel itu hanya berjalan saat kamu mengaktifkannya.
5. Dengarkan stem/mix dan bandingkan Base dengan XL pada sumber/seed sama.

## Mac Apple Silicon / Jupyter Homebrew

Jika diagnostik menunjukkan `Darwin / arm64`, jalankan di Terminal Mac:

```bash
brew install ffmpeg
cd /path/to/StemsGenerating-Models
git pull --ff-only
```

Simpan salinan notebook yang sudah diedit sebelum pull jika Git melaporkan
perubahan lokal. Restart kernel dan buka ulang notebook versi terbaru. Helper
mencari `ffmpeg`/`ffprobe` di PATH serta `/opt/homebrew/bin` dan `/usr/local/bin`.
Bila belum terpasang, notebook berhenti dengan petunjuk instalasi sebelum
persiapan audio, tanpa traceback subprocess yang panjang.

Python 3.15 bawaan JupyterLab tidak perlu diganti. Sel `INSTALL_RUNTIME` memakai
`uv` untuk menyiapkan **Python 3.11 terpisah** bagi dependency model. `DEVICE="auto"`
memilih MPS dari Python model pada Mac, CUDA pada NVIDIA, dan menolak fallback
CPU diam-diam. Diagnostik pada kernel yang belum memiliki PyTorch belum bisa
memastikan MPS; sel instalasi memeriksanya lagi di environment model.

Pada Mac mulai dengan `MODEL="acestep-v15-base"`, `SEEDS=[0]`, `OFFLOAD=True`.
Lihat RAM terpadu dalam laporan diagnostik dan ukur satu generasi sebelum XL.
Ruang disk yang besar tidak memastikan model muat di RAM. Jalur MPS memakai
PyTorch yang didukung API upstream; MLX DiT belum diaktifkan di adapter ini.
Metrik MPS mencatat **current** allocator/driver memory, bukan peak atau total
memori proses. Keberhasilan Mac tidak membuktikan Core AI/iPhone.

## Kandidat kualitas

- **XL Base:** dokumentasi upstream menempatkannya sebagai opsi kualitas lebih
  tinggi; gunakan sebagai kandidat utama jika resource mencukupi. Rekomendasi
  notebook: GPU NVIDIA dengan kelas 24 GB (laporan driver ≥24000 MiB VRAM). Ini belum pengukuran hardwaremu.
- **Base:** baseline pembanding yang lebih ringan. CPU offload tersedia.
- Keduanya: mode `lego`, 50 langkah, CFG 7, satu instrumen per job, planner mati.
  Angka kecepatan Turbo tidak dipakai untuk menaksir waktu lego.

Bobot yang benar-benar diunduh menurut metadata resmi: sekitar **14.88 GB untuk
Base** atau **30.05 GB untuk XL** beserta komponen utama upstream. Sediakan
headroom disk setidaknya dua kali ukuran unduhan ditambah 5 GiB. Komponen Turbo
serta LM ikut disediakan untuk memenuhi pemeriksaan initializer upstream; LM
planner tidak diaktifkan. Runtime/dependency ML memerlukan ruang tambahan.

Model pretrained dipin pada revisi HF dan source Git tertentu. Semua aset dicek
terhadap hash penerbit sebelum dipakai. Runtime memakai salinan kode model agar
sinkronisasi kode oleh upstream tidak mengubah aset unduhan asli.

## Bridge GitHub ↔ lokal

GitHub menyinkronkan **kode, notebook tanpa output, dan feedback kecil**. Ini tidak
memberi agen akses langsung ke filesystem lokal. Jangan push bobot, rekaman,
cache, `.env`, atau keluaran audio; semuanya sudah diabaikan oleh `.gitignore`.
Simpan input di `data/`, hasil di `runs/`. Tinjau perubahan sebelum push.

Untuk memberi informasi hardware, cukup isi berkas di `feedback/` berdasarkan
`reports/hardware.json`, commit dan push. Contoh format tersedia di
[feedback/README.md](feedback/README.md). Setelah ada feedback atau traceback,
runner bisa disesuaikan dengan hardware JupyterLab yang sebenarnya.

## Status validasi

25 pengujian audio/CLI plan, helper, serta kontrak MPS/CUDA lulus di Linux tanpa GPU. Seluruh 9
sel kode notebook default juga dieksekusi tanpa error, tanpa mengunduh bobot
atau menghasilkan stem model. Checkout source upstream pada commit yang dipin
juga berhasil diverifikasi.
Pemilihan MPS dan pemanggilan API diuji dengan mock. Inferensi MPS/CUDA nyata,
instalasi dependency ML penuh, dan kualitas model pada rekamanmu belum diverifikasi. Default notebook tidak mengklaim keberhasilan generasi.
Tidak ada gerbang review lisensi untuk POC sesuai permintaan; catatan singkat ada
di [docs/model-notes.md](docs/model-notes.md).

Pengujian:

```bash
python3 -m unittest discover -s Tests/StemEvaluationPython -v
```

STAGE tetap tersedia sebagai adapter riset CLI dari pekerjaan sebelumnya;
notebook pertama berfokus pada ACE-Step. Konversi Core AI, benchmark iPhone,
training/fine-tuning, dan sinkronisasi hasil otomatis belum diimplementasikan.
