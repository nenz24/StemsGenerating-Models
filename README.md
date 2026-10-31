# StemsGenerating-Models

POC lokal di JupyterLab: rekaman gitar/piano → bass atau drum pengiring memakai
model pretrained ACE-Step. Mulai dari evaluasi kualitas, kemudian putuskan apakah
fine-tuning diperlukan. Belum ada model baru yang dilatih atau konversi iPhone.

## Glitch setelah sekitar detik 10 pada mode chunking

Ada laporan hasil nyata dengan glitch berat yang berlanjut setelah detik 10.
Mode notebook 03 masih eksperimen; tes timeline numerik bukan bukti kualitas
musikal. Jangan mengulang seluruh generasi hanya untuk mencoba crossfade lain.

Buka **[notebooks/04_seam_diagnostics.ipynb](notebooks/04_seam_diagnostics.ipynb)**
untuk membaca cache run yang sudah ada dan mendengarkan chunk terpisah, raw model,
bass gabungan, serta mix. Tidak ada model dimuat, unduhan atau inferensi baru.
Cuplikan dan report disimpan di subfolder diagnostik baru; hasil lama dipertahankan.
Pastikan folder RUN yang tercetak adalah run dengan suara bermasalah.

Chunk kedua biasanya mewakili detik lagu 9–19. Crossfade pertama hanya 9–10;
bagian 10–18 seharusnya identik dengan interior chunk kedua. Diagnostik memeriksa
kesamaan sampel tersebut. Bila glitch sudah ada pada chunk kedua sebelum stitching,
perbaikan harus menyasar generasi/konversi, bukan menebak pengaturan crossfade.
Bila raw model bersih tetapi stem kanonis rusak, periksa konversi/clipping. Bila
chunk kanonis bersih tetapi gabungan rusak, periksa stitching/fase/timing. Statistik
peak/RMS/full-scale tidak otomatis menentukan kualitas atau penyebab glitch.

Untuk menelusuri lanjut, bagikan report dan/atau cuplikan bass sekitar detik 7–22
serta `chunk_001_before_stitch.wav` jika bersedia. Tidak perlu seluruh audio asli.
Belum ada klaim bahwa glitch ini sudah diperbaiki tanpa pemeriksaan audio nyata.

## Bass untuk lagu penuh sekitar 3 menit

Jika notebook 02 sudah berhasil menghasilkan bass, gunakan
**[notebooks/03_mac_full_song.ipynb](notebooks/03_mac_full_song.ipynb)** untuk
seluruh rekaman. Notebook 02 mengambil klip 10 detik; notebook 03 memakai semua
audio hingga batas 10 menit POC.

1. Pull kode terbaru.
2. Taruh satu file rekaman sendiri di **`data/full_song/`**. Folder ini dipisah
   dari input notebook 02 agar rekaman uji lama tidak terpilih. WAV/MP3/M4A didukung.
3. Buka notebook 03, lalu Restart Kernel → Run All.
4. Hasil berada di folder run yang dicetak notebook: **`bass_full.wav`** dan
   **`mix_full.wav`**, dengan panjang sama seperti audio yang didecode.

Base/MPS/seed 0/CPU offload sudah diatur. Runtime dan bobot terverifikasi yang
sudah tersedia dipakai kembali; notebook ini tidak memasang atau mengunduh model.
Untuk rekaman tepat 180 detik dibuat 20 potongan 10 detik dengan overlap nominal
1 detik (potongan akhir mengikuti akhir lagu). Model dimuat sekali per proses.
Runtime total dapat mendekati banyak kali waktu generasi satu klip, tergantung
hardware, warm-up dan isi audio; belum ada benchmark 3 menit pada Mac pengguna.

Sumber/profil/manifest model yang sama memakai folder run yang sama. Chunk selesai
dipakai ulang setelah dicek hash; chunk gagal dicoba dalam folder attempt baru.
Hasil/cache yang berubah ditolak, bukan dipakai diam-diam. Jangan menjalankan dua
generasi bersamaan untuk run sama; lock OS melindungi folder itu. Hasil final yang
sudah selesai diverifikasi dan dipakai ulang. Run yang gagal tetap menyimpan
metadata dan attempt-nya. Mengubah seed/instrument menghasilkan run terpisah.

**Batas musikal:** setiap chunk digenerasi mandiri, bukan inferensi dengan konteks
lagu 3 menit sekaligus. Crossfade linear menjaga kontinuitas audio dan panjang
sampel, tetapi pola bass/groove, artikulasi dan fase dapat berbeda atau melemah
pada sambungan. Dengarkan harmoni dan transisi di sekitar 9,18,27 detik dan
seterusnya. Tidak ada jaminan lagu penuh mengikuti feel secara konsisten.

Pengujian menyiapkan audio sintetis 180 detik dan membuktikan round-trip timeline
20 chunk tanpa perubahan sampel untuk stem identitas. Runner juga diuji dengan
model palsu yang gagal pada chunk kedua lalu melanjutkan tanpa meregenerasi chunk
pertama. Tes tersebut tidak memvalidasi kualitas bass maupun inferensi MPS panjang.

## Langsung jalan untuk Mac Apple Silicon 24 GB

Buka **[notebooks/02_mac_24gb_full.ipynb](notebooks/02_mac_24gb_full.ipynb)**.
Profil di `configs/mac-24gb.json` sudah berisi Base, MPS, bass, seed 0 dan CPU
offload. Memerlukan macOS 14 atau lebih baru untuk dependency MLX pada lockfile
upstream. Tidak perlu mengubah tiga flag instalasi/unduhan/generasi notebook lama.

1. Pull kode terbaru. Jika notebook lama berubah, simpan lewat Git stash dahulu.
   Notebook 02 adalah file baru, sehingga notebook 01 beserta konfigurasi lokalnya
   dapat tetap disimpan dalam stash.
2. Letakkan satu rekaman gitar/piano minimal 10 detik di `data/`. Folder ini
   dibuat oleh sel pertama; bisa dibuat sendiri di file browser JupyterLab.
   Satu file WAV/MP3/M4A dipilih otomatis, atau gunakan WAV bernama `input.wav`.
3. Jalankan **Restart Kernel → Run All** pada notebook 02.
4. Notebook memeriksa sumber dan tool, mempersiapkan klip, menginstal runtime
   Python 3.11 terpisah, memverifikasi MPS, mengunduh/verifikasi aset sekitar
   14.88 GB, lalu mencoba generasi dan menampilkan stem/mix.

Runtime dan bobot yang sudah ada dipakai kembali setelah diperiksa. Instalasi
pertama memerlukan jaringan dan ruang tambahan; bukan proses yang sudah selesai
hanya karena notebook berhasil dibuka. Tidak ada audio sintetis sebagai pengganti
hasil model. Sumber yang belum tersedia menghentikan notebook sebelum instalasi
atau unduhan. Parameter SOURCE/START/CONDITIONING tetap bisa diubah jika diperlukan.

`DRY_RUN=False` adalah default penuh. Mode True hanya untuk validasi pipeline dan
selalu menandai instalasi, unduhan dan inferensi sebagai tidak dijalankan.
Notebook ini dipersiapkan untuk Mac-mu; inferensi MPS nyata tetap belum diverifikasi
oleh agen Linux. Kirim traceback jika instalasi atau inferensi gagal.

## Mulai di komputer JupyterLab

```bash
git clone https://github.com/nenz24/StemsGenerating-Models.git
cd StemsGenerating-Models
jupyter lab
```

Untuk diagnostik aman/lintas platform, buka **[notebooks/01_quality_poc.ipynb](notebooks/01_quality_poc.ipynb)** dan jalankan
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

## Melihat progress di JupyterLab

Notebook meneruskan stdout/stderr subprocess melalui output Python kernel.
Unduhan menampilkan nomor file, nama dan ukuran. Progress transfer HF ditampilkan
jika client memberikannya; baris checksum menunjukkan byte file yang benar-benar
dibaca dan diverifikasi. Tahap menyalin dan total aset yang selesai diverifikasi
juga dicatat. Total aset terverifikasi bukan jumlah byte jaringan yang diunduh.

Jika child tidak menulis output selama 15 detik, heartbeat menunjukkan proses
masih hidup dan durasi tanpa output. Itu tidak membuktikan jaringan bergerak dan
bukan persentase unduhan. Exit nonzero tetap menjadi error notebook.

Untuk mengambil pembaruan saat notebook sudah dijalankan, simpan perubahan lokal
ke stash jika pull terhalang. Tunggu sel unduhan aktif selesai sebelum membuka
ulang notebook yang baru dan memulai proses berikutnya. File lengkap yang lolos
checksum dipakai kembali. Pemulihan transfer parsial mengikuti cache HF.

## Error matplotlib_inline saat mulai generasi

Jika semua 34 aset sudah terverifikasi tetapi impor backend gagal dengan
`MPLBACKEND=module://matplotlib_inline.backend_inline`, pull perbaikan terbaru.
Subprocess model sekarang memakai backend standar `Agg` dan cache konfigurasi
lokal `.runtime-cache/matplotlib`, sebelum dependency Lightning/TorchMetrics
mengimpor Matplotlib. Kernel Jupyter dan paketnya tetap terpisah.

Lakukan Restart Kernel → Run All pada notebook 02 untuk membuat folder run baru.
Folder hasil gagal dipertahankan untuk diagnosis; bobot lengkap yang sudah ada
akan diverifikasi/dipakai kembali, tanpa diunduh ulang. Pengujian mengulang
kegagalan impor dengan Matplotlib 3.10.8 dalam environment terpisah dan membuktikan
impor serta rendering PNG berhasil dengan Agg. Ini belum membuktikan seluruh
inferensi MPS pada Mac berhasil.

## Jika file Base tertahan berjam-jam

Nomor [32/34] dengan `acestep-v15-base/model.safetensors` berarti transfer bobot
Base sekitar 4.46 GiB belum selesai. Itu berbeda dari inferensi MPS. Heartbeat
menandakan proses hidup saja; tanpa log lain belum dapat dibedakan jaringan,
transfer Xet, atau penantian lock cache.

Downloader sekarang memakai jalur **HTTP resmi HF** secara default, dengan Xet
dinonaktifkan pada subprocess download, timeout metadata 20 detik dan HTTP read
60 detik. Timeout tersebut bukan batas waktu keseluruhan transfer. TLS dan
checksum tetap aktif. Cache HTTP terpisah menghindari lock/partial Xet yang
tertahan tanpa menghapusnya. File target yang sudah diverifikasi digunakan lagi;
file Base yang belum selesai dapat mulai diunduh dari awal pada cache HTTP baru.

Hentikan sel unduhan lama memakai **Interrupt Kernel** sebelum menjalankan
percobaan baru. Pull kode terbaru, lalu jalankan ulang sel unduhan notebook 02.
Perubahan default transport ada di script sehingga tidak perlu mengubah notebook.
Pilihan Xet tetap tersedia lewat `--transport xet` untuk eksperimen terpisah.

Untuk bukti tambahan, dari Terminal di repository jalankan:

```bash
python3 scripts/diagnose_download.py --interval 10
```

Ini hanya membaca metadata target/cache selama 10 detik dan memeriksa proses
yang membuka file lock bila `lsof` tersedia. Tidak membaca token/log autentikasi,
menghapus cache, atau menghentikan proses. Ukuran/mtime/alokasi disk bukan
pengukuran byte jaringan; lock yang ada belum berarti sedang dipegang. Kirim
outputnya jika retry masih tertahan. Jangan menghapus cache atau lock secara massal.

## Perbaikan permission uv / folder home

Jika instalasi lama gagal pada `~/.local/share/uv/python`, cukup pull kode terbaru
lalu jalankan ulang sel instalasi notebook 02. Checkout source dan `.bootstrap`
yang sudah ada dipakai kembali; tidak perlu sudo, chown, atau reinstall Jupyter.

Python unduhan uv sekarang disimpan di `.runtime-cache/uv/python`, cache uv/pip
juga di `.runtime-cache/`, dan environment model tetap di
`vendor/ACE-Step-1.5/.venv`. Cache unduhan model Hugging Face/Xet diarahkan ke
`.runtime-cache/huggingface/`. Folder cache diabaikan Git. HOME, token, proxy dan
CA trust yang diwariskan tidak diganti; checksum/TLS tetap diperiksa.

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

52 pengujian audio/CLI plan, helper, profil lengkap Mac, cache lokal, progress dan kontrak MPS/CUDA lulus di Linux tanpa GPU. Seluruh 9
sel kode notebook 01 juga dieksekusi tanpa error, tanpa mengunduh bobot
atau menghasilkan stem model. Ketujuh sel kode notebook 02 diuji dalam mode
dry-run yang ditandai jelas dengan persiapan audio nyata. Rencana dependency
macOS 14 arm64/Python 3.11 berhasil diperiksa memakai uv 0.10.0 dan lockfile
upstream: 125 paket dapat direncanakan, tanpa memasang dependency ML tersebut. Checkout source upstream pada commit yang dipin
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
