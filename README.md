# StemsGenerating-Models

POC JupyterLab untuk membuat bass/drum pengiring dari audio memakai pretrained
ACE-Step 1.5 Base. Model belum dilatih ulang atau dikonversi ke Core AI/iPhone.

**Status kualitas:** bass pada bagian B masih glitch di Mac/MPS. Rekonstruksi
piano melalui encoder CPU/MLX bersih, decoder CPU/MLX menghasilkan output hampir
identik, dan CFG 1 justru lebih buruk. Belum ada perbaikan kualitas yang terbukti.
Langkah aktif adalah **satu pembanding CUDA dengan B dan CFG 7 yang sama**.

## Satu uji di JupyterLab CUDA

Di checkout Mac yang menyimpan hasil percobaan sebelumnya:

```bash
git pull --rebase --autostash
python3 scripts/cuda_reference.py --export
```

Perintah mencetak lokasi `cuda_reference_*.zip`. Paket berisi input B 30 detik
byte-identik, baseline MPS, setting/hash, dan snapshot kode commit Git saat ini.
Tidak menjalankan model lagi di Mac. Bobot, kredensial, run lain dan output
notebook lokal yang belum di-commit tidak disertakan.

1. Upload ZIP itu ke server JupyterLab **Linux dengan GPU NVIDIA**.
2. Di Terminal server, ekstrak ke folder baru:

   ```bash
   python3 -m zipfile -e cuda_reference_xxx.zip cuda-reference
   ```

3. Buka **`cuda-reference/source/notebooks/08_cuda_reference.ipynb`**, lalu Run All.
4. Bandingkan `mps_baseline_listen.wav` dan `cuda_listen.wav`; paket `cuda_result.zip`
   yang dicetak berisi audio/report tanpa bobot model.

Notebook memeriksa GPU/disk, menyiapkan runtime Python 3.11 terisolasi, menyediakan
aset pinned, dan menjalankan **satu** generasi: Base, seed 0, CFG 7, 50 langkah,
30 detik. Bukan sweep seed/CFG dan bukan generasi seluruh lagu. Jika CUDA tidak
tersedia di runtime model, proses gagal dengan jelas; tidak beralih ke CPU.

Server baru memerlukan ruang sekitar 45–50 GiB untuk runtime, bobot dan cache,
serta ffmpeg/ffprobe. Bobot lokal yang valid dipakai ulang. Dtype CUDA dipilih
upstream sesuai GPU dan dicatat; RNG/dtype CUDA dapat berbeda dari MPS. Satu hasil
CUDA saja belum membuktikan bug MPS. CUDA nyata belum dijalankan di chat ini,
karena environment chat tidak mempunyai GPU NVIDIA.

Lihat [panduan CUDA](docs/cuda-reference.md) untuk rincian dan penggunaan CLI.

## Batas POC saat ini

- ACE-Step masih **audio-only conditioning**. BPM/key/time signature dari SongMap
  belum dikirim sebagai parameter model; chord/beat disimpan untuk evaluasi.
  Click B/C tersedia pada adapter STAGE. Hipotesis manfaat metadata SongMap belum
  diuji pada jalur ACE.
- Input pembanding tetap 32 kHz PCM16 hasil persiapan sebelumnya. Versi native/
  48 kHz harus diuji terpisah setelah baseline, bukan dicampur ke perubahan device.
- Batas 10/30 detik adalah batas bridge saat ini, bukan bukti batas arsitektur
  ACE-Step. Inferensi 60–180 detik memerlukan uji memori/kualitas tersendiri.
- Chunk independen dapat berbeda karakter/groove. Crossfade tidak memperbaiki
  audio yang sudah rusak sebelum stitching. Mode lagu penuh masih eksperimen.
- Ini belum model generatif on-device siap iPhone. Feature/model integration,
  kualitas musik dan ukuran model akhir masih perlu diselesaikan.

## Notebook dan catatan lain

- [01_quality_poc](notebooks/01_quality_poc.ipynb): hardware dan plan aman.
- [02_mac_24gb_full](notebooks/02_mac_24gb_full.ipynb): setup/inferensi Mac lokal.
- Notebook 03–07: percobaan lagu penuh dan diagnosis lama; lihat
  [arsip diagnosis](docs/diagnostics-history.md) sebelum mengulangnya.
- [Catatan model](docs/model-notes.md) dan [format feedback](feedback/README.md).

## Validasi kode

```bash
python3 -m unittest discover -s Tests/StemEvaluationPython -v
```

Tes menilai audio/timeline, hash/cache, CLI plan, kontrak adapter, dan simulasi
kegagalan tanpa bobot model. Hasil tes kode tidak membuktikan mutu bass atau
keberhasilan CUDA nyata. Notebook CUDA diuji dalam mode dry-run dengan fixture
sintetis; hasil inferensi CUDA harus berasal dari server GPU pengguna.
