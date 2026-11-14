# Review audio lagu penuh yang sudah selesai

Status eksekusi `COMPLETED` dan sampled memory `PASS` tidak membuktikan mutu
musik. Window dibuat dengan trajectory masing-masing. Context/noise global
yang konsisten dan normalized overlap-add belum menjamin akor, timing, fase
atau gaya yang konsisten di antara window. Bunyi yang memburuk sepanjang lagu
perlu diperiksa pada raw window dan hasil perakitannya sebelum mengubah model.
Thermal state akhir juga bukan diagnosis penyebab glitch atau harmoni salah.

Jangan ulang33 generation untuk mengirim bukti audio. Pilih hasil yang sudah
ada. Helper berikut memakai stdlib Python, tanpa Torch, inference atau export.
Baseline BF16, FP32 APG/VAE,50 steps dan toleransi lama tidak diubah.

## Jika hanya WAV yang tersedia

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
python3 scripts/package_coreai_full_song_quality_review.py --choose-audio
```

Pilih **piano-with-band.wav** yang sama dengan hasil yang bermasalah. Helper
menulis tiga cuplikan8s dari awal, tengah dan akhir ke satu ZIP kecil, lalu
menunjukkannya di Finder. Kirim ZIP tersebut. Metadata mencatat posisi sampel
sebenarnya dan SHA file sumber. Audio dibaca sebagai Float32/PCM16 stereo48kHz
dan ditulis Float32 tanpa perubahan gain, tempo, filter atau perbaikan glitch.
Input Float32 mempertahankan nilai sampel persis; PCM16 didekode ke Float32.

Tanpa dialog, gunakan `--audio` dengan path file sebenarnya. Tidak ada path
placeholder yang perlu disalin. File ZIP default memakai versi berikutnya
yang belum ada, sehingga review terdahulu tetap tersedia. `--out` eksplisit
harus menunjuk file baru.

## Jika container lengkap sudah diunduh

```bash
python3 scripts/package_coreai_full_song_quality_review.py --choose-container
```

Pilih folder hasil download container **CoreAI Probe**. Helper mencari
`Documents/FullSongGeneration` atau struktur `.xcappdata/AppData/Documents`
hingga8 tingkat, termasuk jika memilih Downloads. Tidak perlu build atau
menekan tombol generasi lagi. `--source` dapat memilih folder tersebut secara
langsung; `--execution-sha` membatasi pencarian pada run yang ditunjuk laporan.
Beberapa run yang berbeda harus dipilih eksplisit; helper tidak mengganti run.

Paket container memuat:

- Cuplikan3s piano, bass, drum, gitar dan band pada awal/tengah/akhir lagu.
- Slice raw dua atau lebih window yang bertemu di overlap akhir, serta slice
  hasil perakitan pada waktu global yang sama, tanpa gain atau DSP.
- Final latent dan laporan pada window awal, tengah, akhir dan window pasangan
  overlap. Metadata mengikat masing-masing pada checkpoint/report asli.
- Pemeriksaan ulang OLA Float64 pada range tersebut dengan batas lama
  abs1e-4/rel1e-3/nRMSE1e-3, dan binding raw ke WAV dengan gain Float32 asli pada
  tiga range per instrumen. Kegagalan numerik dicatat, bukti tetap diekspor.

WAV dan raw stem lengkap diverifikasi terhadap receipt, sedangkan raw window
yang dipilih diverifikasi terhadap checkpoint. File raw window lainnya tidak
diekspor atau diklaim telah diuji waveform-nya. Tidak ada klaim strict parity
trajectory atau penerimaan musik dari tes transport/OLA. Data audio-only tidak
memiliki bukti raw/latent/container dan dicatat demikian.

Ukuran ZIP dibatasi32MiB sebelum diterbitkan. Output tersimpan di luar
container; source, hasil lama dan model tidak ditulis ulang. SHA yang tidak
cocok menghentikan ekspor, tanpa inference ulang atau cache repair. ZIP, audio,
tensor dan laporan perangkat tetap privat dan tidak dimasukkan ke Git.
