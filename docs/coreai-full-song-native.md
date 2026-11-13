# Generasi lagu penuh dari input tersimpan

Runner diagnostik ini meneruskan job input global yang sudah disiapkan di Mac.
DiT BF16 storage, APG FP32 dan decoder FP32 berjalan di iPhone. Source context
dan conditioning masih berasal dari host; ini belum membuktikan pipeline
seluruhnya berjalan di iPhone atau kualitas musik lagu penuh.

## Jalankan di Mac dan iPhone

Gunakan checkout dan cache yang sama dengan persiapan input:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/activate_coreai_full_song.py --execute &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih scheme **CoreAIProbe Reuse** dan iPhone fisik, lalu **Cmd+R**. Di app tekan
**Buat / lanjutkan lagu penuh**. Tetap buka app selama proses. Tombol
**Jeda setelah bagian ini** menghentikan pekerjaan pada batas window; tombol
buat/lanjutkan memakai kembali window selesai yang hash-nya masih cocok.
App yang tidak aktif meminta jeda pada batas berikutnya; iOS dapat menghentikan
proses lebih awal sehingga bagian yang belum selesai perlu dimulai lagi.

Aktivasi tidak melakukan inference, RNG, download bobot atau native export.
Helper memeriksa seluruh input lokal dan bundle lama, menyalin job yang
terverifikasi ke folder build dengan identitas SHA, serta mencadangkan selector
dan konfigurasi sebelumnya. `--inputs` menerima folder job lain yang lengkap.
Tanpa `--execute`, helper hanya mencetak plan. Source, cache dan hasil lama
dipertahankan. Konfigurasi30s yang berbeda ditolak sebelum selector diganti.

Build resource memakai satu set14 partisi DiT, satu APG dan satu decoder256,
bukan salinan model untuk setiap instrumen. Geometri tetap750 frame/127 token,
50 Euler/APG, guidance7, momentum-0.75, cap2.5, projection dim1, BF16 learned
storage/FP32 compute/VAE. Batas abs1e-4/rel1e-3/nRMSE1e-3 tidak dilonggarkan.

## Urutan dan hasil

Instrumen diproses bergantian: seluruh window bass, drum, lalu gitar. Setiap
window mengambil slice context/noise dari timeline global tersimpan dan
memeriksa identitas condition, audio sumber serta epsilon. Sampler mempunyai
maksimal15 retained handles; handles dilepas sebelum enam panggilan decoder256
dengan core128/halo64. Cache repair dan inference retry otomatis dinonaktifkan
pada jalur lagu penuh. Tidak ada reset latent/momentum dari gold.

Raw output30s dan final latent/momentum disimpan sebelum checkpoint window
diterbitkan. Checkpoint mengikat job, konfigurasi, instrumen, index, SHA raw dan
report. Resume memeriksa ulang hasil tersimpan; korupsi menghentikan pekerjaan.
Percobaan terputus dipertahankan dalam folder attempt terpisah. Preview30s dan
pointer hasil lama tidak dipakai sebagai output lagu penuh.

Raw window dirakit dengan normalized cosine overlap-add Float64 pada panjang
lagu yang persis, lalu disimpan sebagai planar Float32. Gain tidak diterapkan
per window. Lima WAV stereo48kHz Float32 ditulis bertahap: piano, bass, drum,
gitar dan piano+band. Salinan dengar memakai satu gain bersama maksimal gain
baseline, dengan headroom-3dB. Mix band mempunyai koefisien0.25 per sumber;
ini mix diagnostik. Tidak ada EQ, limiter atau penghalusan pukulan/akor.

Setelah selesai, tombol dengar dan bagikan tersedia untuk kelima WAV. Bagikan
laporan dari app terlebih dahulu. Hasil lengkap berada di container app,
`Documents/FullSongGeneration/<execution-config-SHA>/`. File WAV lagu penuh
dapat melebihi batas upload; jangan membuat satu zip besar dari semua audio.
Manifest, report perangkat, audio dan tensor tetap privat.

## Verifikasi yang masih diperlukan

Uji host memeriksa kontrak aktivasi, transport input, resource staging dan
matematika overlap-add. Parsing grammar Swift bukan pengganti build/typecheck
dengan Apple SDK. Runner baru ini belum dibangun dengan SDK atau dijalankan
pada iPhone di lingkungan pengembangan cloud.

`COMPLETED` berarti semua window dan file output selesai, bukan strict parity
atau penerimaan musikal. Native full-song trajectory, sumber native, glitch
di overlap, harmoni antarbagian, memori, thermal dan latency perlu diukur dari
run sebenarnya. Gate drum lama4/6 tetap terpisah dan belum lolos. Tidak ada
perkiraan durasi kerja dari perkalian hasil klip pendek.
