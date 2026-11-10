# Replay sampler acuan di CPU Mac

Gunakan replay ini untuk memisahkan selisih trajectory CPU terpartisi dari
selisih native setelah decoder diperiksa dengan **final latent native yang
sama**. Hasil decoder pada latent acuan saja tidak menjawab pertanyaan tersebut.
Uji ini tidak memerlukan generasi iPhone baru.

## Input dan batas yang tetap

- Capture host asli `.runtime-cache/drums-reference30-v3`, termasuk conditioning,
  source context, initial noise, timesteps dan lima checkpoint asli.
- ZIP native yang sudah diekspor: `.runtime-cache/drums-native30-live-review-v1.zip`.
- Checkpoint base dan source publisher asli yang sudah tersimpan di Mac.
- BF16 learned storage, FP32 compute/APG/I/O; CFG7, shift1, momentum−0.75,
  cap2.5 dan projection dim1. APG memakai implementasi FP32 yang diekspor.
- Toleransi asli abs1e-4 + rel1e-3*abs(reference), nRMSE1e-3. Keduanya harus lolos.

Semua 50 langkah memakai feedback latent/momentum CPU sendiri. Gold dan native
dibaca hanya untuk pembandingan checkpoint1/5/10/25/50; keduanya tidak pernah
menjadi reset feedback. Tidak ada Qwen, source encoder, RNG baru, VAE, export
native, perubahan selector/mix atau penggantian baseline. Helper tidak mengunduh
bobot/source; jika berkas lokal belum tersedia, proses berhenti.

## Jalankan

Dari folder repo di Mac, setelah memperbarui kode:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/replay_coreai_drum_reference30_sampler.py --execute &&
open -R .runtime-cache/drums-reference30-sampler-replay-v1/sampler-replay-review.zip
```

Tanpa `--execute`, script hanya menampilkan rencana dan tidak memuat model atau
menulis berkas. `--help` mencantumkan opsi bila lokasi capture, ZIP, checkpoint
atau source publisher berbeda. Gunakan output baru untuk run lain; output lama
selalu dipertahankan. Ini komputasi CPU lokal yang dapat berjalan beberapa menit.

Kirim `sampler-replay-review.zip` secara privat untuk pemeriksaan. ZIP hanya
berisi report dan enam tensor host (lima checkpoint latent serta momentum akhir),
tanpa bobot, source context atau WAV. Jangan commit ZIP atau isinya ke GitHub.

## Arti hasil

Report mencatat tepat 50 langkah dan 700 panggilan partisi CPU (14 per langkah),
terpisah dari hitungan runtime native. `host_vs_gold` mempertahankan acuan asli;
`host_vs_native` membandingkan sparse checkpoint terhadap run iPhone tersimpan.
Selisih finite dicatat sebagai `NOT PASSED` sambil loop independen dilanjutkan;
nonfinite atau kontrak input salah menghentikan replay.

Jika host juga gagal terhadap gold, selidiki perbedaan trajectory partisi/APG
terhadap publisher. Jika host lolos gold sementara native gagal, selidiki
perbedaan komputasi native. Hasil itu belum menunjuk operator tertentu. Lolos
terhadap native juga tidak otomatis berarti lolos gold: kedua gate dinilai
sendiri. Pengujian sparse checkpoint bukan bukti seluruh intermediate dan
tidak menerima full-song bass/drum/gitar atau kualitas musik secara otomatis.

Empat test baru memeriksa mode tanpa dependency, feedback independen, isolasi
observer, penolakan schedule salah dan penghentian pada nonfinite. Test memakai
nilai sintetis dan tidak membuktikan parity learned model. Replay sampler penuh
memerlukan bobot base lokal; belum diuji di cloud yang hanya memiliki VAE.
