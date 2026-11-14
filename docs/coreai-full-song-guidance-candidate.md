# Kandidat kualitas drum satu window

Kandidat ini menguji guidance4 pada satu window dengan input yang sama.
Baseline tetap guidance7. Ini hipotesis kualitas setelah pemeriksaan file,
decoder dan keterulangan hasil; belum ada klaim glitch, harmoni atau gaya
terselesaikan. Kandidat tidak dipakai pada33 run atau mengganti hasil lagu.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Saat iPhone dingin, jalankan **CoreAIProbe Reuse → Cmd+R**, isi **Detik fokus**
yang sama dengan review, lalu langsung tekan **Coba kandidat drum guidance 4**.
Lakukan sebelum ekspor/uji lain dalam proses app tersebut. Setelah selesai,
bandingkan **Dengarkan Drum baseline (fokus)** dan **Dengarkan Drum guidance 4 (fokus)** pada
volume yang sama, kemudian **Bagikan paket fokus**.
App juga menyediakan piano+drum baseline dan kandidat untuk memeriksa groove
pada seluruh window30s. ZIP hanya menyertakan WAV fokus, sehingga ukuran tetap
di bawah batas upload; raw30s kedua hasil tetap disertakan untuk audit.

Uji hanya membuat satu bagian audio drum30s. Input context/noise/condition,
timestep asli,50 Euler steps, BF16 learned storage dan FP32 compute/VAE
dipertahankan. Momentum-0.75, cap2.5, projection dim1 serta seluruh toleransi
tetap. Tidak ada bobot, native export, Qwen, source encoding atau RNG baru.
Jika batas memori, input, thermal atau kontrol aritmetika gagal, uji berhenti.
Cache OS/actual placement tetap tidak terkontrol, tanpa cache repair/deletion.

APG asli menghasilkan `guided7 = conditional + 6*orthogonal_update`.
Momentum/cap/projection tidak bergantung pada guidance untuk state yang sama.
Kandidat memakai `conditional + 0.5*(guided7-conditional)` lalu Euler FP32,
sehingga koreksi guidance efektif berkurang setengah menjadi guidance4.
Native APG tetap menghitung momentum/cap/projection pada state kandidat yang
aktual; latent tidak direset dari baseline. Tidak ada pengurangan gain/filter
audio untuk menyamarkan glitch.

Host memeriksa identitas affine terhadap APG publisher pada input tetap.
Di iPhone, Euler FP32 guidance7 dihitung ulang dari native guided7 untuk setiap
state kandidat dan wajib cocok dengan native latent_next pada batas asli,
50/50, sebelum hasil kandidat dapat diekspor. Ini kontrol aritmetika, bukan
full-trajectory parity kandidat atau penerimaan musikal. Build SDK dan musik
kandidat tetap perlu diverifikasi pada perangkat.

ZIP memuat actual input, raw audio/latent baseline dan kandidat, laporan serta
cuplikan fokus dengan satu gain bersama maksimal gain baseline. Jika diperlukan,
kedua salinan diturunkan sama besar untuk headroom-3dB; raw comparison tetap
sebelum gain dan file baseline asli tidak berubah. Kandidat memiliki profile
`full-song-guidance-candidate-v1`; audit memerlukan flag eksplisit `--candidate`
pada `scripts/audit_coreai_drum_repeat_review.py`. Perbedaan terhadap baseline
yang bermasalah memang diharapkan. Hasil dalam/di luar toleransi pada
perbandingan tersebut bukan kriteria mutu kandidat, dan tidak mengubah gate
strict baseline lama. ZIP dan semua data perangkat tetap privat.

Terima kandidat hanya melalui uji dengar yang memperhatikan glitch, groove,
timbre dan kecocokan dengan piano. Jika glitch berkurang tetapi musik/akor
memburuk, kandidat ditolak. Hasil satu fokus tidak membuktikan kualitas atau
konsistensi gaya seluruh lagu, bass maupun gitar.
