# Uji satu window drum dengan input yang sama

Jika suara yang bermasalah direproduksi oleh CPU VAE dari native final latent
dengan batas asli, diagnosis berlanjut ke hasil sampling/input dan reproduksinya.
Parity decoder pada satu range belum membuktikan full trajectory atau mutu musik.
Uji ini mengulang **satu** window30s drum, bukan seluruh lagu atau parameter baru.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih **CoreAIProbe Reuse** dan iPhone yang sama. Saat perangkat sudah dingin,
jalankan app dengan **Cmd+R**. Isi **Detik fokus**, lalu tekan **Uji ulang drum
fokus saja** sebelum menjalankan ekspor atau generasi lain dalam proses app itu.
Setelah selesai, tekan **Bagikan paket fokus**. App tidak mengganti hasil lagu
penuh. Jika uji berhenti sebelum sampling karena thermal, tunggu perangkat
dingin dan buka ulang app. Tidak ada retry otomatis.

Fokus harus berada di luar overlap agar tepat satu window dipilih. Runner
memeriksa job SHA dan semua array global, mengambil condition/context/noise
yang hash-nya sama dengan run tersimpan, dan menggunakan timestep asli.
Tidak ada Qwen/source encoding, RNG baru atau native export. BF16 learned
storage, FP32 compute/APG/VAE,50 Euler/APG, guidance7, momentum-0.75, cap2.5,
projection dim1 serta abs1e-4/rel1e-3/nRMSE1e-3 tetap. Jalur sampling dan lifecycle
handles yang sama dipakai; original reference, checkpoints dan hasil tetap.

UI mengharuskan proses app baru dan thermal nominal sebelum model pertama.
Cache OS/native dan actual placement tidak dikendalikan. Uji tidak menghapus
cache. Perbedaan antar-run tidak otomatis membuktikan efek thermal/cache;
kesamaan hasil juga tidak membuktikan mutu musik atau semua window lain.

Final latent dan full30s raw audio dibandingkan terhadap capture semula pada
batas asli. Referensi ini adalah hasil bermasalah yang sedang diperiksa,
bukan host gold atau hasil musik yang diterima. PASS berarti hasil dapat
direproduksi pada perbandingan tersebut. FAIL tetap dilaporkan dan tidak
memicu regenerasi, reset gold atau pelebaran toleransi. Gate drum lama4/6
tetap terpisah. Cold-control yang baru tetap perlu build SDK dan uji perangkat.

ZIP kecil memuat actual input sampler, dua raw audio/latent, laporan dan dua
cuplikan fokus dengan gain bersama dari hasil lama. Data bersifat privat;
jangan commit ZIP atau isi hasil perangkat ke repo. Audit stdlib tersedia
di `scripts/audit_coreai_drum_repeat_review.py --archive` dengan path ZIP asli.

Jika hasil buruk direproduksi pada input yang sama, kontrol kualitas terpisah
tersedia melalui [kandidat guidance4 satu window](coreai-full-song-guidance-candidate.md).
Kandidat harus dipilih eksplisit dan tidak mengganti baseline atau mengubah
interpretasi PASS/FAIL uji keterulangan ini.
