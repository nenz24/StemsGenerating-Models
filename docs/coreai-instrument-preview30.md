# Preview bass, drum dan gitar30s dengan sampler bersama

Gunakan cache conditioning yang sudah dihitung pada Mac untuk memeriksa kualitas
bass/gitar pada piano acuan30s sebelum generation seluruh lagu. Source context,
actual initial noise dan schedule tetap dari capture yang sama. Ini preview
musikal tiga instrumen; durasi lagu penuh belum dijalankan.

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_instrument_preview30.py --execute --activate &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Pilih scheme **CoreAIProbe Reuse**, destination iPhone asli, lalu **Cmd+R**.
Pada app pilih **Bass**, tekan **Buat bass 30 detik**. Setelah selesai,
dengarkan solo serta piano+bass, lalu **Bagikan laporan** dan WAV jika perlu
ditinjau. Pilih **Gitar** untuk **Buat gitar 30 detik**; hasil tiap instrumen
tersimpan terpisah dan tersedia kembali setelah app dibuka ulang. **Drum**
tersedia sebagai kontrol, tanpa mewajibkan pengulangan run drum sebelumnya.
Tunggu satu instrumen selesai sebelum memilih/membuat instrumen berikutnya.

Helper memakai `.runtime-cache/full-song-conditioning-v1` dan bundle sampler/
decoder existing di `ModelCandidates`. Ia memeriksa21 array finite beserta
ukuran/hash, definisi preset, checkpoint pins, serta kesamaan tujuh array drum
secara byte terhadap capture acuan. Mask harus berisi127 nilai1. Hanya tiga
packed condition, tiga mask dan report yang disalin ke bundle tambahan.
ID, lirik embedding dan publisher silence juga harus konsisten lintas instrumen.
Tidak ada model inference, export, download, perubahan source atau baseline.

Pemilihan profil baru dilakukan setelah verifikasi; selector lama dibackup.
Bundle/config existing yang berbeda dipertahankan dan helper berhenti.
Tanpa `--execute`, helper hanya mencetak plan. `--source` menerima folder cache
existing lain, `--candidates` menerima folder model existing lain.

Build memakai satu set15 aset sampling BF16/FP32 dan satu decoder256 FP32.
Qwen/source encoder/full750 decoder tidak ikut build ini. Setiap instrumen
tetap50 Euler/APG, guidance7, momentum-0.75, cap2.5, projection dim1. Step1
memakai scope serial, mulai step2 maksimal15 retained handles; pool dilepas
sebelum enam panggilan decoder berjendela. Tidak ada pemuatan tiga set bobot.
Guard2000MiB dan sampler footprint10ms tetap berlaku per run; pelepasan handle
tidak menjamin native eviction atau transient peak. Tidak ada estimasi waktu
lagu penuh dari run preview ini.

Drum tetap dibandingkan pada checkpoint1/5/10/25/50 serta final raw audio,
abs1e-4/rel1e-3/nRMSE1e-3. Bass/gitar tidak memiliki matching host trajectory
atau waveform; numerical parity keduanya **NOT EVALUATED**, tidak dibandingkan
terhadap gold drum. Kelulusan conditioning/source tidak mengganti gate sampler
drum sebelumnya yang4/6. Uji dengar tetap terpisah dari numerical parity.

Hasil baru disimpan di `Documents/InstrumentReference30/<instrument>/<UUID>/`,
dengan pointer terakhir per instrumen. Capture drum diagnostik lama dan pointer
lamanya tetap terpisah. Setiap run menyimpan lima live latent checkpoint,
final latent/momentum, raw audio, report, solo WAV, piano WAV serta mix WAV.
Preview tidak menulis50 velocity trace tambahan. Diagnostic drum lama tetap
memiliki trace tersebut untuk audit yang sudah ada.

Salinan untuk didengar menggunakan satu gain headroom untuk solo, piano dan
mix dalam run yang sama, dengan mix0.5piano+0.5instrumen. Raw output disimpan
sebelum attenuation. Gain antar-instrumen dapat berbeda bila peak berbeda;
ini belum merupakan full-band mix atau loudness comparison terkontrol.
Simpan WAV, array, report dan container sebagai artefak privat.

Perubahan Swift perlu build/typecheck SDK Xcode dan eksekusi pada iPhone.
Pemeriksaan grammar pada Linux tidak membuktikan kelulusan build atau runtime.
Setelah bass/gitar didengar, integrasi context penuh/noise global,
overlap-add serta pengukuran resource/thermal seluruh lagu masih diperlukan.

Jika bass/gitar masih memiliki glitch, gunakan [ekspor hasil existing untuk
review transisi](coreai-instrument-transition-review.md). Helper mengambil
raw audio dan latent yang sudah tersimpan; tidak perlu generation atau build
app baru untuk mengambil data diagnosis.
