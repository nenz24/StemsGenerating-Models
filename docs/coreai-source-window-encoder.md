# Source encoder berjendela untuk persiapan lagu penuh

Source harus berubah mengikuti piano sepanjang lagu. Profil diagnostik ini
memakai encoder FP32 lama dengan256 frame input, menyimpan core224 dan halo16.
Proof menelusuri setiap Conv1d secara terbalik; padding buatan tidak boleh
mencapai core. Snake dan posterior sampling bersifat lokal; operator temporal
baru/global ditolak oleh guard arsitektur. Tidak ada crossfade/filter pada latent.

## Audit CPU dan pilihan backend referensi

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/audit_coreai_source_window.py --execute
```

Empat window dibandingkan langsung ke capture750 asli: mean, std, source latent
dan context pada empat core, tiga sambungan dan keseluruhan. Target32 gates
tetap abs1e-4+rel1e-3*abs(reference), nRMSE1e-3. Epsilon/mask memakai byte capture
asli, tanpa RNG baru, gain tambahan atau reset ke gold.

ATen CPU dipakai eksplisit untuk referensi ini (`--convolution-backend aten`).
Kontrol `default` tetap tersedia dan hasil gagal dipertahankan. oneDNN dan ATen
dapat berbeda pada titik batas akibat urutan operasi FP32; ini pilihan backend
referensi host, bukan perubahan precision, learned value, native asset atau
toleransi. Kelulusan terhadap full encoder pada CPU yang sama tidak otomatis
menerima perbandingan terhadap acuan Mac: kedua gate dinilai terpisah.

## Siapkan uji satu aset di iPhone

Dari folder repo Mac:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_source_window.py --execute --activate &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Helper memverifikasi dan menyalin `source-context-1024/source-context.aimodel`
yang sudah ada. Tidak ada export/unduhan baru. Profil harus FP32/256 dengan
checkpoint VAE dan folding asli; tiga payload native diperiksa hash. Bila
aset tidak tersedia atau tidak cocok, proses berhenti sebelum inference host.
Seluruh32 gate CPU harus lolos sebelum bundle baru disalin atau selector berubah.
Selector lama dibackup; aset, source dan cache lama dipertahankan. Build hanya
menyertakan13 file: satu encoder, capture/referensi dan provenance; tanpa DiT,
APG, Qwen atau decoder. Native runtime memverifikasi identitas semua file serta
hash audio/epsilon/mask setiap tile sebelum call.

Pilih **CoreAIProbe Reuse**, iPhone asli, **Cmd+R**, lalu tekan **Uji input piano
30 detik** sekali. Sesudah selesai pilih **Bagikan laporan**. Normalnya empat
call, satu open/bind, tiga reuse hits dan32 perbandingan. Proses baru diperlukan
untuk mencoba lagi; jika gagal, bagikan laporan tersimpan/phase dahulu.
Empat array source native lengkap juga disimpan dalam folder Documents per-run
beserta hash di report agar kegagalan finite dapat dianalisis tanpa inference ulang.

Tidak ada retry/cache deletion, generasi/WAV baru atau perubahan BF16 sampler.
Numerical gate dan sampled process budget2000MiB dinilai terpisah. Pre-call dan
post-bind guards serta sampling10ms tidak menjamin exact transient/native peak.
Host gates bukan bukti Apple SDK build, native parity, full-song resources atau
musical acceptance; statusnya tetap `NOT EVALUATED` sampai ada bukti perangkat.

## Context seluruh lagu di host

Setelah [source penuh](coreai-full-song-preparation.md) siap dan audit CPU lolos:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_full_song_context.py --execute
```

Frontend melakukan resample sekali pada timeline penuh, lalu padding hanya di
ujung global ke grid40ms. Array epsilon global aktual disimpan dan setiap tile
memakai slice dari array itu; seed hanya metadata. Mask harus terpelihara persis.
Mean/std/source/context disusun dengan setiap frame disimpan sekali. Hash semua
artifact disimpan; tidak ada generasi DiT, Qwen, native export atau aktivasi.

Ini capture host penuh berbasis proof lokal dan audit30s, bukan perbandingan
dengan encoder monolitik seluruh lagu atau native full-song PASS. Source128
tersedia hanya sebagai kontrol host; profil perangkat ini tetap256. Conditioning
ketiga instrumen, global sampling noise, overlap-add audio dan full-song
resource/listening masih perlu diselesaikan. Kegagalan trajectory baseline
tetap dicatat pada toleransi asli dan tidak ditimpa gate source ini.
