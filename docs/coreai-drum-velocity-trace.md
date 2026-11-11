# Rekam velocity native untuk memisahkan DiT dan APG

Kelulusan lima checkpoint latent tidak otomatis menerima momentum akhir atau
waveform: batas setiap titik diterapkan pada masing-masing output. Report
sampler lama `host_vs_*_strict_verdict` hanya merangkum lima latent. Field baru
`*_acceptance` memisahkan latent, momentum dan sampled state; waveform tetap
`NOT EVALUATED` jika belum didecode.

Run lama tidak menyimpan seluruh velocity dari DiT. Data yang hilang tidak bisa
dipulihkan dari checkpoint sparse. Revision `reference30-live50-window256-v2`
merekam **50 velocity aktual tepat sebelum APG**, masing-masing `[2,750,64]`
Float32 conditional/unconditional. Pencatatan hanya menyalin output ke disk;
APG tetap menerima tensor native yang sama. BF16, FP32 compute, parameter,
bobot, feedback, decoder dan toleransi tetap. Tidak ada export model baru.
Run baru memakai folder UUID baru dan hasil lama dipertahankan.

## Satu run baru di iPhone

Dari folder repo Mac:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Selector generasi acuan yang sudah aktif tetap dipakai. Pilih **CoreAIProbe
Reuse**, target iPhone asli, lalu **Cmd+R**. Tekan **Generate drum 30 detik (acuan)**
sekali. Tunggu50/50 +6 decode. Report harus mencatat revisionv2 dan
`Velocity trace SHA256: …; tensors50/50`. Ini pengumpulan data, bukan perubahan
yang menjanjikan PASS baru. Replay sampler CPU Mac tidak perlu diulang.

Unduh container terbaru lewat **Devices and Simulators → iPhone → Apps → CoreAI
Probe → klik kanan → App Container → Download…**. Simpan ke lokasi baru agar
unduhan lama tetap ada. Kemas dengan mengganti path contoh ke unduhan baru:

```bash
python3 scripts/package_coreai_drum_reference30_run.py \
  --source "/path/to/new/CoreAIProbe-container" \
  --run-id latest \
  --require-velocity-trace \
  --out .runtime-cache/drums-native30-velocity-review-v2.zip &&
open -R .runtime-cache/drums-native30-velocity-review-v2.zip
```

`latest` memakai pointer run yang disimpan app. Trace harus lengkap50/50 dan
metadata diikat hash ke report. ZIP berisi58 tensor +3 metadata, sekitar32,1MB
raw sebelum kompresi dan tetap dibatasi32MiB. Bagikan ZIP secara privat; jangan
commit ke GitHub. Formatv1 delapan tensor tetap didukung. `--source` menerima
folder container, `.xcappdata`, Documents atau direktori run tertentu (untuk
run tertentu gunakan UUID eksplisit).

`--require-velocity-trace` menolak runv1 sebelum membuat ZIP. Nama ZIPv2 tidak
membuktikan isi runv2. Jika container masih memilih v1, periksa baris Revision
di laporan terbaru langsung dalam app: bila sudahv2 selesai, cukup unduh
container baru; tidak perlu Generate ulang. Bila app masihv1, update/build
terbaru dahulu. Pakai nama output baru jika ZIP lama sudah ada.

## Replay setelah trace diterima

CPU mengulang APG dengan velocity native dibekukan, sambil meneruskan latent/
momentum CPU sendiri. Dua pemeriksaan memakai formula FP32 yang diekspor serta
source publisher asli dengan projection FP64. Ini hanya diagnostik; APG app
dan precision produksi tidak berubah.

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/replay_coreai_drum_reference30_apg_trace.py \
  --archive .runtime-cache/drums-native30-velocity-review-v2.zip \
  --out .runtime-cache/drums-native30-apg-trace-audit-v2.json
```

Helper memakai capture host dan source APG lokal yang diverifikasi, tanpa bobot
DiT/VAE, download, native export atau perubahan selector. Lima latent serta
momentum akhir dibandingkan ke native yang menghasilkan velocity itu. Ini
mengukur kontribusi APG pada trace yang sama, bukan DiT dengan feedback CPU baru,
kelulusan seluruh native sampler atau penerimaan lagu penuh bass/drum/gitar.

Tidak ada runv2 learned/native atau Apple SDK build yang dijalankan di cloud.
Grammar Swift dan test sintetis memeriksa integrasi/transport/control flow;
bukti numerik native berikutnya tetap berasal dari perangkat asli.

Setelah kontribusi APG pada trace yang sama diperiksa, lanjutkan ke
[DiT pada input native identik](coreai-drum-dit-native-inputs.md). ZIP velocity
yang sudah ada menyediakan lima pasangan exact input/output tanpa run iPhone baru.
