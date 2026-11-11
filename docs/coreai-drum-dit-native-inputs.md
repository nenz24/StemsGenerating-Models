# DiT CPU pada input native yang sama

Gunakan setelah trace50 velocity terverifikasi dan kontribusi APG diperiksa.
Uji ini tidak memerlukan run iPhone baru atau trajectory50 langkah CPU baru.
Ia memakai ZIP velocity native yang sudah ada, capture host asli dan checkpoint
base/source publisher lokal yang sama, tanpa download.

Pasangan yang tersedia persis dari capture:

| Velocity yang dibandingkan | Input latent | Timestep index |
| --- | --- | --- |
| step1 | initial noise asli | 0 |
| step2 | native post-step1 | 1 |
| step6 | native post-step5 | 5 |
| step11 | native post-step10 | 10 |
| step26 | native post-step25 | 25 |

State sebelum step50 tidak tersedia secara persis. Helper tidak merekonstruksi
state itu atau menyebut input yang dekat sebagai input identik. Tabel di atas
mempertahankan alignment timestep dan menghindari pembandingan velocity pada
latent CPU yang sudah berbeda dari native.

## Jalankan di Mac

Setelah update kode, jalankan dari folder repo (sesuaikan lokasi ZIP jika berbeda):

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/audit_coreai_drum_reference30_dit_inputs.py --execute \
  --native-archive .runtime-cache/drums-native30-velocity-review-v2.zip &&
open -R .runtime-cache/drums-reference30-dit-inputs-v1/same-input-dit-review.zip
```

Tanpa `--execute`, helper hanya menampilkan rencana. Ia memverifikasi seluruh
input, source publisher dan base checkpoint sebelum memuat DiT. Output lama
dipertahankan; gunakan `--out` baru bila perlu. Ada lima forward publisher CPU
dengan learned values asli/FP32, kemudian lima forward melalui14 partisi dengan
BF16 learned storage dan FP32 compute. Cache KV control publisher dilepas sebelum
konversi wrapper untuk varian terpartisi. Tidak ada APG, VAE, Qwen/source encoder,
export native, perubahan selector, bobot produksi atau toleransi.

Report membandingkan publisher CPU terhadap native, CPU terpartisi terhadap
native, dan CPU terpartisi terhadap publisher, pada conditional/unconditional
serta gabungannya. Toleransi tetap abs1e-4+rel1e-3*abs(reference), nRMSE1e-3.
Semua pembandingan harus lolos untuk PASS per varian; hasil finite yang gagal
tetap dicatat. Ini lima kasus pada input yang sama, bukan seluruh trajectory.

Kirim `same-input-dit-review.zip` secara privat untuk analisis. ZIP berisi report
dan10 velocity CPU, tanpa bobot/source/audio. Jangan commit ZIP atau isinya.
Pengujian unit memakai nilai sintetis untuk alignment, mode plan dan penolakan
mutasi input; tidak membuktikan parity learned DiT. Uji learned memerlukan
checkpoint base lokal yang tidak tersedia di cloud. Baseline BF16 dan semua
gate native sebelumnya tetap; kelulusan APG tidak menggantikan gate trajectory.

Jika semua kasus input identik lolos, gunakan [dekomposisi efek input](coreai-drum-feedback-decomposition.md)
untuk mengukur selisih komputasi terhadap respons model pada drift latent yang
sudah teramati. Mode itu hanya menjalankan sembilan evaluasi publisher CPU dan
tidak memerlukan run iPhone baru.
