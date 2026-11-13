# Input global dan perakitan lagu penuh

Tahap ini menyiapkan satu timeline piano penuh, actual global source epsilon,
noise per instrumen dan identitas setiap window30s. Bass, drum dan gitar tetap
memakai satu geometri sampler750-frame/127-token,50 Euler/APG guidance7,
momentum-0.75/cap2.5/projection dim1, BF16 learned storage dan FP32 compute/VAE.
Toleransi abs1e-4/rel1e-3/nRMSE1e-3 tetap. Preview30s lama tidak ditimpa.

## Jalankan di Mac

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/prepare_coreai_full_song_inputs.py \
  --execute --choose-source --reuse-verified &&
open -R .runtime-cache/full-song-inputs-v1/reference.json
```

Pilih audio piano asli untuk lagu penuh pada dialog. Jika persiapan sudah ada
dan lolos pemeriksaan, `--reuse-verified` memakai input tersimpan tanpa meminta
pilihan ulang, mengulang encoder atau membuat noise baru. `--source` menerima
file sebenarnya sebagai alternatif dialog; jika diberikan pada job existing,
hash-nya wajib cocok dengan source job tersebut. Output yang berbeda/korup
dipertahankan dan helper berhenti, tanpa fallback diam-diam atau overwrite.

Default input existing:

- `.runtime-cache/full-song-conditioning-v1`: cache tiga instrumen yang sudah diuji;
- `.runtime-cache/drums-reference30-v3`: timestep asli yang dipatok;
- `ModelCandidates/source-reference30-window-encoder-256`: bundle source dengan32 gate host asli.

Checkpoint VAE harus sudah ada lokal. Bundle source diverifikasi sebelum proof
host digunakan kembali, tanpa mengulang32 pemeriksaan atau export. Override
`--source-audit` menerima standalone audit tertentu; audit gagal yang dipilih
eksplisit tetap gagal dan tidak diganti dengan proof dari bundle lain.

Audio lengkap disiapkan sekali ke32kHz PCM16 dengan source gain0.7 dan
nearest-even yang sama. Resampling ke48kHz dilakukan sekali pada timeline
global; source encoding berjendela256/core224 menggunakan actual global
epsilon yang disimpan. Bila global context sudah ada, hash, ukuran, padding,
mask serta hubungan posterior/epsilon/source/context wajib cocok sebelum reuse.
Tidak ada Qwen/DiT generation atau download bobot pada tahap ini.

## Isi job privat

`full-song-inputs-v1/reference.json` mencatat jadwal window, setting yang tetap,
provenance, shape/hash semua file dan input setiap run. Data global terdiri
dari piano48, context, epsilon, timestep asli dan tiga condition/mask/noise.
Noise global memakai CPU FP32 dengan generator terpisah per instrumen dan
disimpan sebagai array aktual. Seed hanya informasi; seed tidak menjamin
kesamaan lintas CPU atau kesamaan input dengan baseline30s lama.

Window memakai slice timeline yang sama pada overlap. Source audio planar
stereo, context, epsilon dan noise masing-masing memiliki hash slice yang
dicatat untuk setiap run. Identitas33 run, misalnya, tidak berarti33 generation
sudah dilakukan. Input source dibatasi minimal750 global latent frame.

Plan cache lama yang hanya belum memiliki `right_padding_samples48` dapat
diverifikasi dengan menurunkan nilai tersebut dari start/end yang sama, di
memori saja. Semua posisi dan field lain tetap wajib cocok; plan asal tidak
ditulis ulang. Durasi akhir mengikuti jumlah sampel PCM sebenarnya.

Input besar tetap lokal. Untuk review, cukup bagikan `reference.json`, bukan
seluruh folder. Audio, noise, context, manifest dan report bersifat privat;
jangan commit semuanya ke repo publik.

## Overlap-add dan batas pembuktiannya

`coreai_full_song_assembly.py` memakai ramp cosine dengan offset setengah
sampel dan penjumlahan Float64. Jumlah bobot dinormalisasi per sampel; seluruh
panjang lagu wajib tercakup sebelum output Float32 dapat diambil. Overlap tiga
window, padding akhir, stereo, window duplikat/hilang dan data nonfinite diuji.
Window menerima raw30s output, tanpa gain otomatis per window atau filter.

Perakitan ini menggabungkan generation window yang berbeda pada timeline lagu
penuh. Ia tidak mengubah decoder256/core128 atau menutupi numerical mismatch
decoder. Rekonstruksi identik dari slice waveform yang sama hanya membuktikan
matematika perakitan; belum membuktikan musik dari beberapa generation akan
tersambung mulus. Phase/timing/harmoni di overlap tetap perlu uji dengar.

## Status native

Job disimpan **inactive**, `ready_for_native_execution: false`; tidak mengubah
selector/scheme/build iPhone. Native batch runner untuk seluruh job belum
diimplementasikan pada tahap ini. Full-song source encoder/generation/musik,
resource, thermal dan latency belum dievaluasi. Frontend context pada job ini
berasal dari host, sehingga bukan bukti pipeline seluruhnya berjalan di iPhone.
Gate sampler drum4/6 sebelumnya tetap terpisah dan belum lolos.

Tanpa `--execute`, helper hanya mencetak plan menggunakan stdlib. Tidak ada
estimasi waktu lagu penuh dari perkalian waktu klip pendek. Lihat
[panduan performa](coreai-bass-preview-performance.md) dan
[precision A/B/A](../reviews/coreai-iphone17pro-precision-aba/README.md).
