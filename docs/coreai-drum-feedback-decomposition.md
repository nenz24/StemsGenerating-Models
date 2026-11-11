# Pisahkan selisih operator dari efek drift input DiT

Mode ini melanjutkan pemeriksaan saat lima kasus DiT pada input identik lolos,
tetapi trajectory50 langkah masih gagal. Kelulusan komponen tidak membuktikan
kelulusan trajectory; report/gate awal tidak diganti oleh diagnosis ini.

Tiga output dibandingkan pada setiap kasus:

- `N`: velocity native yang direkam dari run iPhone.
- `C`: publisher CPU pada latent native yang persis sama.
- `R`: publisher CPU pada checkpoint latent acuan asli, dengan timestep yang sama.

Identitas selisihnya: `N-R = (N-C) + (C-R)`. `N-C` mengukur selisih komputasi pada
input identik; `C-R` mengukur respons model CPU terhadap drift input yang sudah
teramati. Kedua evaluasi CPU memakai model, dtype dan cache conditioning yang
sama. Komponen error bisa saling memperkuat atau membatalkan: RMSE tidak boleh
dijumlahkan sebagai persentase penyebab. Report menyimpan cosine, cross term
dan pemeriksaan rekonstruksi vektor selain RMSE.

## Jalankan di Mac

Gunakan ZIP native velocity yang sudah ada. Sesuaikan nama file bila berbeda:

```bash
.runtime-cache/coreai-decoder-venv/bin/python \
  scripts/audit_coreai_drum_reference30_dit_inputs.py --execute \
  --input-drift-only \
  --native-archive .runtime-cache/drums-native30-velocity-review-v2.zip \
  --out .runtime-cache/drums-reference30-feedback-v1 &&
open -R .runtime-cache/drums-reference30-feedback-v1/feedback-decomposition-review.zip
```

Tanpa `--execute`, hanya rencana ditampilkan. Gunakan `--out` baru bila output
sudah ada. Tidak perlu menjalankan generasi iPhone atau mengunduh container lagi.
Mode ini membutuhkan checkpoint base, capture asli dan source publisher lokal
yang diverifikasi. Helper tidak mengunduh bobot atau mengubah aset produksi.

Ada sembilan forward publisher CPU: lima pada input native untuk langkah
1/2/6/11/26, serta empat pada input acuan untuk langkah2/6/11/26. Pada langkah1
initial noise identik sehingga satu hasil dipakai untuk kedua sisi. Tidak ada
replay partisi, APG, VAE, conditioning/source baru atau trajectory50 langkah baru.
Reference input selalu post-step1/5/10/25 tersimpan, bukan state yang direkonstruksi.

Report mencatat hash input, output, sumber dan velocity native. Tiga pembandingan
numerik pada conditional/unconditional/gabungan tetap memakai abs1e-4 +
rel1e-3*abs(reference), nRMSE1e-3. Reference velocity `R` adalah hasil evaluasi CPU
sekarang pada checkpoint acuan; bukan klaim bahwa velocity intermediate asli
yang tidak diarsipkan sudah ditemukan. Scope tidak meliputi langkah27–50 atau
counterfactual trajectory lengkap dengan feedback baru.

Kirim `feedback-decomposition-review.zip` secara privat. Isinya report dan10
velocity CPU, tanpa bobot/source/audio; jangan commit ZIP atau isinya ke GitHub.
Baseline BF16, precision produksi, toleransi dan status trajectory asli tetap.
Test sintetis memeriksa identitas error, pembatalan vektor dan mode tanpa
dependency; evaluasi learned memerlukan checkpoint base lokal pada Mac.
