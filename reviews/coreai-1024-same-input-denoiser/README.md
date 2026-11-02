# Denoiser lengkap, empat input identik

Delapan forward aktual dijalankan pada Linux: empat melalui implementasi penuh
FP32 asli, empat melalui seluruh 24 layer yang dipartisi dengan BF16 storage /
FP32 calculations. Input adalah latent FP32 setelah gold step 1, 10, 25 dan 50,
dengan timestep berikutnya (`times[step]`), conditioning FP32 yang sama serta
konteks audio yang sama. Referensi dibuat sebelum parameter model dikonversi.

Keempat kasus **strict PASS** menggunakan batas abs 1e-4, rel 1e-3 dan normalized
RMSE 1e-3 semula. Normalized RMSE berkisar 5,61e-7–1,12e-6; max absolute error
terbesar 2,15e-5. Hash fixture ditautkan ke biner gold yang benar-benar dipakai.
Ini bukti per-call pada empat fixture, bukan seluruh ruang input atau kesamaan
lintasan bebas 50 langkah. Gold fixture tidak pernah menjadi feedback atau
pengganti output model dalam sampler perangkat.

Aset native yang ada tidak diekspor ulang atau diubah; manifest lokal ditambah
hasil pemeriksaan aktual dan diverifikasi ulang secara penuh. Aktivasi opt-in
dan dry-run packaging berhasil dengan satu profil, sekitar 6,261 GB resource.
Tidak ada eksekusi CoreAI native Apple, build Swift atau uji iPhone dalam bukti
ini. Mac pengguna harus menjalankan pemeriksaan yang sama sebelum mode baru
boleh mengekspor. `checks.json` menyimpan hasil dan lingkup buktinya.
