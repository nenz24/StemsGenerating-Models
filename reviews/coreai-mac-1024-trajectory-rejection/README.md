# Mac: lintasan 10,24 detik ditolak gate

Output pengguna menunjukkan normalized RMSE latent akhir **0,0014409045**,
melebihi batas **0,001**, dengan 52,893% elemen melewati batas pointwise.
Producer berhenti sebelum ekspor sampler dan aktivasi; `Ready` tidak tercapai.
SubprocessError di wrapper adalah akibat kegagalan gate tersebut, bukan bukti
kehabisan RAM, kegagalan Xcode atau kerusakan checkpoint.

`console.txt` dipertahankan byte-for-byte. `summary.json` memisahkan metrik yang
tercetak dari data yang tidak disertakan: hanya 11 checkpoint yang dicetak,
bukan seluruh 50 baris maupun manifest source/conditioning. Empat uji denoiser
pada input identik belum tersedia pada versi producer ini di Mac pengguna.

Perubahan berikut menambahkan mode **opt-in** `--diagnostic-trajectory`, hanya
untuk 10,24 detik. Denoiser lengkap dibandingkan dengan implementasi asli pada
empat fixture identik; seluruh 50 APG identical-input checks tetap wajib strict
PASS. Free-trajectory yang gagal tetap gagal, dengan label
`trajectory_drift_diagnostic`, bukan `bounded_diagnostic` atau numerical PASS.
Jalur default tetap menolak kasus ini. Perubahan tidak memperbaiki ataupun
membuktikan tidak pentingnya selisih penuh; ia memungkinkan eksperimen uji
dengar yang lingkup validasinya dinyatakan jelas. Belum diuji pada Mac/iPhone.
