# Pemeriksaan aset native pada proses baru

**Perkembangan:** [run drum berikutnya](../reviews/coreai-iphone17pro-drums-quality-rejection/README.md)
selesai 50/50 dan menghasilkan WAV, tetapi kualitas ditolak. Untuk keluhan
glitch/level saat ini ikuti [evaluasi kualitas](coreai-drums-quality.md), tanpa
preflight/generasi ulang. Tidak ada laporan preflight yang disertakan untuk
run selesai tersebut; hubungan sebab-akibat dengan warmup belum terbukti.

[Run drum pertama](../reviews/coreai-iphone17pro-drums-initial-specialization/README.md)
gagal saat bind `blocks-16-17`, lalu gagal specialization pada retry yang sudah
dibatasi satu kali. Conditioning enam checks PASS; 0/50 langkah dan tanpa WAV.
Penyebab internal belum diketahui. Jangan menambah retry atau menghapus cache
lain berdasarkan laporan ini.

Mode berikutnya terpisah dari generasi: 16 forward sintetis, masing-masing
satu pada front, 12 pasangan DiT, tail, APG dan decoder. Encoder/Qwen/conditioning
tidak dijalankan. Prior failure dari laporan instrumen yang sama didahulukan
jika nama aset ada pada allowlist; nama laporan tidak dapat memilih path lain.
Pada laporan saat ini diharapkan `blocks-16-17.aimodel` menjadi aset pertama.

## Jalankan satu kali, tanpa preparation/export ulang

Hentikan proses app lama dari Xcode, lalu tutup Xcode dan update:

```bash
git fetch origin main &&
python3 scripts/update_coreai_project.py &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Profil drum yang sudah aktif tetap digunakan. Pilih **CoreAIProbe Reuse**,
Release, iPhone; Cmd+R. **Jangan impor piano atau menjalankan model lain dulu.**
Tekan **Periksa aset sebelum generasi** sekali. Header laporan
`native-preflight-v1`; cek baris prior failure FIRST dan planned order.

Tombol tersedia sebelum pekerjaan lain pada proses itu. Sesudah preflight,
Generate/prefix ditahan sampai restart; tidak ada warmup tersembunyi dalam
sampling produksi. Bagikan laporan pemeriksaan dari bagian hasil.

- Jika **NOT COMPLETED**, berhenti dan bagikan laporan serta baris BNNS/Xcode
  console terakhir. Jangan mengulang tombol atau Generate berkali-kali.
- Jika **COMPLETED 16/16**, simpan/bagikan laporan. Tekan **Stop di Xcode** untuk
  menghentikan proses, kemudian buka app yang sama dari **ikon iPhone**.
  Jangan Cmd+R/rebuild lagi di antara dua tahap ini. Pilih piano yang sama lalu
  **Generate drum dari rekaman** sekali dan bagikan laporan + WAV bila selesai.

Membuka ikon saat proses lama masih hidup bukan restart. Proses segar diminta
oleh workflow, bukan dibuktikan sebagai tidak adanya seluruh native residency.
Run generasi setelah preflight bukan controlled cold-cache benchmark.

## Kontrak dan batas

Manifest profil/stage/input/output/binding/checkpoint/presisi diperiksa terhadap
allowlist; tiga file native setiap aset diperiksa SHA-256 sebelum aset itu
dibuka. Input mandiri Float32 nol, kecuali timestep/rope_cos=1 dan dt=0,02.
Tidak ada RNG, feedback antaraset atau kemiripan yang diklaim dengan aktivasi
piano aktual. Output shape/finite diperiksa; **numerical parity NOT EVALUATED**.
Decoder menghasilkan tensor sintetis saja, tidak ditulis atau diputar sebagai WAV.

Maksimal satu scope model/fungsi Swift per aset; target sukses 16 opens, 16
binds, 16 completed calls, 0 hits. Percobaan aset dan completed calls dibedakan
saat gagal. Tidak ada retry, deletion, repair atau AOT dalam mode ini. Default
cache policy tidak diubah; native eviction/cache persistence tidak dijamin.
Guard proses 2000 MiB tetap dan bukan jaminan transient compiler peak.

UI progress singkat, breadcrumb di `last-native-preflight-phase.txt`. Laporan
dan snapshot laporan sebelumnya disimpan di folder `native-preflight-<UUID>`;
checkpoint diperbarui per aset. Terminasi mendadak dapat kehilangan detail aset
yang masih berjalan, tetapi checkpoint terakhir dan breadcrumb dipertahankan.
File live `last-denoiser-probe.txt` diperbarui seperti mode lain, dengan snapshot
laporan sebelumnya disimpan terpisah. Audio/checkpoint/baseline lain tidak ditimpa.

Berhasil 16/16 tidak membuktikan bahwa cache siap untuk generasi selanjutnya
atau bahwa masalah sudah diperbaiki. Actual input, conditioning sebelumnya,
cache/compiler state dan fase 50 langkah berbeda. Produksi tetap 50 langkah,
BF16/FP32, detached copies, APG, batas numerik dan satu budget repair awal-bind
yang sama. Native exporter, SDK/build dan run mode baru belum diverifikasi di
Linux; delapan tes kontrak instrumen dan pemeriksaan sumber lulus.
