# Dari probe CoreAI ke generasi instrumen

ACE pretrained sudah tersedia. Encoder/decoder VAE, semua 24 layer denoiser,
loop empat langkah sintetis dan profiling sudah lolos di iPhone. Input loop
masih tensor sintetis: belum ada pipeline dari rekaman pengguna ke audio pengiring.

Perkiraan saat ini adalah **empat tahap besar** menuju preview pengiring pendek
dari rekaman, dengan beberapa pengujian atau revisi di dalam setiap tahap.
Ini urutan dependensi, bukan janji jumlah run atau tanggal selesai.

| Tahap | Hasil yang harus tersedia | Posisi sekarang |
| --- | --- | --- |
| 1. Jalur eksekusi yang layak | Keputusan eksekusi berdasarkan waktu/memori: first-use vs reuse, backend/precision atau ukuran model jika diperlukan | Selesai untuk keputusan jalur prototipe: kedua uji utama PASS |
| 2. Conditioning dari rekaman dan instrumen tujuan | Piano/gitar pengguna serta prompt menghasilkan context/condition/mask yang benar, mengikuti jalur ACE yang dipin | Gate audio-context dan Qwen/ConditionEncoder PASS secara terpisah untuk fixture piano/preset bass; kanonisasi audio dan tokenisasi di Mac |
| 3. Pipeline lengkap menjadi WAV pendek | Conditioning → sampling dengan jumlah langkah/guidance yang sesuai → VAE decoder → file audio yang bisa didengar; kontrak tensor diperpanjang untuk preview pendek | Komponen dan loop pendek sudah lolos secara terpisah; integrasi audio belum ada |
| 4. Pengiring yang layak didengar | Akor, groove, timing, timbre serta level dinilai terhadap baseline CUDA; lanjut ke tiga instrumen secara berurutan | Baseline CUDA tersedia; evaluasi hasil iPhone belum bisa dilakukan |

Setelah preview itu, lagu panjang/chunking, sambungan antarchunk, ekspor/mixer,
thermal/baterai serta pengalaman aplikasi menjadi pekerjaan lanjutan. Menyelesaikan
empat tahap di atas tidak otomatis memvalidasi lagu 90 detik di iPhone.

Tahap 1 selesai untuk keputusan jalur prototipe: pertahankan model/fungsi per
asset selama sampling, lalu lepas pool Swift sebelum komponen berikutnya.
Kasus empat langkah retained selesai 18,52 detik, dengan langkah 3/4 sekitar
0,50/0,53 detik pada run ini. Monolitik tetap dinonaktifkan. Guard tidak terpicu,
tetapi footprint tersampel bukan seluruh RAM model. Input lebih panjang,
conditioning/codec tambahan dan anggaran produk belum tervalidasi; kompresi atau
perubahan model mungkin diperlukan bila skala berikutnya tidak muat.

Rencana tahap 1 memiliki dua uji utama. [Uji pasangan layer](coreai-pair-reuse.md)
sudah PASS: 1042,53 ms pertama versus rata-rata 8,52 ms untuk run ulang.
Uji kedua [reuse seluruh loop 24 layer](coreai-retained-loop.md) juga PASS dengan
14 open, 14 bind, 42 reuse dan 16 pembandingan.

Kedua gate komponen tahap 2 juga selesai untuk fixture piano/preset bass:
[audio-context](../reviews/coreai-iphone17pro-source-context/README.md) PASS
3 run/12 pembandingan dan [conditioning bass](../reviews/coreai-iphone17pro-bass-conditioning/README.md)
PASS seluruh 40 layer/27 aset/6 pembandingan. **Tahap aktif kini 3: integrasi
sampling dan decoder menjadi WAV pendek.** Condition 117 token dan source 128
latent frame membutuhkan kontrak denoiser yang sesuai; fixture sintetis lama
64 token/25 frame tidak langsung dipakai. Komponen belum dijalankan bersama.
Kanonisasi PCM dan tokenisasi tetap pada host; raw recording import dan prompt
bebas di iPhone belum tervalidasi. Setelah pipeline WAV tersedia, tahap 4 menilai
kualitas musikal dan melanjutkan tiga instrumen. Model dan referensi
baseline dipertahankan; metode MIDI/soundfont tidak menjadi jalur generasi ini.
Target tetap seluruh inference di iPhone melalui CoreAI, record dahulu lalu
generate instrumen selain sumber yang direkam.
