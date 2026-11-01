# Dari probe CoreAI ke generasi instrumen

ACE pretrained sudah tersedia. Encoder/decoder VAE, semua 24 layer denoiser,
loop empat langkah sintetis dan profiling sudah lolos di iPhone. Input loop
masih tensor sintetis: belum ada pipeline dari rekaman pengguna ke audio pengiring.

Perkiraan saat ini adalah **empat tahap besar** menuju preview pengiring pendek
dari rekaman, dengan beberapa pengujian atau revisi di dalam setiap tahap.
Ini urutan dependensi, bukan janji jumlah run atau tanggal selesai.

| Tahap | Hasil yang harus tersedia | Posisi sekarang |
| --- | --- | --- |
| 1. Jalur eksekusi yang layak | Keputusan eksekusi berdasarkan waktu/memori: first-use vs reuse, backend/precision atau ukuran model jika diperlukan | Uji utama 1/2 (pasangan layer) PASS; uji 2/2 (loop 24 layer dengan reuse) disiapkan |
| 2. Conditioning dari rekaman dan instrumen tujuan | Piano/gitar pengguna serta prompt menghasilkan context/condition/mask yang benar, mengikuti jalur ACE yang dipin | Belum diimplementasikan; tensor conditioning sekarang sintetis |
| 3. Pipeline lengkap menjadi WAV pendek | Conditioning → sampling dengan jumlah langkah/guidance yang sesuai → VAE decoder → file audio yang bisa didengar; kontrak tensor diperpanjang untuk preview pendek | Komponen dan loop pendek sudah lolos secara terpisah; integrasi audio belum ada |
| 4. Pengiring yang layak didengar | Akor, groove, timing, timbre serta level dinilai terhadap baseline CUDA; lanjut ke tiga instrumen secara berurutan | Baseline CUDA tersedia; evaluasi hasil iPhone belum bisa dilakukan |

Setelah preview itu, lagu panjang/chunking, sambungan antarchunk, ekspor/mixer,
thermal/baterai serta pengalaman aplikasi menjadi pekerjaan lanjutan. Menyelesaikan
empat tahap di atas tidak otomatis memvalidasi lagu 90 detik di iPhone.

Tahap 1 masih menentukan kelayakan arsitektur: kasus sintetis empat langkah
membutuhkan sekitar 52 detik pada requested CPU-only. Monolitik sebelumnya
melewati batas memori yang dilaporkan OS. Jika hasil uji memerlukan kompresi,
distilasi atau model lain, ada pekerjaan konversi/pelatihan dan evaluasi kualitas
tambahan. Belum ada bukti bahwa ACE yang sekarang akan memenuhi kebutuhan
waktu/memori produk hanya dengan merangkai komponen yang sudah lolos.

Rencana tahap 1 memiliki dua uji utama. [Uji pasangan layer](coreai-pair-reuse.md)
sudah PASS: 1042,53 ms pertama versus rata-rata 8,52 ms untuk run ulang.
Langkah aktif adalah [reuse seluruh loop 24 layer](coreai-retained-loop.md).
Jika uji kedua berhasil, hasilnya menjadi dasar keputusan eksekusi untuk masuk
conditioning. Kegagalan memori/latensi memerlukan alternatif yang spesifik;
jumlah uji tambahan belum bisa dijamin. Model dan referensi
baseline dipertahankan; metode MIDI/soundfont tidak menjadi jalur generasi ini.
Target tetap seluruh inference di iPhone melalui CoreAI, record dahulu lalu
generate instrumen selain sumber yang direkam.
