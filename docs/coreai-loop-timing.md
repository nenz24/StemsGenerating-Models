# Pisahkan biaya loop CoreAI

Baseline empat langkah sudah [PASS di iPhone](../reviews/coreai-iphone17pro-denoiser-loop/README.md),
tetapi totalnya 53,00 detik. Sebanyak 48 interval pasangan layer memakan sekitar
51,32 detik. Interval lama mencampur binding fungsi, validasi, eksekusi dan copy,
sehingga belum menunjukkan operasi mana yang perlu dioptimalkan.

## Jalankan pengukuran

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Build dengan **Cmd+R**, lalu tekan **Profil waktu loop 4 langkah**.
Bagikan laporan lengkap setelah selesai, termasuk `Timing totals`,
`Timing outside-stage ms`, `Largest measured phase`, dan hasil parity.
Tombol uji loop sebelumnya tetap tersedia.

Tidak perlu menyiapkan model, mengunduh bobot, menghapus cache atau mengekspor
ulang. Jadwal, tensor input, referensi, toleransi, kebijakan CPU-only dan urutan
56 asset sama dengan baseline yang sudah lolos. Tidak ada warmup tambahan,
pengulangan tahap, atau optimisasi eksekusi dalam perubahan ini. Keadaan cache
awal, beban sistem dan temperatur tetap dapat berbeda antar percobaan.

## Arti interval

| Field dalam laporan | Yang diukur |
| --- | --- |
| `model-open` | Validasi asset dan `AIModel(contentsOf:options:)` |
| `function-bind` | Pemanggilan sinkron `model.loadFunction(named:)` |
| `input-validation` | Descriptor, statelessness, nama input, bentuk/dtype/layout |
| `backend-await` | Waktu mengelilingi `await function.run(inputs:)` |
| `output-copy` | Validasi output, akses Float, penyalinan dan finite checks, pembuatan NDArray CPU mandiri |
| `run-other` | Selisih interval run+copy lama dengan empat sub-interval di dalamnya; mencakup bookkeeping/batas scope yang belum terisolasi |
| `outside-stage` | Waktu seluruh loop dikurangi interval tahap yang selesai; termasuk baca/hash fixture, update latent, parity, logging/UI waits, pelepasan resource dan tahap gagal bila ada |

`backend-await` bukan waktu kernel murni: penjadwalan async, inisialisasi backend,
paging atau kerja tertunda dapat masuk interval itu. Kerja yang baru terjadi
saat read-back juga dapat muncul di `output-copy`. `function-bind` dapat memuat
pekerjaan internal selain pencarian nama fungsi. Laporan ini mengukur batas
pemanggilan API; ia tidak mengukur residency cache, bandwidth, atau penempatan
accelerator sebenarnya. Jangan menyebut binding atau copy sebagai penyebab
biaya 53 detik sebelum hasil pengukuran tersedia.

Setiap tahap mencetak interval lama dan rincian fase. Laporan juga menjumlahkan
fase per langkah dan untuk seluruh loop. `Largest measured phase` menunjukkan
fase terbesar di antara **tahap yang selesai**, beserta persentase dari total
interval tahap tersebut; persentasenya bukan bagian dari seluruh waktu loop.
Jumlah fase termasuk `run-other` sama dengan interval tahap lama, terlepas dari
pembulatan floating point. `outside-stage` dicatat terpisah agar overhead caller
tidak dianggap sebagai kerja backend. Jika ada kegagalan, jumlah tahap yang
terukur bisa kurang dari 56; waktu tahap yang gagal tidak diatribusikan ke fase.

Model dan fungsi tetap berada dalam scope tahap; hanya output NDArray CPU baru
yang diteruskan. Tidak ada model tambahan yang dipertahankan agar pengukuran
menjadi cepat. Semua 16 pembandingan velocity/latent masih wajib PASS, dengan
target 4/4 langkah dan 56/56 tahap. Guard footprint dan sampling memori tetap
sama, tanpa klaim total RAM model atau peak OS.

Kode pengukuran ini belum dibuild dengan Apple SDK di workspace Linux dan belum
diukur di iPhone. Pemeriksaan host/fixture tidak menguji clock atau runtime
CoreAI. Hasil profiling akan menentukan perubahan berikutnya; belum ada klaim
percepatan atau generasi audio instrumen.
