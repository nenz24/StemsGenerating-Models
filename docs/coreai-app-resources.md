# Resource app CoreAI sesuai pipeline aktif

Build default tidak lagi menyalin seluruh `ModelCandidates` ke iPhone.
Xcode menjalankan `scripts/stage_coreai_app_models.py` dalam build phase
**Stage active CoreAI models**. Skrip memakai Python standar bawaan developer
tools Mac; tidak memasang Python, PyTorch, checkpoint atau dependensi tambahan.

## Pemakaian biasa

```bash
git pull --ff-only &&
open prototypes/ios27-coreai/CoreAIProbe.xcodeproj
```

Stop app lama, lalu Cmd+R. Tidak perlu menjalankan ulang preparation atau
kompilasi AOT yang sudah selesai. Build phase membersihkan resource lama hanya
di produk build `.app/ModelCandidates`, sehingga tidak wajib Clean Build Folder.
Model di repository Mac, checkpoint cache, working copy AOT dan hasil rekaman
**tidak dihapus atau ditulis ulang**. Pengurangan ini berlaku pada app hasil build;
ruang source/cache Mac tetap dipakai untuk reuse dan resume.

Build log mencatat profil, ukuran file yang disertakan/dikecualikan, jumlah copy
baru dan resource build lama yang dibuang. Copy berikutnya melewati file yang
ukuran, waktu modifikasi dan identitas manifestnya sama. Tidak ada hash ulang
seluruh bobot pada setiap build; validasi checksum runtime tetap berjalan.

## Build script dan akses file di Mac

Target **CoreAIProbe**, untuk Debug dan Release, mengatur
`ENABLE_USER_SCRIPT_SANDBOXING = NO`. Satu-satunya build script target ini
memilih file secara dinamis dari manifest dan profil lokal, membaca metadata
kandidat, serta membuang resource build yang sudah tidak dipakai. Deklarasi
folder input sebelumnya tidak memberi akses baca yang dibutuhkan pada build
Xcode pengguna: Python gagal membuka `source-context/reference.json` dengan
`PermissionError: [Errno 1] Operation not permitted` di dalam `sandbox-exec`.
Pengaturan ini berlaku untuk build script target di Mac; bukan entitlement
atau izin aplikasi iPhone. Project-level default tetap `YES`.

Pembatasan sumber/output di Python tetap berlaku: sumber/cache hanya dibaca,
penulisan dan prune hanya pada direktori produk `.app/ModelCandidates` yang
terpisah dari sumber. Dependency input/output phase tetap dicatat. Pesan
**will be run during every build** adalah informasi karena pemilihan resource
memang diperiksa setiap build; copy file yang tidak berubah tetap dilewati.

## Pilihan resource

Build setting target **COREAI_MODEL_PROFILE** berlaku untuk Debug dan Release.
Default `auto`; tidak perlu mengubahnya untuk menjalankan preview bass.

| Nilai | Resource yang masuk app |
| --- | --- |
| `auto` | Pipeline bass jika ketiga bundle source-context, conditioning dan preview tersedia. Memilih AOT bila folder bundle AOT final ada; selain itu source. Fresh clone memakai `runtime`. |
| `bass-aot` | Source-context, Qwen/ConditionEncoder, 16 aset compiled AOT, serta manifest dan 12 fixture/reference numerik preview. Tanpa salinan 16 model source sampling/decoder. |
| `bass-source` | Source-context, Qwen/ConditionEncoder dan preview source lengkap; tanpa AOT dan probe historis. |
| `runtime` | Tensor probe dan affine native kecil jika tersedia; tanpa model musik. |
| `all` | Seluruh kandidat untuk diagnosis lama; ukuran besar merupakan pilihan eksplisit. |

Menu diagnosis lama hanya muncul dalam profil `all`. Tombol komponen aktif dan
Generate tampil jika bundle terkait disertakan. Metode diagnosis tetap ada di
source Swift, tetapi bobot modelnya tidak menambah ukuran app default.

Bundle AOT yang ada tetapi tidak lengkap/stale menyebabkan build gagal sebelum
resource lama dibuang; tidak diam-diam mengganti mode. Manifest source preview
**dipertahankan byte demi byte**, termasuk hash asli model source yang tidak
ikut disalin. Loader AOT memerlukan identitas manifest ini dan 12 file numerik;
loader tidak membuka model source sampling/decoder pada mode AOT. Semua bobot
source-context dan Qwen/ConditionEncoder masih diperlukan karena keduanya tetap
melakukan inference nyata di iPhone.

Untuk diagnosis source saja, pilih `bass-source` di Build Settings target.
Untuk seluruh probe lama pilih `all`, lalu rebuild. Tidak perlu memindahkan atau
menghapus model di folder source. Perubahan profil juga membuang sisa resource
profil sebelumnya dari produk build, bukan dari repository.

## Ukuran dan status pengujian

Inventaris Linux saat perubahan ini dibuat: folder kandidat 11,441 GB;
mode source terpilih 6,256 GB, sehingga 5,185 GB probe lama tidak ikut app.
Ini jumlah ukuran file, bukan ukuran instalasi/cache iPhone atau penggunaan RAM.
Laporan kompilasi AOT pengguna mencatat 3,830 GB untuk 16 varian h18p.
Dengan input yang setara, isi model mode AOT diperkirakan sekitar 6,59 GB;
**angka app Mac sebenarnya dicetak build phase**, dan dapat berbeda karena
aset lokal serta metadata/filesystem. AOT tetap berukuran besar karena bobot
aktif dibutuhkan; pengurangan resource tidak membuktikan kelayakan memori inference.

Uji Linux memakai fixture non-executable untuk seleksi source/AOT, preservasi
manifest, kegagalan sebelum prune, copy incremental, pergantian profil,
perlindungan source/cache dan parse konfigurasi Xcode. Native Xcode build,
akses build phase setelah perbaikan konfigurasi dan instalasi iPhone masih
perlu diverifikasi di Mac pengguna.
