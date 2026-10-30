# Catatan model POC

Sesuai permintaan, tidak ada gerbang review lisensi untuk eksperimen ini.
Fokus awal: kualitas pengiring pada rekaman asli, memakai model pretrained.

Sumber kode: ACE-Step 1.5 commit
`ca1e85fe9430179831e6bc6be790c332190a3866`.
Metadata Hugging Face yang diambil 31 Oktober 2026:

- Base: `e432212fec32b8965a14ffa57ae653438d6abd14`.
- XL Base: `220c1166efbdd9583eafcb12eb160594bbfcb241`.
- Komponen utama: `19671f406d603126926c1b7e2adc169acbcade22`.

Revisi, ukuran, SHA-256 LFS dan hash Git blob ada di `configs/assets-*.json`.
Ukuran checkpoint XL sendiri sekitar 19.95 GB pada revisi ini; ukuran file bukan
ukuran RAM/VRAM inferensi. Manifest berisi kode model dan bobot, tanpa menggantinya
secara diam-diam ketika upstream berubah.

Kartu Base dan bundle utama menyatakan MIT, penggunaan hasil secara komersial,
dan data licensed/royalty-free/synthetic. Pernyataan data adalah klaim penerbit,
bukan audit independen. Qwen embedding memiliki metadata Apache 2.0 di repo
asalnya. STAGE/MusicGen/MoisesDB mempunyai batasan lain yang tidak diselesaikan di
POC ini; sebelum rilis produk, tinjau kembali semua komponen dan atribusinya.

Sumber: https://github.com/ace-step/ACE-Step-1.5 ;
https://huggingface.co/ACE-Step/acestep-v15-base ;
https://huggingface.co/ACE-Step/acestep-v15-xl-base ;
https://huggingface.co/ACE-Step/Ace-Step1.5 ;
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B .

Tidak ada bobot model dalam repo bridge, dan tidak ada bobot yang diunduh saat
membangun notebook. Belum ada bukti inferensi Core AI/iPhone atau hasil training.
