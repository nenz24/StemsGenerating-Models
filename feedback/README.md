# Feedback dari JupyterLab lokal

Buat `feedback/hardware.json` dari laporan diagnostik setelah meninjau isinya.
Laporan berisi OS, Python, nama GPU/VRAM/driver, tool tersedia dan ruang disk;
tidak memuat environment/token. Hapus path pribadi bila tidak ingin dibagikan.

Bila ada error, tambahkan `feedback/runtime-error.txt` berisi perintah/sel,
pesan kesalahan dan traceback yang relevan. Jangan salin `.env`, token,
credential, audio pribadi, atau notebook dengan audio tertanam.

Sesudah generasi nyata, boleh salin `summary.csv` dari folder run ke
`feedback/summary.csv`. Rencana berstatus planned bukan hasil inferensi.
Commit dan push berkas feedback itu bila ingin agen membacanya pada giliran
berikutnya; tidak ada pengiriman otomatis dari notebook.
