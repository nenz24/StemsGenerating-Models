# Preset terpilih: Electric Bass

Pengguna memilih hasil `electric_listen.wav` / `electric_mix.wav` dari
`bass_tone_result.zip` pada 1 November 2026 (Asia/Jakarta). Pilihan ini menjadi
acuan timbre POC. Konfigurasi disimpan di
[`configs/electric-bass.json`](../configs/electric-bass.json).

| Pengaturan | Nilai |
| --- | --- |
| Model / task | ACE-Step 1.5 Base / LEGO |
| Karakter | Fingerstyle electric bass, petikan natural |
| Runtime | CUDA, FP32, CPU offload |
| Seed / CFG / langkah | 0 / 7 / 50 |
| Input yang sudah dinilai | Piano B, 30 detik |

Report hasil cocok dengan caption yang diminta; hanya caption berubah dari
baseline CUDA, config lainnya sama. Model/decoder capture memakai FP32.
Hash stem pilihan yang tercatat adalah
`aedb3a3e559697cb86b27a73edae4b14fa4cb0c9a72c264be23ff1d771234d3f`.
ZIP feedback tidak menyertakan raw stem tersebut; hash ini diverifikasi konsisten
antar-report, bukan dihitung ulang dari file pratinjau yang memakai gain 0.7.
Preferensi timbre berasal dari pendengaran pengguna.

## Generasi berikutnya langsung memakai pilihan ini

Hasil yang sudah dipilih tetap menjadi referensi. Untuk klip berikutnya, entry
point `generate` tidak memerlukan baseline A/B. Ia mendukung satu klip prepared
10–30 detik dan memeriksa caption, seed, CFG, langkah serta FP32 aktual.

Di checkout lengkap:

```bash
python3 scripts/compare_bass_tone.py --prepared /path/to/prepared --out runs/electric_plan
python3 scripts/compare_bass_tone.py --prepared /path/to/prepared --out runs/electric_real --execute
```

Untuk sesi Colab lama, pull di Mac dan upload satu helper terbaru
`scripts/compare_bass_tone.py` melalui sel berikut. `FP32_WORKER`, `ROOT` dan
`REFERENCE` berasal dari notebook 09 beserta pemulihan yang sudah berhasil.
Contoh `PREPARED` menunjuk B yang sudah didengar; ganti dengan folder prepared
klip berikutnya. `GENERATE = False` hanya membuat plan.

```python
import importlib.util, uuid
from pathlib import Path
from google.colab import files
from IPython.display import Audio, display

uploaded = files.upload()  # Pilih scripts/compare_bass_tone.py terbaru
assert len(uploaded) == 1 and Path(next(iter(uploaded))).name.startswith("compare_bass_tone")
tools_dir = ROOT / "runs" / ("electric_tools_" + uuid.uuid4().hex[:12])
tools_dir.mkdir(parents=True)
helper = tools_dir / "compare_bass_tone.py"
helper.write_bytes(next(iter(uploaded.values())))
spec = importlib.util.spec_from_file_location("selected_electric_bass", helper)
electric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(electric)

PREPARED = REFERENCE / "prepared"  # B lama; ganti untuk klip berikutnya
GENERATE = False  # True untuk satu inferensi baru
ELECTRIC_OUT = ROOT / "runs" / ("electric_bass_" + uuid.uuid4().hex[:12])
report = electric.generate(
    ROOT, PREPARED, ELECTRIC_OUT,
    execute=GENERATE, float32_worker=FP32_WORKER,
)
print("Status:", report["status"], "| Folder:", ELECTRIC_OUT)
if GENERATE:
    for name in ("electric_listen.wav", "electric_mix.wav"):
        display(Audio(filename=str(ELECTRIC_OUT / name), normalize=False))
    files.download(str(ELECTRIC_OUT / "electric_bass_result.zip"))
```

Helper tunggal membawa nilai preset yang sama untuk sesi lama yang belum memiliki
file JSON konfigurasi baru. Runtime dan bobot yang ada dipakai ulang. Pratinjau
bass memakai gain tetap 0.7; mix memakai gain 0.5 per track, seperti hasil yang
dipilih. File lama tetap utuh dan setiap generasi memakai direktori baru.

## Batas validasi

Pilihan ini sudah dinilai pada satu cuplikan 30 detik. Konsistensi karakter,
harmoni dan artefak pada seluruh lagu tiga menit masih perlu diuji. Caption yang
sama tidak mengunci timbre antar-chunk, dan seed sama tidak menjamin sampling VAE
identik antar-run.

Runner `run_song.py` lama belum menerapkan parameter caption/precision preset
ini. Ia menolak profil tersebut agar tidak diam-diam menghasilkan bass generik.
Gunakan entry point klip di atas untuk preset terpilih. Penerapan ke seluruh lagu
merupakan tahap berikutnya setelah pilihan timbre ini.
