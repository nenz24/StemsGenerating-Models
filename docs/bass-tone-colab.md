# Satu percobaan bass elektrik petikan di sesi Colab yang sama

**Hasil:** pengguna memilih kandidat Electric Bass. Untuk generasi berikutnya,
gunakan [preset terpilih](electric-bass-preset.md) secara langsung. Prosedur A/B
di bawah disimpan untuk reproduksi percobaan.

Target pengguna: bass elektrik petikan yang jelas dan natural. Hasil CUDA FP32
pertama lebih bersih dari glitch menurut pengguna, tetapi timbrenya terlalu dalam.
Percobaan ini mengganti **caption saja**. Input B 30 detik, Base, seed 0, CFG 7,
50 langkah, FP32 dan offload dipertahankan. Caption dapat mengarahkan karakter,
tetapi belum menjamin timbre akan cocok dengan chunk 0 atau seluruh lagu.
VAE/runtime dapat bervariasi antar-run walau seed generasi sama. Perbandingan
dengan cache ini memilih kandidat kualitas, belum mengisolasi efek caption secara
deterministik. Uji kontrol berulang dapat dilakukan jika hasilnya menjanjikan.

## Pakai runtime dan bobot yang sudah tersedia

Di Mac, pull commit terbaru:

```bash
git pull --rebase --autostash
```

Di **sesi Colab yang menghasilkan CUDA FP32 tadi**, tambahkan satu sel di bawah
sel yang sudah berhasil. Salin kode berikut, jalankan, lalu pilih **satu file**
`scripts/compare_bass_tone.py` dari checkout Mac saat tombol upload muncul.
Tidak perlu membuat ulang ZIP referensi atau menginstal runtime lagi.

`ROOT`, `REFERENCE`, `WORK` dan `FP32_WORKER` berasal dari notebook 09 beserta sel
pemulihan yang sudah berhasil. `WORK` harus menunjuk run CUDA FP32 yang berhasil,
misalnya folder bernama `cuda_fp32_...`, bukan run yang berstatus failed.

```python
import importlib.util, uuid
from pathlib import Path
from google.colab import files
from IPython.display import Audio, display

BASELINE_CUDA = WORK
uploaded = files.upload()  # Pilih scripts/compare_bass_tone.py saja
assert len(uploaded) == 1, "Upload satu file compare_bass_tone.py"
assert Path(next(iter(uploaded))).name.startswith("compare_bass_tone")
tools_dir = ROOT / "runs" / ("tone_tools_" + uuid.uuid4().hex[:12])
tools_dir.mkdir(parents=True)
helper_path = tools_dir / "compare_bass_tone.py"
helper_path.write_bytes(next(iter(uploaded.values())))
spec = importlib.util.spec_from_file_location("uploaded_bass_tone", helper_path)
tone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tone)

TONE_WORK = ROOT / "runs" / ("bass_tone_" + uuid.uuid4().hex[:12])
tone_report = tone.compare(
    ROOT, REFERENCE, BASELINE_CUDA, TONE_WORK,
    execute=True, float32_worker=FP32_WORKER,
)
print("Folder hasil:", TONE_WORK)
for name in ("baseline_listen.wav", "electric_listen.wav",
             "baseline_mix.wav", "electric_mix.wav"):
    print(name)
    display(Audio(filename=str(TONE_WORK / name), normalize=False))
files.download(str(TONE_WORK / "bass_tone_result.zip"))
```

Hanya satu generasi baru dijalankan. Bobot yang ada diperiksa dan dipakai ulang.
File helper disimpan di `runs/`, tanpa mengedit snapshot source dalam ZIP.
Baseline disalin untuk pratinjau dan tidak ditimpa. Sel dapat diulang untuk
membuat direktori baru, tetapi setiap pengulangan menjalankan inferensi lagi.

## Apa yang didengar

- `baseline_listen.wav`: bass CUDA FP32 yang dinilai terlalu dalam.
- `electric_listen.wav`: kandidat baru dengan caption bass elektrik petikan.
- `baseline_mix.wav` dan `electric_mix.wav`: masing-masing bass bersama piano B.

Kedua solo memakai gain tetap 0.7. Kedua mix memakai gain 0.5 per track, tanpa EQ,
pitch shift, saturasi atau normalisasi tiap kandidat. Evaluasi artikulasi petikan,
harmonik, kesesuaian dengan piano dan artefak yang masih terdengar. Spektrum bass
yang rendah sendiri bukan bukti glitch.

Script memeriksa hash B/baseline, manifest bobot, FP32 model/decoder, dan seluruh
parameter hasil. Caption dan lokasi file sumber adalah satu-satunya parameter
yang boleh berbeda. Settings aktual disimpan di `comparison.json`. ZIP hasil
berisi empat audio dan report; tidak menyertakan bobot. Status `generated`
menandakan keluaran valid, bukan penilaian kualitas musik.

Jika sesi lama sudah berakhir, sediakan runtime, referensi dan baseline lengkap
lagi. `cuda_result.zip` saja tidak berisi input piano atau raw stem baseline yang
diperlukan untuk percobaan ini. Setup sesi baru mengikuti notebook 09.

## CLI untuk checkout lengkap

```bash
python3 scripts/compare_bass_tone.py --reference /path/to/reference --baseline /path/to/cuda_fp32_success --out runs/tone_plan
python3 scripts/compare_bass_tone.py --reference /path/to/reference --baseline /path/to/cuda_fp32_success --out runs/tone_real --execute
```

Tanpa `--execute`, hanya plan dibuat. Runner bukan pelatihan atau konversi model.
