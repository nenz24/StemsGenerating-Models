# Electric Bass: seluruh lagu dalam satu panggilan

ACE-Step mendokumentasikan durasi 10–600 detik. Batas 10/30 detik sebelumnya
adalah batas bridge untuk diagnosis dan konteks Mac, bukan batas arsitektur model.
Lihat [duration guidelines upstream](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/docs/en/INFERENCE.md#3-duration-guidelines).

Runner `scripts/electric_full_song.py` menerima satu file atau cuplikan 10–600
detik, memakai preset Electric Bass yang sudah dipilih, dan memanggil model
**sekali untuk seluruh konteks**. Ia tidak membuat chunk lagu atau stitching.
VAE boleh memakai tiling/offload internal; proses itu berbeda dari generasi
chunk musik secara independen.

Tidak diperlukan pelatihan model baru sebelum percobaan ini. Pada T4 milik
pengguna, run 30 detik FP32 memakai peak allocated sekitar 13.03 GiB dari 14.56
GiB VRAM yang dilaporkan. Fit durasi lebih panjang belum diukur. Besar memori
tidak diasumsikan bertambah secara linear terhadap durasi.

Urutan praktis untuk T4: uji **60 detik satu panggilan**, lalu jika memori dan
kualitas memadai, coba **seluruh tiga menit satu panggilan**. Uji 60 detik tidak
akan disambung menjadi hasil lagu penuh. Seluruh lagu tetap boleh langsung dicoba
dengan `duration=None`; belum ada jaminan akan muat pada T4 FP32.

## Sesi Colab yang masih aktif

Jika `ROOT` belum terdefinisi, sel di bawah mencarinya dari checkout pada VM,
tanpa memerlukan variabel dari notebook sebelumnya. Upload yang sudah ada di
memori atau disk dipakai kembali. Ganti seluruh sel upload lama dengan versi ini.
Jika ada beberapa checkout yang siap, pilih path melalui `ROOT_HINT`.

Di Mac, pull perubahan terbaru:

```bash
git pull --rebase --autostash
```

Di notebook Colab lama yang sudah memiliki runtime dan bobot, tambahkan sel
berikut. Saat upload, pilih **dua file sekaligus**:

1. `scripts/electric_full_song.py` terbaru dari checkout Mac.
2. Audio piano asli sepanjang lagu, misalnya WAV/MP3/M4A/FLAC.

`REFERENCE/prepared/source.wav` dari uji lama hanya berisi B 30 detik. Gunakan
rekaman asli untuk lagu penuh. `ROOT` adalah folder source yang ditemukan. Helper
disimpan di `runs/` tanpa mengedit snapshot source, memasang runtime baru, atau
mengunduh ulang bobot.

```python
import importlib.util, uuid
from pathlib import Path
from google.colab import files

def discover_root(base, preferred=None):
    """Find a bridge checkout without notebook globals or recursively scanning model caches."""
    base = Path(base).resolve()
    markers = ("configs/assets-base.json", "tools/stem_eval/run.py", "scripts/setup_runtime.py")
    def valid(path):
        return all((path / name).is_file() for name in markers)
    if preferred is not None:
        path = Path(preferred).expanduser().resolve()
        if not valid(path):
            raise RuntimeError(f"ROOT yang dipilih bukan checkout bridge: {path}")
        return path
    candidates = set()
    for folder in dict.fromkeys((base, base.parent)):
        for path in (folder, folder / "source", folder / "StemsGenerating-Models",
                     *folder.glob("songmap_cuda_*/source")):
            if valid(path): candidates.add(path.resolve())
    ready = [path for path in candidates
             if (path / "vendor/ACE-Step-1.5/.venv/bin/python").is_file()
             and (path / "vendor/ACE-Step-1.5/checkpoints/acestep-v15-base/model.safetensors").is_file()]
    choices = sorted(ready or candidates)
    if len(choices) > 1:
        raise RuntimeError("Ada beberapa checkout bridge. Isi ROOT_HINT dengan salah satu path ini: "
                           + ", ".join(map(str, choices)))
    return choices[0] if choices else None

ROOT_HINT = globals().get("ROOT")  # Isi Path("/content/.../source") jika ada beberapa checkout
ROOT = discover_root(Path.cwd(), ROOT_HINT)
if ROOT is None and Path.cwd() != Path("/content") and Path("/content").is_dir():
    ROOT = discover_root(Path("/content"))
if ROOT is None:
    raise RuntimeError("Checkout bridge tidak ditemukan pada VM ini. Buka notebooks/10_colab_electric_full_song.ipynb untuk setup mandiri memakai cuda_reference_*.zip. Upload helper dan audio saja belum memasang runtime/bobot.")

# Pakai kembali upload dari sel yang gagal, atau file yang masih ada di VM.
def input_names(payload):
    scripts = [n for n in payload if Path(n).name.startswith("electric_full_song") and Path(n).suffix == ".py"]
    audios = [n for n in payload if Path(n).suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".aac", ".aif", ".aiff", ".ogg"}]
    return scripts, audios
uploaded = dict(globals().get("uploaded") or {})
packaged_helper = ROOT / "scripts/electric_full_song.py"
if not input_names(uploaded)[0] and packaged_helper.is_file():
    uploaded["electric_full_song.py"] = packaged_helper.read_bytes()
scripts, audios = input_names(uploaded)
if len(scripts) != 1 or len(audios) != 1:
    local_helpers = list(Path.cwd().glob("electric_full_song*.py"))
    local_audio = [p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".aac", ".aif", ".aiff", ".ogg"}]
    helper_file = local_helpers[0] if len(local_helpers) == 1 else (packaged_helper if packaged_helper.is_file() else None)
    if helper_file is not None and len(local_audio) == 1:
        uploaded = {helper_file.name: helper_file.read_bytes(), local_audio[0].name: local_audio[0].read_bytes()}
    else:
        uploaded = files.upload()
        if not input_names(uploaded)[0] and packaged_helper.is_file():
            uploaded["electric_full_song.py"] = packaged_helper.read_bytes()
scripts, audios = input_names(uploaded)
assert len(scripts) == 1 and len(audios) == 1, "Pilih satu helper electric_full_song.py dan satu audio asli"
tools_dir = ROOT / "runs" / ("full_song_tools_" + uuid.uuid4().hex[:12])
tools_dir.mkdir(parents=True)
helper = tools_dir / "electric_full_song.py"
helper.write_bytes(uploaded[scripts[0]])
SOURCE_PATH = tools_dir / ("input" + Path(audios[0]).suffix.lower())
SOURCE_PATH.write_bytes(uploaded[audios[0]])
spec = importlib.util.spec_from_file_location("electric_single_pass", helper)
fullsong = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fullsong)
print("ROOT:", ROOT)
print("Audio siap:", SOURCE_PATH)
```

Sel generasi terpisah. Mulai dengan `DURATION = 60`; sesudah hasil diperiksa,
ubah menjadi `DURATION = None` dan jalankan **sel ini saja** untuk seluruh lagu.
Audio/helper dan cache tidak perlu diupload ulang. Masing-masing run mempunyai
folder baru. `GENERATE = False` dapat dipakai untuk memeriksa plan tanpa GPU.

```python
from IPython.display import Audio, display

DURATION = 60  # None = seluruh file, maksimum 600 detik
START = 0
GENERATE = True
FULL_OUT = ROOT / "runs" / ("electric_single_pass_" + uuid.uuid4().hex[:12])
report = fullsong.generate(
    ROOT, SOURCE_PATH, FULL_OUT,
    start=START, duration=DURATION, execute=GENERATE,
)
print("Status:", report["status"], "| Folder:", FULL_OUT)
if GENERATE:
    print("Peak VRAM allocated (GiB):", report["peak_cuda_allocated_bytes"] / 2**30)
    for name in ("electric_preview.mp3", "mix_preview.mp3"):
        print(name)
        display(Audio(filename=str(FULL_OUT / name), normalize=False))
```

Unduh hasil sebelum sesi Colab berakhir. Sel ini juga dapat dijalankan sendiri
setelah error untuk mengambil report yang tersedia:

```python
for name in ("full_song_result.zip", "full_song_feedback.zip"):
    if (FULL_OUT / name).is_file():
        files.download(str(FULL_OUT / name))
```

- `full_song_result.zip`: bass WAV, pratinjau bass WAV gain 0.7, mix piano+bass
  WAV gain 0.5 per track, serta report. Simpan ini sebagai hasil utama.
- `full_song_feedback.zip`: report dan dua pratinjau MP3 192 kbps untuk diunggah
  kembali ke chat. Paket feedback tidak menyertakan bobot atau WAV besar.

Jika OOM atau upstream memotong durasi, status tetap failed; report menyimpan
error dan runner tidak diam-diam mengganti mode menjadi chunk. GPU lebih besar
dapat dicoba jika tersedia di akun. Report failure tetap bisa diunduh melalui
sel terakhir. Keberhasilan kode/plan bukan bukti kualitas musik atau fit GPU.

## Kernel atau VM baru: notebook mandiri

Gunakan [`10_colab_electric_full_song.ipynb`](../notebooks/10_colab_electric_full_song.ipynb)
untuk alur lengkap yang tidak bergantung pada notebook sebelumnya. Pilih GPU,
lalu Run All. Notebook mencari checkout yang ada. Jika tidak ada, ia meminta
`cuda_reference_*.zip` hasil exporter Mac untuk menyediakan source bridge.
ZIP hasil seperti `cuda_result.zip` atau `bass_tone_result.zip` tidak berisi source
yang dibutuhkan. Notebook tidak menjalankan ulang baseline B atau A/B.
Bila ingin mempertahankan file pada VM yang sedang dipakai, salin dan jalankan
sel-sel notebook 10 pada notebook yang sama. Membuka notebook baru dapat memakai
VM berbeda.

Runtime yang masih lengkap dan bobot yang tersedia dipakai kembali. Runtime
parsial diselesaikan memakai installer pinned, termasuk dukungan paket source
lama yang masih memakai ensurepip. Bobot yang belum ada diunduh dengan progress.
Helper yang sudah disertakan dalam paket tidak perlu diupload terpisah.

Variabel kernel yang hilang tidak membuktikan bahwa cache hilang. Namun VM baru
tidak otomatis memiliki folder/cache VM lama: bobot sekitar 13.86 GiB perlu
disediakan lagi jika belum ada. Mengupload helper Python dan audio saja belum
menyediakan checkout bridge, runtime atau bobot model.

## Colab atau JupyterLab

Colab bukan syarat model. Runner yang sama dapat dipanggil dari JupyterLab
dengan kernel di server NVIDIA/CUDA dan runtime/bobot yang sesuai. Antarmuka
notebook tidak menentukan jenis akselerator.

JupyterLab lokal pada Mac Apple Silicon menjalankan MPS/MLX, bukan CUDA. Jalur
Mac sebelumnya masih memiliki glitch dan belum divalidasi menghasilkan kualitas
preset yang sama. Untuk uji panjang saat ini, gunakan Colab yang sudah berhasil
atau JupyterLab pada server CUDA. Mac tetap dapat menyiapkan audio dan memutar/
mengevaluasi hasil. Port dan validasi inferensi Mac merupakan pekerjaan terpisah.

## Durasi dan pengaturan

Preset tetap Base / LEGO, CUDA FP32, seed 0, CFG 7, 50 langkah, CPU offload,
caption electric bass yang dipilih, dengan audio-only conditioning.
Tidak ada pengisian BPM/key/chord palsu. Panjang output diperiksa sehingga hasil
yang terpotong tidak ditandai sukses.

Untuk panjang file pecahan detik, input model diberi padding diam di akhir
hingga detik bulat berikutnya, kurang dari satu detik. Padding ini dicatat dan
dibuang dari ekspor; bass/mix akhir memiliki jumlah frame yang persis sama dengan
audio sumber yang didekode. Ini tidak memecah konteks lagu.

Runner memakai API engine yang sudah ada dengan batas ekspor khusus 600 detik,
tanpa mengubah batas CLI diagnosis lama atau source ACE-Step pinned. Caption
diterapkan sebelum params dibuat dan FP32 sebelum bobot dimuat. Report merekam
parameter aktual, capture decoder, hash input/output, waktu inferensi dan peak
VRAM. Konsistensi timbre/harmoni selama lagu tetap perlu dinilai lewat audio.

## CLI pada checkout lengkap

```bash
# Plan seluruh lagu tanpa memuat model
python3 scripts/electric_full_song.py --source /path/to/piano.wav --out runs/full_plan
# 60 detik, satu panggilan
python3 scripts/electric_full_song.py --source /path/to/piano.wav --duration 60 --out runs/full_60 --execute
# Seluruh file, satu panggilan, sampai 600 detik
python3 scripts/electric_full_song.py --source /path/to/piano.wav --out runs/full_real --execute
```

Gunakan direktori baru setiap percobaan. Runner ini tidak melanjutkan setengah
proses diffusion yang terhenti dan tidak mengklaim hasil GPU sudah diuji di chat.
