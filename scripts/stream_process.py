"""Stream child logs into Jupyter's Python output and keep quiet phases visible."""
import os
import queue
import signal
import subprocess
import threading
import time


def run_logged(command, *, cwd=None, env=None, check=True, heartbeat_seconds=15):
    environment = dict(os.environ if env is None else env)
    environment.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="0",
                       TQDM_MININTERVAL="2")
    environment.pop("TQDM_DISABLE", None)
    stage = next((os.path.basename(str(arg)) for arg in command[1:] if str(arg).endswith(".py")), os.path.basename(str(command[0])))
    started = time.monotonic()
    last_output = started
    print("Memulai proses; output ditampilkan langsung di bawah…", flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=os.name == "posix")
    print(f"Tahap={stage}; PID={process.pid}", flush=True)
    lines = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        eof = False
        warned_quiet = False
        while not eof or process.poll() is None:
            try:
                line = lines.get(timeout=max(heartbeat_seconds, 60) if warned_quiet else heartbeat_seconds)
            except queue.Empty:
                if process.poll() is None:
                    print(f"[Status] {stage} PID={process.pid}; proses masih berjalan, total {time.monotonic()-started:.0f}s; "
                          f"tanpa output baru {time.monotonic()-last_output:.0f}s. "
                          "Ini bukan pengukuran byte unduhan.", flush=True)
                    if time.monotonic() - last_output >= 300 and not warned_quiet:
                        print("[Perlu diperiksa] Tidak ada output selama >=5 menit. Untuk sel unduhan, "
                              "jalankan `python3 scripts/diagnose_download.py` di Terminal repository. "
                              "Jangan anggap heartbeat sebagai bukti transfer masih maju.", flush=True)
                        warned_quiet = True
                continue
            if line is None:
                eof = True
            else:
                print(line, end="", flush=True)
                last_output = time.monotonic()
                warned_quiet = False
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        raise
    finally:
        reader.join(timeout=5)
        process.stdout.close()
    print(f"Proses selesai: exit={returncode}, waktu={time.monotonic()-started:.1f}s", flush=True)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)
