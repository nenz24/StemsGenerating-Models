"""Throttled plain-text verification progress that works in notebook subprocess output."""
import time


def size_text(size):
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size / 1024:.1f} KiB" if size >= 1024 else f"{size} B"


class FileProgress:
    def __init__(self, phase, total, interval=2):
        self.phase, self.total, self.interval = phase, total, interval
        self.last = time.monotonic()
        print(f"  {phase}: 0% dari {size_text(total)}", flush=True)

    def __call__(self, count):
        now = time.monotonic()
        if count == self.total or now - self.last >= self.interval:
            percent = 100 * count / self.total if self.total else 100
            print(f"  {self.phase}: {percent:.1f}% ({size_text(count)}/{size_text(self.total)})", flush=True)
            self.last = now
