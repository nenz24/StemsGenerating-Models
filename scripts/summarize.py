"""Produce a small, shareable run report without audio, model weights or credentials."""
import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    fields = ["job", "status", "instrument", "condition", "seed", "inference_seconds",
              "device", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
              "mps_current_allocated_bytes", "mps_driver_allocated_bytes", "error"]
    reports = []
    for file in sorted(directory.glob("*/result.json")):
        record = json.loads(file.read_text())
        reports.append(dict(record, job=file.parent.name))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(f'{len(reports)} real job reports; planned jobs are not inference results')
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run)
