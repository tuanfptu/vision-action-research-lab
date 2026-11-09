"""Run a command while recording 2-second nvidia-smi telemetry; never overwrite logs."""

import argparse
import csv
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    for name in [args.csv, args.log]:
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        if Path(name).exists():
            raise FileExistsError(name)
    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    subprocess.run(query, check=True, capture_output=True)
    with Path(args.csv).open("x") as csv_file, Path(args.log).open("x") as log:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            ["unix_time", "gpu_util_percent", "memory_used_mib", "temperature_c", "power_w"]
        )
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                measured = subprocess.run(query, check=True, capture_output=True, text=True)
                for gpu in measured.stdout.strip().splitlines():
                    writer.writerow([time.time()] + gpu.split(", "))
                csv_file.flush()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
