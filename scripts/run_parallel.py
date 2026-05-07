"""
Parallel scene runner - shards work across machines.

Usage examples:

    # On Colab (runs scenes 5-9 of c3vd):
    python scripts/run_parallel.py --config configs/c3vd/c3vd_best.py --scenes 5-9

    # On local GPU (runs scenes 0-4 of c3vd):
    python scripts/run_parallel.py --config configs/c3vd/c3vd_best.py --scenes 0-4

    # On a single machine, all scenes sequentially:
    python scripts/run_parallel.py --config configs/c3vd/c3vd_best.py --scenes 0-9

    # Run only specific scenes (for retries):
    python scripts/run_parallel.py --config configs/c3vd/c3vd_best.py --scenes 2,5,7
"""
import argparse
import os
import subprocess
import sys
import time


def parse_scenes(s):
    """Parse '0-4' or '0,2,5' into list of indices."""
    indices = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a), int(b) + 1))
        else:
            indices.append(int(part))
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config .py file")
    parser.add_argument("--scenes", required=True, help="Scene indices (e.g., '0-4' or '0,2,5')")
    parser.add_argument("--log_dir", default=None, help="Directory for log files")
    parser.add_argument("--cwd", default=".", help="Working directory")
    args = parser.parse_args()

    indices = parse_scenes(args.scenes)
    print(f"Will run {len(indices)} scenes: {indices}\n")

    results = []
    for idx in indices:
        env = os.environ.copy()
        env["SCENE_NUM"] = str(idx)

        log_dir = args.log_dir or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"scene_{idx}.log")

        print(f"\n{'='*60}")
        print(f"Scene index {idx}  (log: {log_file})")
        print(f"{'='*60}")

        start = time.time()
        with open(log_file, "w") as log:
            proc = subprocess.run(
                [sys.executable, "scripts/main.py", args.config],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=args.cwd,
            )
        elapsed = (time.time() - start) / 60
        status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        results.append((idx, status, elapsed))
        print(f"  {status}  |  {elapsed:.1f} min")

        if proc.returncode != 0:
            with open(log_file) as f:
                tail = f.readlines()[-15:]
            for line in tail:
                print(f"    {line.rstrip()}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for idx, status, elapsed in results:
        print(f"  scene {idx}  |  {status:10s}  |  {elapsed:.1f} min")


if __name__ == "__main__":
    main()
