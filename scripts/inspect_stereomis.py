"""
Quick inspection script for StereoMIS dataset.

Run after downloading one StereoMIS sequence to:
1. Verify the file structure matches what the loader expects
2. Print actual filenames, intrinsics, depth ranges
3. Help fill in the correct values in configs/data/stereomis.yaml

Usage:
    python scripts/inspect_stereomis.py /path/to/StereoMIS/P1
"""
import os
import sys
import glob

import numpy as np
from PIL import Image


def inspect(seq_dir):
    print(f"Inspecting: {seq_dir}\n")
    print(f"Top-level contents:")
    for item in sorted(os.listdir(seq_dir)):
        path = os.path.join(seq_dir, item)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  [DIR]  {item}/  ({count} items)")
        else:
            size_kb = os.path.getsize(path) / 1024
            print(f"  [FILE] {item}  ({size_kb:.1f} KB)")

    # Find image folders
    print("\n--- Color images ---")
    for subdir in ["image_0", "left", "rgb", "color", "frames"]:
        path = os.path.join(seq_dir, subdir)
        if os.path.isdir(path):
            files = sorted(glob.glob(f"{path}/*"))
            print(f"  Found {len(files)} files in {subdir}/")
            if files:
                print(f"    First: {os.path.basename(files[0])}")
                print(f"    Last:  {os.path.basename(files[-1])}")
                img = Image.open(files[0])
                print(f"    Image size: {img.size}, mode: {img.mode}")

    print("\n--- Depth/Disparity ---")
    for subdir in ["depth", "disparity", "depth_left"]:
        path = os.path.join(seq_dir, subdir)
        if os.path.isdir(path):
            files = sorted(glob.glob(f"{path}/*"))
            print(f"  Found {len(files)} files in {subdir}/")
            if files:
                ext = os.path.splitext(files[0])[1]
                print(f"    Extension: {ext}")
                if ext.lower() in [".png", ".tiff", ".tif"]:
                    img = np.array(Image.open(files[0]))
                    print(f"    Shape: {img.shape}, dtype: {img.dtype}")
                    print(f"    Range: [{img.min()}, {img.max()}], mean: {img.mean():.2f}")

    print("\n--- Pose/Trajectory files ---")
    for fname in ["groundtruth.txt", "pose.txt", "trajectory.txt", "poses.txt"]:
        path = os.path.join(seq_dir, fname)
        if os.path.isfile(path):
            print(f"  Found: {fname}")
            with open(path) as f:
                lines = f.readlines()
            print(f"    Total lines: {len(lines)}")
            non_comment = [l for l in lines if not l.startswith("#")]
            print(f"    Non-comment lines: {len(non_comment)}")
            if non_comment:
                print(f"    First line: {non_comment[0].strip()[:120]}")
                print(f"    Tokens per line: {len(non_comment[0].split())}")

    print("\n--- Calibration ---")
    for fname in ["calibration.yaml", "calibration.txt", "calib.yaml", "intrinsics.yaml"]:
        path = os.path.join(seq_dir, fname)
        if os.path.isfile(path):
            print(f"  Found: {fname}")
            with open(path) as f:
                print(f.read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_stereomis.py /path/to/StereoMIS/P1")
        sys.exit(1)
    inspect(sys.argv[1])
