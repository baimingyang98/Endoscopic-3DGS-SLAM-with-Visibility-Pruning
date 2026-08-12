"""Rendering-throughput benchmark for a finished SLAM map.

Loads params.npz from one or more experiment directories, re-renders every pose
in the stored trajectory, and reports frames per second. This measures the
*downstream* cost of the map -- the thing a compact map is supposed to buy --
and is independent of the online SLAM loop timings in runtimes.txt.

Two timings are reported per arm:

  raster_ms  rasterizer call only (comparable to published 3DGS FPS numbers)
  total_ms   per-frame pose transform + rendervar assembly + rasterizer call

Both exclude the GPU->CPU copy of the output image, which no real-time consumer
of the map would pay.

Usage
-----
Single arm:

    python scripts/benchmark_fps.py --experiment_dir ./experiments/C3VD_mo15/full/sigmoid_t2_a

Compare arms across every scene in two group dirs, write a CSV:

    python scripts/benchmark_fps.py \\
        --group Baseline=./experiments/C3VD_test/Baseline \\
        --group Full=./experiments/C3VD_mo15/full \\
        --out ./fps_benchmark.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F

from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transform_to_frame_eval,
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

_TENSOR_KEYS = [
    "means3D", "rgb_colors", "unnorm_rotations", "logit_opacities",
    "log_scales", "feature_rest", "cam_unnorm_rots", "cam_trans",
    "deform_offsets",
]


def load_experiment(exp_dir):
    params_path = os.path.join(exp_dir, "params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.npz not found in {exp_dir}")
    data = dict(np.load(params_path, allow_pickle=True))
    params = {}
    for k, v in data.items():
        params[k] = torch.from_numpy(v).float().cuda() if k in _TENSOR_KEYS else v
    return params


def make_camera(params):
    return setup_camera(int(params["org_width"]), int(params["org_height"]),
                        params["intrinsics"], params["w2c"])


def benchmark(exp_dir, warmup=20, repeats=3, include_depth=False):
    """Render the whole trajectory `repeats` times; return the best-of-N timing.

    Best-of-N rather than mean, because Colab GPUs are shared and a preempted
    slice inflates the mean without telling us anything about the map.
    """
    params = load_experiment(exp_dir)
    cam = make_camera(params)
    w2c_t = torch.from_numpy(params["w2c"]).float().cuda()
    n_frames = params["cam_trans"].shape[-1]
    n_gauss = params["means3D"].shape[0]

    def render_once(t, time_raster_only):
        """Returns elapsed seconds for the portion being timed."""
        if not time_raster_only:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., t].detach())
        cam_tran = params["cam_trans"][..., t].detach()
        transformed_pts = transform_to_frame_eval(params, (cam_rot, cam_tran))
        rendervar = transformed_params2rendervar(params, transformed_pts)
        if include_depth:
            depth_var = transformed_params2depthplussilhouette(
                params, w2c_t, transformed_pts)
        if time_raster_only:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.no_grad():
            Renderer(raster_settings=cam)(**rendervar)
            if include_depth:
                Renderer(raster_settings=cam)(**depth_var)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for t in range(min(warmup, n_frames)):
        render_once(t, True)

    best = {}
    for label, raster_only in (("raster", True), ("total", False)):
        runs = []
        for _ in range(repeats):
            runs.append(sum(render_once(t, raster_only) for t in range(n_frames)))
        best[label] = min(runs) / n_frames * 1000.0     # ms/frame

    return dict(scene=Path(exp_dir).name, n_gauss=n_gauss, n_frames=n_frames,
                raster_ms=best["raster"], total_ms=best["total"],
                raster_fps=1000.0 / best["raster"], total_fps=1000.0 / best["total"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", action="append", default=[],
                    help="A single scene directory containing params.npz.")
    ap.add_argument("--group", action="append", default=[],
                    help="NAME=PATH; benchmarks every scene subdirectory of PATH.")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--include_depth", action="store_true",
                    help="Also rasterize the depth+silhouette pass per frame.")
    ap.add_argument("--out", default=None, help="Write per-scene results to CSV.")
    args = ap.parse_args()

    jobs = [("", d) for d in args.experiment_dir]
    for spec in args.group:
        name, _, path = spec.partition("=")
        scenes = sorted(p for p in Path(path).iterdir()
                        if (p / "params.npz").exists())
        if not scenes:
            print(f"[warn] no scenes with params.npz under {path}")
        jobs += [(name, str(p)) for p in scenes]

    if not jobs:
        ap.error("nothing to benchmark: pass --experiment_dir or --group")

    rows = []
    for arm, d in jobs:
        r = benchmark(d, args.warmup, args.repeats, args.include_depth)
        r["arm"] = arm
        rows.append(r)
        print(f"{arm:>12} {r['scene']:<14} {r['n_gauss']:>8} G  "
              f"raster {r['raster_ms']:6.2f} ms ({r['raster_fps']:6.1f} FPS)  "
              f"total {r['total_ms']:6.2f} ms ({r['total_fps']:6.1f} FPS)")

    arms = [a for a in dict.fromkeys(r["arm"] for r in rows) if a]
    if arms:
        print("\n=== per-arm means ===")
        for a in arms:
            sub = [r for r in rows if r["arm"] == a]
            mean = lambda k: sum(r[k] for r in sub) / len(sub)
            print(f"{a:>12}  n={len(sub):2d}  {mean('n_gauss'):9.0f} G  "
                  f"raster {mean('raster_ms'):6.2f} ms ({1000/mean('raster_ms'):6.1f} FPS)  "
                  f"total {mean('total_ms'):6.2f} ms ({1000/mean('total_ms'):6.1f} FPS)")

    if args.out:
        import csv
        cols = ["arm", "scene", "n_gauss", "n_frames",
                "raster_ms", "raster_fps", "total_ms", "total_fps"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
