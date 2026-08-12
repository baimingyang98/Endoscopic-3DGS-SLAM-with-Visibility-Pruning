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


def make_runner(exp_dir, include_depth=False):
    """Load a map onto the GPU and return (info, pass_fn).

    pass_fn(raster_only) renders the whole stored trajectory once and returns
    the elapsed seconds for the timed portion.
    """
    params = load_experiment(exp_dir)
    cam = make_camera(params)
    w2c_t = torch.from_numpy(params["w2c"]).float().cuda()
    n_frames = params["cam_trans"].shape[-1]
    n_gauss = params["means3D"].shape[0]

    def render_once(t, raster_only):
        if not raster_only:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., t].detach())
        cam_tran = params["cam_trans"][..., t].detach()
        transformed_pts = transform_to_frame_eval(params, (cam_rot, cam_tran))
        rendervar = transformed_params2rendervar(params, transformed_pts)
        if include_depth:
            depth_var = transformed_params2depthplussilhouette(
                params, w2c_t, transformed_pts)
        if raster_only:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.no_grad():
            Renderer(raster_settings=cam)(**rendervar)
            if include_depth:
                Renderer(raster_settings=cam)(**depth_var)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def warm(n):
        for t in range(min(n, n_frames)):
            render_once(t, True)

    def full_pass(raster_only):
        return sum(render_once(t, raster_only) for t in range(n_frames))

    info = dict(scene=Path(exp_dir).name, n_gauss=n_gauss, n_frames=n_frames)
    return info, warm, full_pass


def benchmark_scene(arm_dirs, warmup=20, repeats=3, include_depth=False):
    """Benchmark every arm on one scene, interleaved.

    All arms are resident on the GPU at once and their passes are round-robined
    with the order reversed on alternating repeats. Running one arm to
    completion before starting the next confounds the comparison with whatever
    the GPU clock is doing over the session -- on a shared, thermally throttled
    T4 that drift is the same order as the effect being measured, and it lands
    entirely on the arm that ran second.

    After the round-robin, the first arm is re-timed to measure the residual
    drift across the scene, reported as `drift_pct`.
    """
    runners = [(name, make_runner(d, include_depth)) for name, d in arm_dirs]
    for _, (_, warm, _) in runners:
        warm(warmup)

    results = {}
    for label, raster_only in (("raster", True), ("total", False)):
        best = {name: float("inf") for name, _ in runners}
        for rep in range(repeats):
            order = runners if rep % 2 == 0 else runners[::-1]
            for name, (_, _, full_pass) in order:
                best[name] = min(best[name], full_pass(raster_only))
        results[label] = best

    first_name, (_, _, first_pass) = runners[0]
    redo = min(first_pass(True) for _ in range(repeats))
    drift_pct = 100.0 * (redo - results["raster"][first_name]) / results["raster"][first_name]

    out = []
    for name, (info, _, _) in runners:
        nf = info["n_frames"]
        r = results["raster"][name] / nf * 1000.0
        t = results["total"][name] / nf * 1000.0
        out.append(dict(arm=name, **info, raster_ms=r, total_ms=t,
                        raster_fps=1000.0 / r, total_fps=1000.0 / t,
                        drift_pct=drift_pct))
    return out


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

    # An explicit dir with no --group gets its arm name from its parent, so
    # ./maps/Baseline/sigmoid_t2_a and ./maps/Full/sigmoid_t2_a pair up.
    jobs = [(Path(d).parent.name, d) for d in args.experiment_dir]
    for spec in args.group:
        name, _, path = spec.partition("=")
        scenes = sorted(p for p in Path(path).iterdir()
                        if (p / "params.npz").exists())
        if not scenes:
            print(f"[warn] no scenes with params.npz under {path}")
        jobs += [(name, str(p)) for p in scenes]

    if not jobs:
        ap.error("nothing to benchmark: pass --experiment_dir or --group")

    by_scene = {}
    for arm, d in jobs:
        by_scene.setdefault(Path(d).name, []).append((arm, d))

    rows = []
    for scene, arm_dirs in by_scene.items():
        for r in benchmark_scene(arm_dirs, args.warmup, args.repeats,
                                 args.include_depth):
            rows.append(r)
            print(f"{r['arm']:>12} {r['scene']:<14} {r['n_gauss']:>8} G  "
                  f"raster {r['raster_ms']:6.2f} ms ({r['raster_fps']:6.1f} FPS)  "
                  f"total {r['total_ms']:6.2f} ms ({r['total_fps']:6.1f} FPS)")
        print(f"{'':>12} {scene:<14} clock drift over scene: "
              f"{rows[-1]['drift_pct']:+.1f}%")

    arms = [a for a in dict.fromkeys(r["arm"] for r in rows) if a]
    if arms:
        print("\n=== per-arm means ===")
        for a in arms:
            sub = [r for r in rows if r["arm"] == a]
            mean = lambda k: sum(r[k] for r in sub) / len(sub)
            print(f"{a:>12}  n={len(sub):2d}  {mean('n_gauss'):9.0f} G  "
                  f"raster {mean('raster_ms'):6.2f} ms ({1000/mean('raster_ms'):6.1f} FPS)  "
                  f"total {mean('total_ms'):6.2f} ms ({1000/mean('total_ms'):6.1f} FPS)")
        if len(arms) > 1:
            print("\n=== paired per-scene deltas vs %s ===" % arms[0])
            ref = {r["scene"]: r for r in rows if r["arm"] == arms[0]}
            for a in arms[1:]:
                sub = [r for r in rows if r["arm"] == a and r["scene"] in ref]
                faster = sum(1 for r in sub if r["raster_ms"] < ref[r["scene"]]["raster_ms"])
                dg = [100 * (r["n_gauss"] - ref[r["scene"]]["n_gauss"])
                      / ref[r["scene"]]["n_gauss"] for r in sub]
                dr = [100 * (r["raster_ms"] - ref[r["scene"]]["raster_ms"])
                      / ref[r["scene"]]["raster_ms"] for r in sub]
                print(f"{a:>12}  mean dG {sum(dg)/len(dg):+6.1f}%  "
                      f"mean d_raster {sum(dr)/len(dr):+6.1f}%  "
                      f"faster on {faster}/{len(sub)} scenes")

    if args.out:
        import csv
        cols = ["arm", "scene", "n_gauss", "n_frames",
                "raster_ms", "raster_fps", "total_ms", "total_fps", "drift_pct"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
