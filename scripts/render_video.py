"""
Render comparison videos: GT vs baseline vs ours.

Loads a saved experiment (params.npz from a completed SLAM run), re-renders
every frame using the trained Gaussian map + estimated camera trajectory,
optionally composites against a ground-truth video and additional experiment
directories side-by-side, and encodes as MP4.

Usage examples
--------------

Single experiment, just render the whole trajectory:

    python scripts/render_video.py \\
        --experiment_dir ./experiments/C3VD_best/sigmoid_t3_a \\
        --gt_dir         ./data/C3VD/sigmoid_t3_a \\
        --output         ./videos/sigmoid_t3_a_ours.mp4

Side-by-side GT | baseline | ours:

    python scripts/render_video.py \\
        --gt_dir          ./data/C3VD/sigmoid_t3_a \\
        --experiment_dir  ./experiments/C3VD_parity/sigmoid_t3_a \\
        --experiment_dir  ./experiments/C3VD_best/sigmoid_t3_a \\
        --label           Baseline \\
        --label           Ours \\
        --output          ./videos/sigmoid_t3_a_compare.mp4 \\
        --include_depth

Batch all 10 C3VD scenes into one video each:

    python scripts/render_video.py --batch_c3vd \\
        --baseline_group ./experiments/C3VD_parity \\
        --ours_group     ./experiments/C3VD_best \\
        --gt_root        ./data/C3VD \\
        --output_dir     ./videos
"""
import argparse
import os
import sys
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transform_to_frame_eval,
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


# ============================================================
# Helpers
# ============================================================

def load_experiment(exp_dir):
    """Load params.npz and convert numpy arrays back to torch tensors on cuda."""
    params_path = os.path.join(exp_dir, "params.npz")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.npz not found in {exp_dir}")

    data = dict(np.load(params_path, allow_pickle=True))

    # Trainable Gaussian + camera params -> torch tensors
    keys_to_tensor = [
        "means3D", "rgb_colors", "unnorm_rotations", "logit_opacities",
        "log_scales", "feature_rest", "cam_unnorm_rots", "cam_trans",
        "deform_offsets",
    ]
    params = {}
    for k, v in data.items():
        if k in keys_to_tensor:
            params[k] = torch.from_numpy(v).float().cuda()
        else:
            params[k] = v
    return params


def make_camera(params):
    """Build a GaussianRasterizationSettings using saved intrinsics + image size."""
    intrinsics = params["intrinsics"]   # 3x3 numpy
    w2c = params["w2c"]                 # 4x4 numpy
    H = int(params["org_height"])
    W = int(params["org_width"])
    return setup_camera(W, H, intrinsics, w2c)


def render_frame(params, cam, time_idx):
    """Render the scene from the camera pose stored at `time_idx`. Returns (rgb, depth)."""
    cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
    cam_tran = params["cam_trans"][..., time_idx].detach()
    transformed_pts = transform_to_frame_eval(params, (cam_rot, cam_tran))

    rendervar = transformed_params2rendervar(params, transformed_pts)
    w2c_t = torch.from_numpy(params["w2c"]).float().cuda()
    depth_sil_var = transformed_params2depthplussilhouette(params, w2c_t, transformed_pts)

    with torch.no_grad():
        out = Renderer(raster_settings=cam)(**rendervar)
        # Robust unpack: stock returns 3, patched returns 4
        if len(out) >= 4:
            im = out[0]
        elif len(out) == 3:
            im = out[0]
        else:
            raise RuntimeError(f"Unexpected renderer output count: {len(out)}")

        depth_out = Renderer(raster_settings=cam)(**depth_sil_var)
        depth = depth_out[0][0]  # first channel of "colors_precomp" output

    rgb = torch.clamp(im, 0, 1).permute(1, 2, 0).cpu().numpy()
    rgb = (rgb * 255).astype(np.uint8)
    depth_np = depth.cpu().numpy()
    return rgb, depth_np


def colorize_depth(depth, vmin=0.0, vmax=6.0):
    """Apply a colormap to a depth array; returns BGR uint8 image."""
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def load_gt_frame(gt_dir, frame_idx, target_size):
    """Load GT color/depth for frame `frame_idx`. target_size = (W, H)."""
    color_files = sorted(Path(gt_dir, "color").glob("*.png"))
    if frame_idx >= len(color_files):
        return None, None
    color = np.asarray(imageio.imread(color_files[frame_idx]), dtype=np.uint8)
    if color.shape[1] != target_size[0] or color.shape[0] != target_size[1]:
        color = cv2.resize(color, target_size, interpolation=cv2.INTER_LINEAR)

    depth_files = sorted(Path(gt_dir, "depth").glob("*.tiff"))
    if frame_idx < len(depth_files):
        depth = np.asarray(imageio.imread(depth_files[frame_idx]), dtype=np.float64)
        # Convert from C3VD's stored representation to metric depth
        # (uint16 with png_depth_scale=2.55 in c3vd.yaml)
        depth = depth / 2.55  # not the full pipeline, but visually OK
        if depth.shape[1] != target_size[0] or depth.shape[0] != target_size[1]:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        depth = None
    return color, depth


def stack_panel(frames, labels, label_height=30, divider=4):
    """Horizontally stack frames with labels above each one."""
    H = max(f.shape[0] for f in frames)
    panels = []
    for f, label in zip(frames, labels):
        # pad to common height
        if f.shape[0] != H:
            pad = np.zeros((H - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        # add label bar
        bar = np.full((label_height, f.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, label, (10, label_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        panels.append(np.vstack([bar, f]))

    # Insert vertical dividers
    out = panels[0]
    for p in panels[1:]:
        div = np.full((out.shape[0], divider, 3), 60, dtype=np.uint8)
        out = np.hstack([out, div, p])
    return out


# ============================================================
# Main rendering pipeline
# ============================================================

def render_video(gt_dir, experiment_dirs, labels, output_path,
                 fps=15, include_depth=False, max_frames=None):
    """
    Render a side-by-side comparison video.

    Args:
        gt_dir: directory containing color/ and depth/ for the GT sequence
        experiment_dirs: list of experiment dirs (each contains params.npz)
        labels: list of labels for each experiment (same length)
        output_path: output .mp4 path
        fps: video frame rate
        include_depth: stack depth maps below RGB panels
        max_frames: optional cap for testing
    """
    assert len(experiment_dirs) == len(labels), "labels must match experiment_dirs length"

    # Load all experiments
    print(f"Loading {len(experiment_dirs)} experiment(s)...")
    exps = []
    for d, label in zip(experiment_dirs, labels):
        print(f"  {label}: {d}")
        params = load_experiment(d)
        cam = make_camera(params)
        n_frames = params["cam_unnorm_rots"].shape[-1]
        H, W = int(params["org_height"]), int(params["org_width"])
        exps.append({
            "label": label, "params": params, "cam": cam,
            "n_frames": n_frames, "size": (W, H),
        })

    # Use the smallest n_frames across experiments (and apply max_frames cap)
    n_frames = min(e["n_frames"] for e in exps)
    if max_frames:
        n_frames = min(n_frames, max_frames)
    target_size = exps[0]["size"]
    print(f"Rendering {n_frames} frames at size {target_size[0]}x{target_size[1]}")

    # Set up video writer
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)

    try:
        for t in tqdm(range(n_frames), desc="Rendering"):
            rgb_panels = []
            depth_panels = []
            panel_labels = []

            # GT panel
            gt_color, gt_depth = load_gt_frame(gt_dir, t, target_size)
            if gt_color is None:
                print(f"  [warn] GT frame {t} missing, stopping.")
                break
            rgb_panels.append(gt_color)
            panel_labels.append("GT")
            if include_depth:
                if gt_depth is not None:
                    depth_panels.append(colorize_depth(gt_depth))
                else:
                    depth_panels.append(np.zeros_like(gt_color))

            # Each experiment
            for e in exps:
                rgb, depth = render_frame(e["params"], e["cam"], t)
                rgb_panels.append(rgb)
                panel_labels.append(e["label"])
                if include_depth:
                    depth_panels.append(colorize_depth(depth))

            # Stack horizontally
            row_rgb = stack_panel(rgb_panels, panel_labels)

            if include_depth:
                row_depth = stack_panel(
                    depth_panels,
                    [f"{l} depth" for l in panel_labels],
                )
                # Combine RGB row above depth row
                # (match widths if needed)
                if row_rgb.shape[1] != row_depth.shape[1]:
                    target_w = max(row_rgb.shape[1], row_depth.shape[1])
                    if row_rgb.shape[1] < target_w:
                        pad = np.zeros((row_rgb.shape[0], target_w - row_rgb.shape[1], 3), dtype=np.uint8)
                        row_rgb = np.hstack([row_rgb, pad])
                    if row_depth.shape[1] < target_w:
                        pad = np.zeros((row_depth.shape[0], target_w - row_depth.shape[1], 3), dtype=np.uint8)
                        row_depth = np.hstack([row_depth, pad])
                frame = np.vstack([row_rgb, row_depth])
            else:
                frame = row_rgb

            # Frame index overlay (top-right)
            cv2.putText(frame, f"frame {t:04d}", (frame.shape[1] - 160, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

            writer.append_data(frame)
    finally:
        writer.close()

    print(f"\nSaved video: {output_path}")
    print(f"  size: {os.path.getsize(output_path) / 1e6:.1f} MB")


# ============================================================
# Batch helper for all 10 C3VD scenes
# ============================================================

def batch_c3vd(args):
    """Render comparison videos for all 10 C3VD scenes that have results."""
    scenes = [
        "cecum_t1_b", "cecum_t2_b", "cecum_t3_a",
        "sigmoid_t1_a", "sigmoid_t2_a", "sigmoid_t3_a",
        "trans_t1_b", "trans_t2_c", "trans_t4_a", "trans_t4_b",
    ]
    os.makedirs(args.output_dir, exist_ok=True)

    for scene in scenes:
        gt = os.path.join(args.gt_root, scene)
        baseline = os.path.join(args.baseline_group, scene)
        ours = os.path.join(args.ours_group, scene)

        # Check what's actually available
        exp_dirs, labels = [], []
        if os.path.exists(os.path.join(baseline, "params.npz")):
            exp_dirs.append(baseline); labels.append("Baseline")
        if os.path.exists(os.path.join(ours, "params.npz")):
            exp_dirs.append(ours); labels.append("Ours")
        if not exp_dirs:
            print(f"\n[skip] {scene}: no completed experiments")
            continue
        if not os.path.exists(gt):
            print(f"\n[skip] {scene}: no GT data")
            continue

        out = os.path.join(args.output_dir, f"{scene}_compare.mp4")
        print(f"\n=== {scene} -> {out} ===")
        try:
            render_video(gt, exp_dirs, labels, out,
                         fps=args.fps, include_depth=args.include_depth,
                         max_frames=args.max_frames)
        except Exception as e:
            print(f"  [error] {scene}: {e}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Render SLAM comparison videos")

    parser.add_argument("--gt_dir", type=str,
                        help="GT data directory (must contain color/ and depth/)")
    parser.add_argument("--experiment_dir", action="append", default=[],
                        help="Experiment directory containing params.npz (repeatable)")
    parser.add_argument("--label", action="append", default=[],
                        help="Label for each --experiment_dir (in order)")
    parser.add_argument("--output", type=str, default="./comparison.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--include_depth", action="store_true",
                        help="Add depth panels below RGB panels")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap number of frames (for quick previews)")

    # Batch mode
    parser.add_argument("--batch_c3vd", action="store_true",
                        help="Render all 10 C3VD scenes from group dirs")
    parser.add_argument("--gt_root", type=str, default="./data/C3VD")
    parser.add_argument("--baseline_group", type=str)
    parser.add_argument("--ours_group", type=str)
    parser.add_argument("--output_dir", type=str, default="./videos")

    args = parser.parse_args()

    if args.batch_c3vd:
        if not (args.baseline_group or args.ours_group):
            parser.error("--batch_c3vd requires --baseline_group and/or --ours_group")
        batch_c3vd(args)
        return

    if not args.experiment_dir:
        parser.error("Provide --experiment_dir or --batch_c3vd")

    if not args.gt_dir:
        parser.error("--gt_dir is required")

    # Auto-fill labels if not provided
    if len(args.label) < len(args.experiment_dir):
        for i in range(len(args.label), len(args.experiment_dir)):
            args.label.append(f"Method{i+1}")
    elif len(args.label) > len(args.experiment_dir):
        parser.error("Too many --label entries")

    render_video(args.gt_dir, args.experiment_dir, args.label, args.output,
                 fps=args.fps, include_depth=args.include_depth,
                 max_frames=args.max_frames)


if __name__ == "__main__":
    main()
