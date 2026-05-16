"""
Precompute optical flow for all consecutive frame pairs using RAFT.

Saves dense optical flow as float16 .npy files for use in the SLAM pipeline.

Usage:
    python scripts/precompute_flow.py --data_dir ./data/C3VD --scene sigmoid_t3_a
    python scripts/precompute_flow.py --data_dir ./data/C3VD --all

Requirements:
    - RAFT repository cloned (e.g., /content/RAFT)
    - RAFT Sintel weights downloaded
"""
import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add RAFT to path (user must clone RAFT repo)
RAFT_DIR = os.environ.get("RAFT_DIR", "/content/RAFT")
if os.path.exists(RAFT_DIR):
    sys.path.insert(0, RAFT_DIR)
    sys.path.insert(0, os.path.join(RAFT_DIR, "core"))


def load_raft_model(weights_path, device="cuda"):
    """Load RAFT model with Sintel weights."""
    from raft import RAFT
    from argparse import Namespace

    args = Namespace(
        model=weights_path,
        small=False,
        mixed_precision=True,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    return model


def preprocess_frame(image_path, device="cuda"):
    """Load and preprocess an image for RAFT (uint8 -> float32 tensor)."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return img


def compute_flow_pair(model, img1, img2, iters=20):
    """Compute optical flow from img1 to img2 using RAFT."""
    with torch.no_grad():
        _, flow = model(img1, img2, iters=iters, test_mode=True)
    return flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)


def precompute_scene(data_dir, scene, model, device="cuda"):
    """Precompute all flows for a single scene."""
    scene_dir = os.path.join(data_dir, scene)
    color_dir = os.path.join(scene_dir, "color")
    flow_dir = os.path.join(scene_dir, "flow")
    os.makedirs(flow_dir, exist_ok=True)

    # Get sorted color image paths
    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    if not color_files:
        print(f"  [skip] No color images found in {color_dir}")
        return

    n_pairs = len(color_files) - 1
    print(f"  {scene}: {len(color_files)} frames -> {n_pairs} flow pairs")

    # Check if already computed
    existing = glob.glob(os.path.join(flow_dir, "*.npy"))
    if len(existing) >= n_pairs:
        print(f"  [skip] Already computed ({len(existing)} files)")
        return

    for i in tqdm(range(n_pairs), desc=f"  {scene}"):
        out_path = os.path.join(flow_dir, f"flow_{i:04d}.npy")
        if os.path.exists(out_path):
            continue

        img1 = preprocess_frame(color_files[i], device)
        img2 = preprocess_frame(color_files[i + 1], device)

        # RAFT requires dimensions divisible by 8; pad if needed
        _, _, H, W = img1.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img1 = torch.nn.functional.pad(img1, [0, pad_w, 0, pad_h], mode="replicate")
            img2 = torch.nn.functional.pad(img2, [0, pad_w, 0, pad_h], mode="replicate")

        flow = compute_flow_pair(model, img1, img2)

        # Remove padding from flow
        if pad_h > 0 or pad_w > 0:
            flow = flow[:H, :W, :]

        # Save as float16 to reduce disk usage
        np.save(out_path, flow.astype(np.float16))

    print(f"  Done: {flow_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute RAFT optical flow")
    parser.add_argument("--data_dir", type=str, default="./data/C3VD", help="Base data directory")
    parser.add_argument("--scene", type=str, default=None, help="Single scene to process")
    parser.add_argument("--all", action="store_true", help="Process all 10 C3VD scenes")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to RAFT weights (default: RAFT_DIR/models/raft-sintel.pth)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Find RAFT weights
    if args.weights is None:
        candidates = [
            os.path.join(RAFT_DIR, "models", "raft-sintel.pth"),
            os.path.join(RAFT_DIR, "models", "raft-things.pth"),
            "/content/raft-sintel.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.weights = c
                break
        if args.weights is None:
            print("ERROR: RAFT weights not found. Download raft-sintel.pth and specify --weights.")
            sys.exit(1)

    print(f"Loading RAFT model from: {args.weights}")
    model = load_raft_model(args.weights, device=args.device)
    print("RAFT model loaded.")

    # Determine which scenes to process
    if args.all:
        scenes = [
            "cecum_t1_b", "cecum_t2_b", "cecum_t3_a",
            "sigmoid_t1_a", "sigmoid_t2_a", "sigmoid_t3_a",
            "trans_t1_b", "trans_t2_c", "trans_t4_a", "trans_t4_b",
        ]
    elif args.scene:
        scenes = [args.scene]
    else:
        parser.error("Specify --scene or --all")

    for scene in scenes:
        scene_path = os.path.join(args.data_dir, scene)
        if not os.path.exists(scene_path):
            print(f"  [skip] {scene}: directory not found")
            continue
        precompute_scene(args.data_dir, scene, model, device=args.device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
