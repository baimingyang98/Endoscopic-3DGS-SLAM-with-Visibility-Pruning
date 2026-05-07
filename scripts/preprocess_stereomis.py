"""
StereoMIS preprocessing pipeline.

Converts raw StereoMIS data (video.mp4 + masks/ + groundtruth.txt + calibration)
into the RGB-D format expected by our SLAM loader.

Pipeline:
1. Extract left + right frames from vertically-stacked stereo video.mp4
2. Compute depth from stereo disparity (OpenCV StereoSGBM)
3. Filter to frames that have masks (these define the valid set)
4. Convert poses to 4x4 matrices
5. Save in format compatible with StereoMISDataset loader

Output structure:
    output_dir/sequence/
        color/000241.png       # left RGB (camera distortion-corrected)
        depth/000241.tiff      # depth in millimeters (uint16)
        pose.txt               # one 4x4 pose per line (16 comma-separated values)

Usage:
    python scripts/preprocess_stereomis.py \\
        --input data/StereoMIS_0_0_1/P1 \\
        --output data/StereoMIS/P1
"""
import argparse
import configparser
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm


# ============================================================
# Calibration parsing
# ============================================================

def parse_calibration(ini_path):
    """Parse StereoCalibration.ini -> left/right intrinsics + stereo extrinsics."""
    cp = configparser.ConfigParser()
    cp.read(ini_path)

    def section_to_dict(s):
        return {k: float(v) for k, v in cp.items(s)}

    L = section_to_dict("StereoLeft")
    R = section_to_dict("StereoRight")

    K_left = np.array([
        [L["fc_x"], 0,         L["cc_x"]],
        [0,         L["fc_y"], L["cc_y"]],
        [0,         0,         1.0],
    ])
    K_right = np.array([
        [R["fc_x"], 0,         R["cc_x"]],
        [0,         R["fc_y"], R["cc_y"]],
        [0,         0,         1.0],
    ])
    dist_left = np.array([L[f"kc_{i}"] for i in range(5)])
    dist_right = np.array([R[f"kc_{i}"] for i in range(5)])

    R_rel = np.array([
        [R["r_0"], R["r_1"], R["r_2"]],
        [R["r_3"], R["r_4"], R["r_5"]],
        [R["r_6"], R["r_7"], R["r_8"]],
    ])
    T_rel = np.array([R["t_0"], R["t_1"], R["t_2"]])  # in mm

    return {
        "K_left": K_left,
        "K_right": K_right,
        "dist_left": dist_left,
        "dist_right": dist_right,
        "R": R_rel,
        "T": T_rel,
        "image_size": (int(L["res_x"]), int(L["res_y"])),  # (W, H)
    }


# ============================================================
# Pose conversion
# ============================================================

def quaternion_to_matrix(qx, qy, qz, qw):
    """Quaternion (xyzw) to 3x3 rotation matrix."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    s = 2.0 / max(n, 1e-12)
    return np.array([
        [1 - s*(qy*qy + qz*qz),  s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),      1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),      s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])


def parse_groundtruth(gt_path):
    """
    Parse groundtruth.txt -> dict mapping frame_idx -> 4x4 pose matrix.
    
    Each line: frame_idx tx ty tz qx qy qz qw
    """
    poses = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            idx = int(parts[0])
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
            R = quaternion_to_matrix(qx, qy, qz, qw)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[idx] = T
    return poses


# ============================================================
# Stereo depth computation
# ============================================================

def make_stereo_matcher():
    """Create OpenCV StereoSGBM matcher tuned for endoscopic scenes."""
    # WLS-filtered SGBM gives reasonable quality on textured tissue
    window_size = 5
    min_disp = 0
    num_disp = 96  # multiple of 16, covers ~ 14cm at fx*B/96 ≈ 4.7cm/disp
    
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def compute_depth(left_rect, right_rect, matcher, fx, baseline_mm):
    """
    Compute depth map (in mm) from rectified stereo pair.
    
    depth = (fx * baseline) / disparity
    """
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    
    # SGBM returns disparity in 1/16 pixel units
    disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # Avoid division by zero / invalid pixels
    valid = disparity > 0.5
    depth = np.zeros_like(disparity)
    depth[valid] = (fx * baseline_mm) / disparity[valid]
    
    # Clip extreme values (>50 cm = noise) and return as uint16 mm
    depth = np.clip(depth, 0, 500)
    return depth.astype(np.float32)


# ============================================================
# Stereo rectification
# ============================================================

def setup_rectification(calib):
    """Compute rectification maps for stereo pair."""
    W, H = calib["image_size"]
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=calib["K_left"],
        distCoeffs1=calib["dist_left"],
        cameraMatrix2=calib["K_right"],
        distCoeffs2=calib["dist_right"],
        imageSize=(W, H),
        R=calib["R"],
        T=calib["T"],
        alpha=0,  # crop to valid region
    )
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib["K_left"], calib["dist_left"], R1, P1, (W, H), cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib["K_right"], calib["dist_right"], R2, P2, (W, H), cv2.CV_32FC1
    )
    
    # Effective intrinsics after rectification (from P1)
    fx_rect = P1[0, 0]
    fy_rect = P1[1, 1]
    cx_rect = P1[0, 2]
    cy_rect = P1[1, 2]
    baseline_mm = abs(calib["T"][0])  # |T_x|
    
    return {
        "map_left": (map1_l, map2_l),
        "map_right": (map1_r, map2_r),
        "fx": fx_rect, "fy": fy_rect, "cx": cx_rect, "cy": cy_rect,
        "baseline_mm": baseline_mm,
        "image_size": (W, H),
    }


# ============================================================
# Main pipeline
# ============================================================

def get_valid_frame_indices(masks_dir):
    """Extract frame indices from mask filenames (e.g., '000241l.png' -> 241)."""
    indices = []
    for fname in os.listdir(masks_dir):
        if fname.endswith("l.png"):
            try:
                idx = int(fname[:-5])  # strip 'l.png'
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)


def preprocess(input_dir, output_dir):
    print(f"Preprocessing: {input_dir} -> {output_dir}\n")
    os.makedirs(output_dir, exist_ok=True)
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # 1. Parse calibration
    calib_path = os.path.join(input_dir, "StereoCalibration.ini")
    calib = parse_calibration(calib_path)
    print(f"  Image size: {calib['image_size']}")
    print(f"  Left K: fx={calib['K_left'][0,0]:.2f}, fy={calib['K_left'][1,1]:.2f}, "
          f"cx={calib['K_left'][0,2]:.2f}, cy={calib['K_left'][1,2]:.2f}")
    print(f"  Stereo baseline: {abs(calib['T'][0]):.2f} mm\n")

    # 2. Setup rectification
    rect = setup_rectification(calib)
    print(f"  Rectified intrinsics: fx={rect['fx']:.2f}, fy={rect['fy']:.2f}, "
          f"cx={rect['cx']:.2f}, cy={rect['cy']:.2f}")
    print(f"  Baseline: {rect['baseline_mm']:.2f} mm\n")

    # 3. Parse poses
    gt_path = os.path.join(input_dir, "groundtruth.txt")
    poses = parse_groundtruth(gt_path)
    print(f"  Loaded {len(poses)} poses\n")

    # 4. Determine valid frames from masks
    masks_dir = os.path.join(input_dir, "masks")
    valid_indices = get_valid_frame_indices(masks_dir)
    print(f"  Valid frames (from masks): {len(valid_indices)}")
    print(f"    Range: {valid_indices[0]} to {valid_indices[-1]}\n")

    # 5. Open video
    video_path = os.path.join(input_dir, "video.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {n_frames} frames\n")

    # 6. Setup matcher
    matcher = make_stereo_matcher()

    # 7. Process each valid frame
    valid_set = set(valid_indices)
    pose_lines = []
    written_count = 0

    pbar = tqdm(total=len(valid_indices), desc="Processing frames")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in valid_set and frame_idx in poses:
            # Split top/bottom into left/right
            left_raw = frame[:1024, :, :]
            right_raw = frame[1024:, :, :]

            # Rectify
            left_rect = cv2.remap(left_raw, rect["map_left"][0], rect["map_left"][1],
                                  cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, rect["map_right"][0], rect["map_right"][1],
                                   cv2.INTER_LINEAR)

            # Compute depth
            depth_mm = compute_depth(left_rect, right_rect, matcher,
                                     rect["fx"], rect["baseline_mm"])

            # Save (save depth as uint16 for compatibility, scale: 1 unit = 1 mm)
            color_path = os.path.join(color_dir, f"{frame_idx:06d}.png")
            depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.tiff")
            cv2.imwrite(color_path, left_rect)
            cv2.imwrite(depth_path, depth_mm.astype(np.uint16))

            # Pose: convert to comma-separated 16-value line (matching C3VD format)
            pose_mat = poses[frame_idx]
            line = ",".join(str(v) for v in pose_mat.reshape(-1))
            pose_lines.append(line)

            written_count += 1
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    # 8. Write pose file
    with open(os.path.join(output_dir, "pose.txt"), "w") as f:
        f.write("\n".join(pose_lines) + "\n")

    # 9. Write rectified intrinsics for the loader
    intrinsics_yaml = f"""# Auto-generated from StereoMIS preprocessing
# Use these values in configs/data/stereomis.yaml
dataset_name: "stereomis"
camera_params:
  image_height: {rect['image_size'][1]}
  image_width: {rect['image_size'][0]}
  fx: {rect['fx']:.4f}
  fy: {rect['fy']:.4f}
  cx: {rect['cx']:.4f}
  cy: {rect['cy']:.4f}
  png_depth_scale: 1.0   # depth saved in mm, scale to mm = 1.0
  crop_edge: 0
"""
    with open(os.path.join(output_dir, "intrinsics.yaml"), "w") as f:
        f.write(intrinsics_yaml)

    print(f"\n  Wrote {written_count} frames")
    print(f"  Output: {output_dir}")
    print(f"\n  IMPORTANT: copy {output_dir}/intrinsics.yaml contents into")
    print(f"  configs/data/stereomis.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw sequence (e.g., data/StereoMIS_0_0_1/P1)")
    parser.add_argument("--output", required=True, help="Path to output directory (e.g., data/StereoMIS/P1)")
    args = parser.parse_args()
    preprocess(args.input, args.output)
