"""
Optical flow utilities for EndoGSLAM.

Two main functions:
1. flow_guided_pose_init: Use optical flow + depth to estimate initial camera pose
2. compute_flow_loss: Per-Gaussian flow loss for mapping optimization

Based on ideas from:
- Flow4DGS-SLAM (Wang & Lee, 2026): flow-guided pose initialization
- EndoFlow-SLAM (Wu et al., 2025): Gaussian flow loss in mapping
- GaussianFlow (Gao et al., 2024): per-Gaussian 2D displacement theory
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Flow-Guided Pose Initialization
# ============================================================

def load_flow(flow_dir, frame_idx):
    """
    Load precomputed optical flow for frame pair (frame_idx, frame_idx+1).

    Args:
        flow_dir: directory containing flow_XXXX.npy files
        frame_idx: index of the source frame

    Returns:
        flow: (H, W, 2) numpy float32 array, or None if not found
    """
    flow_path = os.path.join(flow_dir, f"flow_{frame_idx:04d}.npy")
    if not os.path.exists(flow_path):
        return None
    flow = np.load(flow_path).astype(np.float32)
    return flow


def flow_guided_pose_init(params, time_idx, flow, depth, intrinsics,
                          confidence_threshold=0.5, n_points=500):
    """
    Estimate initial camera pose from optical flow + depth using PnP.

    Given flow from frame (t-1) -> t and depth at frame (t-1), compute
    3D-2D correspondences and solve PnP for the relative pose.

    Falls back to constant-velocity model if PnP fails or flow is unreliable.

    Args:
        params: parameter dict (cam_unnorm_rots, cam_trans)
        time_idx: current frame index (we estimate pose for this frame)
        flow: (H, W, 2) optical flow from frame t-1 to t
        depth: (1, H, W) or (H, W) depth map at frame t-1
        intrinsics: (3, 3) camera intrinsic matrix
        confidence_threshold: min flow magnitude to use a point
        n_points: number of correspondence points to sample

    Returns:
        params: updated with initial pose estimate for time_idx
        success: bool, whether PnP succeeded
    """
    if flow is None or time_idx < 1:
        return params, False

    H, W = flow.shape[:2]
    if depth.dim() == 3:
        depth_np = depth[0].cpu().numpy()
    else:
        depth_np = depth.cpu().numpy()

    # Compute flow magnitude for confidence filtering
    flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Valid points: positive depth + sufficient flow
    valid_mask = (depth_np > 0.01) & (depth_np < 100.0) & (flow_mag > confidence_threshold)
    valid_ys, valid_xs = np.where(valid_mask)

    if len(valid_ys) < 10:
        return params, False

    # Sample points (uniform random from valid set)
    n_sample = min(n_points, len(valid_ys))
    indices = np.random.choice(len(valid_ys), n_sample, replace=False)
    ys = valid_ys[indices]
    xs = valid_xs[indices]

    # Intrinsics
    K = intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 3D points in frame t-1 (camera coordinates)
    z = depth_np[ys, xs]
    x3d = (xs - cx) * z / fx
    y3d = (ys - cy) * z / fy
    pts_3d = np.stack([x3d, y3d, z], axis=1).astype(np.float64)  # (N, 3)

    # 2D points in frame t (pixel coordinates after flow)
    u_t = xs + flow[ys, xs, 0]
    v_t = ys + flow[ys, xs, 1]
    pts_2d = np.stack([u_t, v_t], axis=1).astype(np.float64)  # (N, 2)

    # Solve PnP (RANSAC for robustness)
    K_cv = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_cv, None,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 5:
        return params, False

    # Convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Construct relative w2c from frame t-1 to frame t
    # PnP gives us the pose of frame t in the frame t-1 coordinate system
    # We need to compose with the previous frame's pose to get world-to-camera
    # Actually, since our SLAM stores per-frame w2c directly:
    # The relative transformation is R_rel, t_rel
    # new_w2c = R_rel @ prev_w2c (for the rotation part)

    # For simplicity in our framework: use the PnP result as a relative transform
    # and compose with the previous frame's estimated pose

    from utils.slam_external import build_rotation

    with torch.no_grad():
        # Get previous pose
        prev_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx - 1].detach())
        prev_tran = params["cam_trans"][..., time_idx - 1].detach()

        # Previous w2c matrix
        prev_w2c = torch.eye(4).cuda().float()
        prev_w2c[:3, :3] = build_rotation(prev_rot)
        prev_w2c[:3, 3] = prev_tran

        # Relative transform from PnP
        R_rel = torch.from_numpy(R).float().cuda()
        t_rel = torch.from_numpy(tvec.squeeze()).float().cuda()
        rel_transform = torch.eye(4).cuda().float()
        rel_transform[:3, :3] = R_rel
        rel_transform[:3, 3] = t_rel

        # New w2c = relative @ previous
        new_w2c = rel_transform @ prev_w2c

        # Extract quaternion from new rotation
        from utils.slam_helpers import matrix_to_quaternion
        new_rot_quat = matrix_to_quaternion(new_w2c[:3, :3].unsqueeze(0))
        new_tran = new_w2c[:3, 3]

        # Set initial pose
        params["cam_unnorm_rots"][..., time_idx] = new_rot_quat.detach()
        params["cam_trans"][..., time_idx] = new_tran.detach()

    return params, True


# ============================================================
# Gaussian Flow Loss for Mapping
# ============================================================

def compute_flow_loss(params, time_idx_a, time_idx_b, gt_flow, cam_settings,
                      intrinsics, max_points=10000):
    """
    Compute optical flow loss between two frames.

    Projects Gaussian means at both camera poses, computes per-Gaussian 2D
    displacement ("Gaussian flow"), alpha-blends into a per-pixel flow map,
    and compares against precomputed optical flow.

    Simplified version: compare projected Gaussian mean displacements directly
    against sampled flow vectors (avoids full per-pixel rendering).

    Args:
        params: Gaussian parameters (means3D, cam_unnorm_rots, cam_trans, logit_opacities)
        time_idx_a: source frame index
        time_idx_b: target frame index
        gt_flow: (H, W, 2) precomputed optical flow from a to b (numpy or tensor)
        cam_settings: GaussianRasterizationSettings (for image dimensions)
        intrinsics: (3, 3) intrinsic matrix
        max_points: max Gaussians to use (subsample for efficiency)

    Returns:
        flow_loss: scalar loss tensor (differentiable w.r.t. cam poses and means3D)
    """
    from utils.slam_external import build_rotation

    # Get camera poses for both frames
    rot_a = F.normalize(params["cam_unnorm_rots"][..., time_idx_a])
    tran_a = params["cam_trans"][..., time_idx_a]
    rot_b = F.normalize(params["cam_unnorm_rots"][..., time_idx_b])
    tran_b = params["cam_trans"][..., time_idx_b]

    # Build w2c matrices
    w2c_a = torch.eye(4).cuda().float()
    w2c_a[:3, :3] = build_rotation(rot_a)
    w2c_a[:3, 3] = tran_a
    w2c_b = torch.eye(4).cuda().float()
    w2c_b[:3, :3] = build_rotation(rot_b)
    w2c_b[:3, 3] = tran_b

    # Get Gaussian positions
    pts = params["means3D"]  # (N, 3), requires_grad
    N = pts.shape[0]

    # Subsample for efficiency if too many Gaussians
    if N > max_points:
        # Prioritize visible Gaussians (high opacity)
        opacities = torch.sigmoid(params["logit_opacities"].squeeze())
        _, top_idx = torch.topk(opacities, max_points)
        pts_sub = pts[top_idx]
    else:
        pts_sub = pts
        top_idx = None

    # Project to frame A
    pts_ones = torch.ones(pts_sub.shape[0], 1, device="cuda")
    pts4 = torch.cat([pts_sub, pts_ones], dim=1)  # (M, 4)
    cam_pts_a = (w2c_a @ pts4.T).T[:, :3]  # (M, 3)
    cam_pts_b = (w2c_b @ pts4.T).T[:, :3]  # (M, 3)

    # Intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Filter: only use Gaussians visible in both frames (z > 0)
    valid = (cam_pts_a[:, 2] > 0.01) & (cam_pts_b[:, 2] > 0.01)
    if valid.sum() < 10:
        return torch.tensor(0.0, device="cuda", requires_grad=True)

    cam_a_valid = cam_pts_a[valid]
    cam_b_valid = cam_pts_b[valid]

    # Project to 2D pixel coordinates
    u_a = fx * cam_a_valid[:, 0] / cam_a_valid[:, 2] + cx
    v_a = fy * cam_a_valid[:, 1] / cam_a_valid[:, 2] + cy
    u_b = fx * cam_b_valid[:, 0] / cam_b_valid[:, 2] + cx
    v_b = fy * cam_b_valid[:, 1] / cam_b_valid[:, 2] + cy

    # Gaussian flow: displacement of projected mean
    gs_flow_u = u_b - u_a  # (M_valid,)
    gs_flow_v = v_b - v_a  # (M_valid,)

    # Get corresponding GT flow at projected positions in frame A
    H = int(cam_settings.image_height)
    W = int(cam_settings.image_width)

    # Convert GT flow to tensor if needed
    if isinstance(gt_flow, np.ndarray):
        gt_flow_t = torch.from_numpy(gt_flow).float().cuda()
    else:
        gt_flow_t = gt_flow.cuda()

    # Sample GT flow at Gaussian projection locations (bilinear)
    # Normalize pixel coords to [-1, 1] for grid_sample
    u_norm = 2.0 * u_a.detach() / (W - 1) - 1.0
    v_norm = 2.0 * v_a.detach() / (H - 1) - 1.0

    # Filter out-of-bounds
    in_bounds = (u_norm > -1) & (u_norm < 1) & (v_norm > -1) & (v_norm < 1)
    if in_bounds.sum() < 10:
        return torch.tensor(0.0, device="cuda", requires_grad=True)

    grid = torch.stack([u_norm[in_bounds], v_norm[in_bounds]], dim=1)  # (K, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # (1, K, 1, 2) for grid_sample

    # gt_flow_t: (H, W, 2) -> (1, 2, H, W)
    gt_flow_perm = gt_flow_t.permute(2, 0, 1).unsqueeze(0)

    # Sample
    sampled_flow = F.grid_sample(gt_flow_perm, grid, mode="bilinear", align_corners=True)
    # sampled_flow: (1, 2, K, 1)
    gt_fu = sampled_flow[0, 0, :, 0]  # (K,)
    gt_fv = sampled_flow[0, 1, :, 0]  # (K,)

    # Compute L2 loss between Gaussian flow and GT optical flow
    gs_fu = gs_flow_u[in_bounds]
    gs_fv = gs_flow_v[in_bounds]

    flow_loss = torch.mean((gs_fu - gt_fu) ** 2 + (gs_fv - gt_fv) ** 2)

    return flow_loss
