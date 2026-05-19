"""
Optical flow utilities for EndoGSLAM.

Two main functions:
1. flow_guided_pose_init: Flow4DGS-style Camera-Induced Motion Decomposition
   (linearized image-Jacobian twist solver with IRLS Cauchy weighting) for
   coarse camera pose initialization.
2. compute_flow_loss: Per-Gaussian flow loss for mapping optimization.

References:
- Flow4DGS-SLAM (Wang & Lee, 2026): "Camera-Induced Motion Decomposition"
  via the linearized image Jacobian F = J(x) * xi and IRLS-weighted
  least-squares (Eqs. 1-3). Our flow_guided_pose_init faithfully adapts this
  method for endoscopic SLAM: we drop the YOLOv9 dynamic-object mask (the
  endoscopic scene is rigid) and operate at the SLAM (downsampled)
  resolution. The image Jacobian (Eq. 2) follows projective geometry as in
  Ma & Soatto, "An Invitation to 3D Vision" (Chapter 5).
- EndoFlow-SLAM (Wu et al., 2025): Gaussian flow loss in mapping.
- GaussianFlow (Gao et al., 2024): per-Gaussian 2D displacement theory.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# SE(3) Lie-algebra helpers
# ============================================================

def _skew(v):
    """3-vector -> 3x3 skew-symmetric matrix [v]_x."""
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def _exp_so3(theta):
    """
    SO(3) exponential map via Rodrigues' formula.
    Input:  theta (3,) axis-angle rotation vector
    Output: R (3, 3) rotation matrix
    Small-angle Taylor series used when |theta| < 1e-6.
    """
    angle = torch.linalg.norm(theta)
    K = _skew(theta)
    I = torch.eye(3, dtype=theta.dtype, device=theta.device)
    if angle < 1e-6:
        # Taylor: R = I + K + 0.5 K^2
        return I + K + 0.5 * (K @ K)
    a = torch.sin(angle) / angle
    b = (1.0 - torch.cos(angle)) / (angle * angle)
    return I + a * K + b * (K @ K)


def _exp_se3(xi):
    """
    SE(3) exponential map.
    Input:  xi (6,) = [rho (3,) translation; theta (3,) rotation]
    Output: T (4, 4) homogeneous transform
    Uses Rodrigues + SE(3) left Jacobian V(theta).
    """
    rho = xi[:3]
    theta = xi[3:]
    angle = torch.linalg.norm(theta)
    K = _skew(theta)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device)
    if angle < 1e-6:
        # Taylor: V = I + 0.5 K + (1/6) K^2
        R = I + K + 0.5 * (K @ K)
        V = I + 0.5 * K + (1.0 / 6.0) * (K @ K)
    else:
        a = torch.sin(angle) / angle
        b = (1.0 - torch.cos(angle)) / (angle * angle)
        c = (angle - torch.sin(angle)) / (angle ** 3)
        R = I + a * K + b * (K @ K)
        V = I + b * K + c * (K @ K)
    t = V @ rho
    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _build_image_jacobian(u_c, v_c, Z, fx, fy):
    """
    Build per-pixel projective image Jacobian (Eq. 2 of Flow4DGS-SLAM).

    J(x) = [[-fx/Z,    0,    u/Z,   u*v/fy,    -fx - u^2/fx,    v],
            [   0, -fy/Z,    v/Z,   fy + v^2/fy,   -u*v/fx,    -u]]

    where (u, v) are CENTERED pixel coordinates (i.e. u = u_pix - cx).

    Inputs:
        u_c, v_c, Z: (N,) tensors -- centered pixel coords and depths
        fx, fy: scalars
    Returns:
        J: (2N, 6) tensor stacking the 2x6 per-pixel Jacobians as
           [row_u_1, row_v_1, row_u_2, row_v_2, ...]
    """
    N = u_c.shape[0]
    invZ = 1.0 / Z

    # Row for u-component of flow (shape (N, 6))
    Ju = torch.stack([
        -fx * invZ,
        torch.zeros_like(Z),
        u_c * invZ,
        u_c * v_c / fy,
        -fx - (u_c * u_c) / fx,
        v_c,
    ], dim=1)

    # Row for v-component of flow (shape (N, 6))
    Jv = torch.stack([
        torch.zeros_like(Z),
        -fy * invZ,
        v_c * invZ,
        fy + (v_c * v_c) / fy,
        -(u_c * v_c) / fx,
        -u_c,
    ], dim=1)

    # Interleave rows: [Ju_1, Jv_1, Ju_2, Jv_2, ...] -> (2N, 6)
    J = torch.stack([Ju, Jv], dim=1).reshape(2 * N, 6)
    return J


# ============================================================
# Flow-Guided Pose Initialization
# ============================================================

def load_flow(flow_dir, frame_idx, target_hw=None):
    """
    Load precomputed optical flow for frame pair (frame_idx, frame_idx+1).

    If `target_hw=(H_tgt, W_tgt)` is provided and the stored flow resolution
    differs, the flow tensor is resized AND the flow vectors are scaled by the
    corresponding axis ratios. This is essential because RAFT was precomputed
    on the raw image resolution (e.g. 1080x1350) while SLAM operates at the
    downsampled resolution (e.g. 540x675); using the raw flow directly would
    produce a systematic 2x scale mismatch in both PnP-based pose init and
    Gaussian-flow loss.

    Args:
        flow_dir: directory containing flow_XXXX.npy files
        frame_idx: index of the source frame
        target_hw: optional (H, W) target resolution; if set, resize + rescale

    Returns:
        flow: (H, W, 2) numpy float32 array at target resolution (or stored
              resolution if target_hw is None), or None if not found
    """
    flow_path = os.path.join(flow_dir, f"flow_{frame_idx:04d}.npy")
    if not os.path.exists(flow_path):
        return None
    flow = np.load(flow_path).astype(np.float32)

    if target_hw is not None:
        H_tgt, W_tgt = int(target_hw[0]), int(target_hw[1])
        H_raw, W_raw = flow.shape[:2]
        if (H_raw, W_raw) != (H_tgt, W_tgt):
            sx = float(W_tgt) / float(W_raw)
            sy = float(H_tgt) / float(H_raw)
            flow = cv2.resize(flow, (W_tgt, H_tgt), interpolation=cv2.INTER_LINEAR)
            # Scale flow VECTORS (displacements are in pixels of their resolution)
            flow[..., 0] *= sx
            flow[..., 1] *= sy
    return flow


def flow_guided_pose_init(params, time_idx, flow, depth, intrinsics,
                          confidence_threshold=0.5,
                          depth_min=0.001, depth_max=10.0,
                          irls_iters=5, cauchy_c=1.0,
                          max_pixels=50000,
                          translation_gate=0.1, rotation_gate=0.1,
                          neg_xi=False, invert_T=False,
                          debug=False):
    """
    Estimate initial camera pose via Flow4DGS-style linearized twist solver.

    Faithful adaptation of Flow4DGS-SLAM "Camera-Induced Motion Decomposition"
    (Wang & Lee, 2026, Eqs. 1-3):

        F(u,v) = J(x) * xi                                    (Eq. 1)
        xi_hat = arg min_xi sum_i w_i ||F_i - J_i xi||^2      (Eq. 3)

    where xi = [rho; theta] in se(3) is the camera twist and J is the 2x6
    projective image Jacobian (Eq. 2). We solve via IRLS with Cauchy weights.

    Adaptations for endoscopic SLAM:
      - No YOLOv9 dynamic-object mask (endoscopic scene is rigid).
      - Operates at SLAM resolution; caller MUST pass flow/depth/intrinsics
        all at the same resolution (use load_flow(target_hw=...) upstream).

    Sign convention: our precomputed flow is F(t-1 -> t) (see
    scripts/precompute_flow.py), so the solved twist xi represents motion
    from frame t-1 to frame t, giving T_rel = T(cam_t <- cam_{t-1}).
    Composition: new_w2c = T_rel @ prev_w2c.

    Args:
        params: parameter dict (cam_unnorm_rots, cam_trans)
        time_idx: current frame index (we estimate pose for this frame)
        flow: (H, W, 2) optical flow from frame t-1 to t (SLAM resolution)
        depth: (1, H, W) or (H, W) depth at frame t-1 (SLAM resolution)
        intrinsics: (3, 3) camera intrinsic matrix (SLAM resolution)
        confidence_threshold: min |flow| (SLAM pixels) to include a pixel
        depth_min, depth_max: valid depth range (endoscopy: ~0.001 to ~10 m)
        irls_iters: number of IRLS iterations (typically 5)
        cauchy_c: Cauchy robust scale (pixels); residuals >> c are down-weighted
        max_pixels: subsampling cap; deterministic stride sample if exceeded
        translation_gate: reject if ||rho|| > this (meters); fallback to CV
        rotation_gate: reject if ||theta|| > this (radians); fallback to CV
        debug: if True, print solver diagnostics

    Returns:
        params: updated with initial pose estimate for time_idx (or unchanged)
        success: bool, whether the IRLS solve passed sanity gates
    """
    if flow is None or time_idx < 1:
        return params, False

    H, W = flow.shape[:2]
    if depth.dim() == 3:
        depth_np = depth[0].cpu().numpy()
    else:
        depth_np = depth.cpu().numpy()

    # Defensive: shapes must match. If they don't, the caller forgot target_hw.
    if depth_np.shape[:2] != (H, W):
        if debug:
            print(f"[flow_init t={time_idx}] SHAPE MISMATCH: "
                  f"flow={flow.shape}, depth={depth_np.shape} -> skip",
                  flush=True)
        return params, False

    # Compute flow magnitude for confidence filtering
    flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Valid pixels: positive depth + sufficient flow
    valid_mask = (depth_np > depth_min) & (depth_np < depth_max) & (flow_mag > confidence_threshold)
    valid_ys, valid_xs = np.where(valid_mask)
    n_valid = len(valid_ys)

    if debug:
        print(f"[flow_init t={time_idx}] depth range=[{depth_np.min():.4f}, "
              f"{depth_np.max():.4f}], flow_mag mean={flow_mag.mean():.3f}, "
              f"valid pixels={n_valid}",
              flush=True)

    if n_valid < 50:
        if debug:
            print(f"[flow_init t={time_idx}] too few valid pixels -> fallback",
                  flush=True)
        return params, False

    # Deterministic stride subsampling (no randomness)
    if n_valid > max_pixels:
        stride = int(np.ceil(n_valid / max_pixels))
        ys = valid_ys[::stride]
        xs = valid_xs[::stride]
    else:
        ys = valid_ys
        xs = valid_xs
    n_used = len(ys)

    # Intrinsics
    K_np = intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else intrinsics
    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])

    # Move sampled data to torch/CUDA for solve
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    Z = torch.from_numpy(depth_np[ys, xs]).to(device=device, dtype=dtype)
    # Centered pixel coordinates for the projective image Jacobian (Eq. 2)
    u_c = torch.from_numpy(xs.astype(np.float32) - cx).to(device=device, dtype=dtype)
    v_c = torch.from_numpy(ys.astype(np.float32) - cy).to(device=device, dtype=dtype)

    # Flow vectors stacked as [F_u, F_v] interleaved per pixel: (2N,)
    Fu = torch.from_numpy(flow[ys, xs, 0].astype(np.float32)).to(device=device, dtype=dtype)
    Fv = torch.from_numpy(flow[ys, xs, 1].astype(np.float32)).to(device=device, dtype=dtype)
    F_stack = torch.stack([Fu, Fv], dim=1).reshape(2 * n_used)  # (2N,)

    # Build the (2N, 6) image Jacobian
    J = _build_image_jacobian(u_c, v_c, Z, fx, fy)  # (2N, 6)

    # IRLS: initialize weights uniformly
    W_stack = torch.ones(2 * n_used, device=device, dtype=dtype)
    xi = torch.zeros(6, device=device, dtype=dtype)
    final_residual_rms = float("inf")

    eps = 1e-8
    for irls_step in range(irls_iters):
        # Weighted normal equations: (J^T W J) xi = J^T W F
        WJ = W_stack[:, None] * J          # (2N, 6)
        A = J.t() @ WJ                     # (6, 6)
        b = J.t() @ (W_stack * F_stack)    # (6,)
        # Tiny ridge for numerical stability on degenerate scenes
        A = A + eps * torch.eye(6, device=device, dtype=dtype)
        try:
            xi = torch.linalg.solve(A, b)
        except RuntimeError:
            if debug:
                print(f"[flow_init t={time_idx}] IRLS solve failed at iter "
                      f"{irls_step} -> fallback",
                      flush=True)
            return params, False

        # Residuals and Cauchy weights for next iter
        F_pred = J @ xi                    # (2N,)
        r = F_stack - F_pred               # (2N,)
        W_stack = 1.0 / (1.0 + (r / cauchy_c) ** 2)
        final_residual_rms = float(torch.sqrt((r * r).mean()).item())

    # Sign convention is controlled by neg_xi flag (set from config).
    # Synthetic experiments suggest the IRLS-recovered xi is the negative of
    # the point-transform twist (i.e., it is the CAMERA-motion twist).
    # To get the point-transform T(B<-A) we need exp(-xi). But the empirical
    # result depends on the actual flow direction convention, so we expose
    # both sign and inversion as config flags.
    if neg_xi:
        xi = -xi

    rho = xi[:3]
    theta = xi[3:]
    rho_norm = float(torch.linalg.norm(rho).item())
    theta_norm = float(torch.linalg.norm(theta).item())

    # Sanity gates -- reject obviously broken solutions
    if rho_norm > translation_gate or theta_norm > rotation_gate:
        if debug:
            print(f"[flow_init t={time_idx}] sanity gate FAIL "
                  f"(|rho|={rho_norm:.4f}, |theta|={theta_norm:.4f}, "
                  f"rms={final_residual_rms:.3f}) -> fallback",
                  flush=True)
        return params, False

    if debug:
        print(f"[flow_init t={time_idx}] IRLS OK: used={n_used}, "
              f"rms={final_residual_rms:.3f}, |rho|={rho_norm:.4f}, "
              f"|theta|={theta_norm:.4f}",
              flush=True)

    # Build T_rel from twist and compose with previous w2c
    from utils.slam_external import build_rotation
    from utils.slam_helpers import matrix_to_quaternion

    with torch.no_grad():
        T_rel = _exp_se3(xi)               # (4, 4)
        if invert_T:
            T_rel = torch.linalg.inv(T_rel)

        prev_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx - 1].detach())
        prev_tran = params["cam_trans"][..., time_idx - 1].detach()
        prev_w2c = torch.eye(4, dtype=dtype, device=device)
        prev_w2c[:3, :3] = build_rotation(prev_rot)
        prev_w2c[:3, 3] = prev_tran

        if debug and time_idx >= 2:
            # Compare to constant-velocity prediction
            prev_tran2 = params["cam_trans"][..., time_idx - 2].detach()
            cv_delta = (prev_tran - prev_tran2).cpu().numpy().astype(float).flatten()
            irls_delta_cam = T_rel[:3, 3].cpu().numpy().astype(float).flatten()
            cv_norm = float(np.linalg.norm(cv_delta))
            print(f"[flow_init t={time_idx}] cv_delta_world="
                  f"({float(cv_delta[0]):+.3f},{float(cv_delta[1]):+.3f},{float(cv_delta[2]):+.3f}) "
                  f"|cv|={cv_norm:.3f}; "
                  f"irls_t_cam="
                  f"({float(irls_delta_cam[0]):+.3f},{float(irls_delta_cam[1]):+.3f},{float(irls_delta_cam[2]):+.3f})",
                  flush=True)

        # new_w2c = T_rel @ prev_w2c  (flow is t-1 -> t, so T_rel = T(t <- t-1))
        new_w2c = T_rel @ prev_w2c

        new_rot_quat = matrix_to_quaternion(new_w2c[:3, :3].unsqueeze(0))
        new_tran = new_w2c[:3, 3]

        params["cam_unnorm_rots"][..., time_idx] = new_rot_quat.detach()
        params["cam_trans"][..., time_idx] = new_tran.detach()

    return params, True


# ============================================================
# Gaussian Flow Loss for Mapping
# ============================================================

def compute_flow_loss(params, time_idx_a, time_idx_b, gt_flow, cam_settings,
                      intrinsics, max_points=10000, debug=False):
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

    # Get camera poses for both frames. Frame A (previous) is detached so the
    # flow loss only sends gradients to the CURRENT pose (frame B) and to the
    # Gaussian positions. Otherwise gradients leak back to already-finalized
    # historical poses, corrupting them.
    rot_a = F.normalize(params["cam_unnorm_rots"][..., time_idx_a].detach())
    tran_a = params["cam_trans"][..., time_idx_a].detach()
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

    if debug:
        with torch.no_grad():
            gs_mag = torch.sqrt(gs_fu ** 2 + gs_fv ** 2).mean().item()
            gt_mag = torch.sqrt(gt_fu ** 2 + gt_fv ** 2).mean().item()
            print(f"[flow_loss a={time_idx_a} b={time_idx_b}] "
                  f"K={in_bounds.sum().item()}, |gs_flow|={gs_mag:.3f}, "
                  f"|gt_flow|={gt_mag:.3f}, loss={flow_loss.item():.4f}")

    return flow_loss
