"""
Core SLAM helper functions for Gaussian parameter transformation and rendering.

Includes:
- Loss functions (L1, weighted L2)
- Quaternion math
- Gaussian-to-camera-frame transformations
- Innovation 3: Deformation-aware transform_to_frame
"""
import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation


# ============================================================
# Loss functions
# ============================================================

def l1_loss_v1(x, y):
    """Element-wise L1 loss, mean over all elements."""
    return torch.abs(x - y).mean()


def l1_loss_v2(x, y):
    """Sum over last dim, then mean."""
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    """Weighted L2 loss (sqrt of weighted squared error)."""
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    """Weighted L2 loss with sum over last dim."""
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


# ============================================================
# Quaternion operations
# ============================================================

def quat_mult(q1, q2):
    """Hamilton quaternion multiplication (w, x, y, z convention)."""
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    sqrt(max(0, x)) with zero subgradient where x is 0.
    Source: PyTorch3D rotation_conversions.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    
    Args:
        matrix: (..., 3, 3) rotation matrices
    
    Returns:
        quaternions: (..., 4) with real part first
    
    Source: PyTorch3D rotation_conversions.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ], dim=-1)
    )

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], dim=-1),
    ], dim=-2)

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# ============================================================
# Gaussian rendering variable construction
# ============================================================

def transformed_params2rendervar(params, transformed_pts):
    """
    Convert Gaussian parameters to the dict expected by the rasterizer (colour render).
    
    Args:
        params: parameter dict
        transformed_pts: (N, 3) Gaussian centers in camera frame
    
    Returns:
        rendervar: dict for GaussianRasterizer
    """
    rendervar = {
        "means3D": transformed_pts,
        "rotations": F.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda") + 0,
    }
    if params["log_scales"].shape[1] == 1:
        # Simplified: isotropic scale + direct RGB
        rendervar["colors_precomp"] = params["rgb_colors"]
        rendervar["scales"] = torch.exp(torch.tile(params["log_scales"], (1, 3)))
    else:
        # Full: SH + anisotropic scale
        rendervar["shs"] = torch.cat(
            (
                params["rgb_colors"].reshape(params["rgb_colors"].shape[0], 3, -1).transpose(1, 2),
                params["feature_rest"].reshape(params["rgb_colors"].shape[0], 3, -1).transpose(1, 2),
            ),
            dim=1,
        )
        rendervar["scales"] = torch.exp(params["log_scales"])
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Compute depth and silhouette channels for each Gaussian center.
    
    Returns (N, 3) tensor: [depth_z, silhouette=1.0, depth_z^2]
    """
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1)
    depth_z_sq = torch.square(depth_z)

    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    return depth_silhouette


def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    """
    Convert Gaussian parameters to the dict for depth+silhouette rendering.
    """
    rendervar = {
        "means3D": transformed_pts,
        "colors_precomp": get_depth_and_silhouette(transformed_pts, w2c),
        "rotations": F.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda") + 0,
    }
    if params["log_scales"].shape[1] == 1:
        rendervar["scales"] = torch.exp(torch.tile(params["log_scales"], (1, 3)))
    else:
        rendervar["scales"] = torch.exp(params["log_scales"])
    return rendervar


# ============================================================
# INNOVATION 3: Deformation-aware transform_to_frame
# ============================================================

def transform_to_frame(params, time_idx, gaussians_grad, camera_grad,
                       apply_deformation=False, variables=None):
    """
    Transform Gaussians from world frame to camera frame at time_idx.
    
    Innovation 3: When apply_deformation=True, adds learned per-Gaussian
    position offsets for Gaussians classified as DEFORMING.
    
    Args:
        params: parameter dict (includes cam_unnorm_rots, cam_trans, means3D, etc.)
        time_idx: frame index for camera pose
        gaussians_grad: whether Gaussian positions get gradients
        camera_grad: whether camera pose gets gradients
        apply_deformation: apply deformation offsets (Innovation 3)
        variables: dict with 'deform_mask' (needed when apply_deformation=True)
    
    Returns:
        transformed_pts: (N, 3) Gaussian centers in camera frame
    """
    # Get camera pose
    if camera_grad:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx])
        cam_tran = params["cam_trans"][..., time_idx]
    else:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = params["cam_trans"][..., time_idx].detach()

    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Gaussian positions
    if gaussians_grad:
        pts = params["means3D"]
    else:
        pts = params["means3D"].detach()

    # Innovation 3: Apply deformation offsets to deforming Gaussians
    if (apply_deformation and "deform_offsets" in params
            and variables is not None and "deform_mask" in variables):
        deform_mask = variables["deform_mask"]
        if deform_mask.shape[0] == pts.shape[0] and deform_mask.any():
            pts = pts.clone()
            pts[deform_mask] = pts[deform_mask] + params["deform_offsets"][deform_mask]

    # Transform to camera frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts


def transform_to_frame_eval(params, camrt=None, rel_w2c=None):
    """
    Transform Gaussians to camera frame for evaluation (no deformation, no gradients).
    
    Args:
        params: parameter dict
        camrt: tuple of (cam_rot, cam_tran) quaternion + translation
        rel_w2c: pre-computed 4x4 world-to-camera matrix
    """
    if rel_w2c is None:
        cam_rot, cam_tran = camrt
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran

    pts = params["means3D"].detach()
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts
