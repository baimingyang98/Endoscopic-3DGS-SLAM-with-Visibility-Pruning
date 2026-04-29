"""Camera setup and image pre-filtering utilities."""
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def setup_camera(w, h, k, w2c, near=0.01, far=100, bg=(0, 0, 0), use_simplification=True):
    """
    Create a GaussianRasterizationSettings (Camera) object for the rasterizer.
    
    Args:
        w: image width
        h: image height
        k: 3x3 intrinsics matrix (numpy)
        w2c: 4x4 world-to-camera matrix (numpy)
        near: near clipping plane
        far: far clipping plane
        bg: background color (R, G, B)
        use_simplification: if True, use SH degree 0 (direct RGB)
    
    Returns:
        cam: GaussianRasterizationSettings object
    """
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)

    opengl_proj = torch.tensor([
        [2*fx/w,    0.0,       -(w - 2*cx)/w,              0.0],
        [0.0,       2*fy/h,    -(h - 2*cy)/h,              0.0],
        [0.0,       0.0,       far/(far - near),           -(far*near)/(far - near)],
        [0.0,       0.0,       1.0,                         0.0],
    ]).cuda().float().unsqueeze(0).transpose(1, 2)

    full_proj = w2c.bmm(opengl_proj)

    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0 if use_simplification else 3,
        campos=cam_center,
        prefiltered=False,
    )
    return cam


def energy_mask(color: torch.Tensor, th_low: float = 0.1, th_high: float = 0.9):
    """
    Compute a brightness-based mask to filter out very dark/bright pixels.
    
    Currently returns all-True mask (filtering disabled for C3VD).
    
    Args:
        color: (C, H, W) RGB tensor in [0, 1]
        th_low: lower brightness threshold
        th_high: upper brightness threshold
    
    Returns:
        mask: (1, H, W) boolean tensor
    """
    # Grayscale conversion weights
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=color.device).view(3, 1, 1)
    gray = torch.sum(color * weights, dim=0).detach()
    # Mask pixels within brightness range
    mask = ((gray >= th_low) & (gray <= th_high)).unsqueeze(0)
    # NOTE: Currently returns all-True for C3VD (no black borders)
    return torch.ones_like(mask).to(mask.device)
