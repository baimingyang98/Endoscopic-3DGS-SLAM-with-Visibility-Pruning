"""Projective geometry utilities for SE(3) transforms and camera operations."""
import torch
import numpy as np


def homogenize_points(pts: torch.Tensor) -> torch.Tensor:
    """Append ones column: (N, D) -> (N, D+1)."""
    ones = torch.ones(*pts.shape[:-1], 1, device=pts.device, dtype=pts.dtype)
    return torch.cat([pts, ones], dim=-1)


def inverse_transformation(trans: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert a 4x4 rigid-body transformation assuming orthogonal rotation.
    
    For T = [R|t; 0|1], T^{-1} = [R^T | -R^T t; 0 | 1]
    
    Args:
        trans: (..., 4, 4) transformation matrix
    
    Returns:
        inv_trans: (..., 4, 4) inverse transformation
    """
    R = trans[..., :3, :3]
    t = trans[..., :3, 3:]
    R_inv = R.transpose(-1, -2)
    t_inv = -R_inv @ t
    inv_trans = torch.zeros_like(trans)
    inv_trans[..., :3, :3] = R_inv
    inv_trans[..., :3, 3:] = t_inv
    inv_trans[..., 3, 3] = 1.0
    return inv_trans


def compose_transformations(trans_01: torch.Tensor, trans_12: torch.Tensor) -> torch.Tensor:
    """
    Compose two transformations: T_02 = T_01 @ T_12
    
    Args:
        trans_01: (..., 4, 4) 
        trans_12: (..., 4, 4)
    
    Returns:
        trans_02: (..., 4, 4)
    """
    return trans_01 @ trans_12


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    """
    Compute the relative transformation T_12 = T_01^{-1} @ T_02.
    
    Args:
        trans_01: (N, 4, 4) or (4, 4) reference transformation
        trans_02: (N, 4, 4) or (4, 4) destination transformation
        orthogonal_rotations: if True, use efficient inverse (transpose R)
    
    Returns:
        trans_12: relative transformation from frame 1 to frame 2
    """
    if orthogonal_rotations:
        trans_10 = inverse_transformation(trans_01)
    else:
        trans_10 = torch.inverse(trans_01)
    trans_12 = compose_transformations(trans_10, trans_02)
    return trans_12


def transform_pts_3d(pts: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points by a 4x4 SE(3) matrix.
    
    Args:
        pts: (N, 3) points
        transform: (4, 4) transformation matrix
    
    Returns:
        transformed: (N, 3)
    """
    pts_h = homogenize_points(pts)
    transformed = (transform @ pts_h.T).T
    return transformed[:, :3]


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    
    Args:
        quaternion: (..., 4) tensor with (w, x, y, z) convention
    
    Returns:
        rotation: (..., 3, 3)
    """
    q = quaternion / quaternion.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(dim=-1)

    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R
