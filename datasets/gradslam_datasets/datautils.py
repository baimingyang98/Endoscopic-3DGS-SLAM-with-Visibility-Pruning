"""Data manipulation utilities for image/intrinsics/pose processing."""
import torch
import numpy as np


def normalize_image(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize image from [0, 255] to [0, 1]."""
    return rgb.float() / 255.0


def channels_first(image: torch.Tensor) -> torch.Tensor:
    """Convert (H, W, C) to (C, H, W)."""
    if image.dim() == 3 and image.shape[-1] in (1, 3, 4):
        return image.permute(2, 0, 1)
    return image


def scale_intrinsics(intrinsics: torch.Tensor, sy: float, sx: float) -> torch.Tensor:
    """Scale camera intrinsics by factors (sy, sx)."""
    scaled = intrinsics.clone()
    scaled[0, 0] *= sx  # fx
    scaled[1, 1] *= sy  # fy
    scaled[0, 2] *= sx  # cx
    scaled[1, 2] *= sy  # cy
    return scaled


def as_intrinsics_matrix(intrinsics_params) -> np.ndarray:
    """Convert [fx, fy, cx, cy] to 3x3 intrinsics matrix."""
    fx, fy, cx, cy = intrinsics_params
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K
