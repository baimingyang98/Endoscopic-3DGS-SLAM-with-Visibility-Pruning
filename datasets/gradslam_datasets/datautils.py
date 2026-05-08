"""Data manipulation utilities for image/intrinsics/pose processing."""
from typing import Union

import torch
import numpy as np


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):
    """Normalize image from [0, 255] to [0, 1]. Handles both tensors and numpy arrays."""
    if torch.is_tensor(rgb):
        return rgb.float() / 255.0
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255.0
    else:
        raise TypeError(f"Unsupported input rgb type: {type(rgb)}")


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):
    """Convert (..., H, W, C) to (..., C, H, W). Handles both tensors and numpy arrays."""
    if isinstance(rgb, np.ndarray):
        if rgb.ndim >= 3 and rgb.shape[-1] in (1, 3, 4):
            return np.moveaxis(rgb, -1, -3)
        return rgb
    if torch.is_tensor(rgb):
        if rgb.dim() >= 3 and rgb.shape[-1] in (1, 3, 4):
            return rgb.permute(*range(rgb.dim() - 3), -1, -3, -2)
        return rgb
    raise TypeError(f"Unsupported input rgb type: {type(rgb)}")


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
