"""Common utilities: seeding, parameter serialization, PLY export."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set seed for reproducibility across all RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set to: {seed}")


def params2cpu(params: dict) -> dict:
    """Convert all tensors in params dict to CPU numpy arrays."""
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params: dict, output_dir: str):
    """Save parameters as .npz file."""
    to_save = params2cpu(output_params)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)
    print(f"Saved parameters to: {save_path}")


def save_means3D(output_means: torch.Tensor, output_dir: str):
    """Save Gaussian centers as PLY point cloud."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "means3D.ply")
    assert output_means.shape[1] == 3, "Tensor must be of shape (N, 3)"

    points = output_means.detach().cpu().numpy()
    ply_header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(save_path, "w") as f:
        f.write(ply_header)
        np.savetxt(f, points, fmt="%f %f %f")
    print(f"Saved means3D to: {save_path}")


def save_params_ckpt(output_params: dict, output_dir: str, time_idx: int):
    """Save checkpoint at a specific time index."""
    to_save = params2cpu(output_params)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"params{time_idx}.npz")
    np.savez(save_path, **to_save)
    print(f"Saved checkpoint to: {save_path}")
