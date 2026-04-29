"""EndoSLAM Unity dataset loader."""
import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rotation_matrix = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)],
    ])
    return rotation_matrix


def create_transformation_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Create 4x4 transformation matrix from translation + quaternion."""
    rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T = np.zeros((4, 4))
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [tx, ty, tz]
    T[3, 3] = 1
    # Z-axis inversion for Unity coordinate system
    z_inv = np.diag([1, 1, -1, 1])
    T = np.dot(T, z_inv)
    return T


class EndoSLAMDataset(GradSLAMDataset):
    """Dataset loader for EndoSLAM Unity synthetic data."""

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 320,
        desired_width: Optional[int] = 320,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = glob.glob(os.path.join(self.input_folder, "Poses/*.csv"))[0]
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        """Get sorted file paths for EndoSLAM format."""
        color_paths = natsorted(glob.glob(f"{self.input_folder}/Frames/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/Pixelwise Depths/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """Load poses from CSV (tx, ty, tz, qx, qy, qz, qw)."""
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()[1:]  # Skip header
        for i in range(self.num_imgs - 1):
            line = lines[i]
            tx, ty, tz, rx, ry, rz, rw, _ = map(float, line.split(sep=","))
            c2w = create_transformation_matrix(tx, ty, tz, rx, ry, rz, rw)
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        """Load pre-computed embeddings."""
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)
