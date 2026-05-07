"""
StereoMIS dataset loader for EndoGSLAM.

StereoMIS is a stereo endoscopic surgery dataset from MICCAI 2023:
Hayoz et al., "Pose Estimation and 3D Reconstruction of Vascular Structures
from Endoscopic Stereo Video"

Dataset: https://zenodo.org/records/7727692

Expected directory structure (verify after download):
    basedir/sequence/
        image_0/                  # left RGB frames (000000.png, 000001.png, ...)
        depth/ or disparity/      # depth maps (or disparity, needs conversion)
        groundtruth.txt           # camera poses (TUM format: timestamp tx ty tz qx qy qz qw)
        calibration.yaml          # intrinsics + baseline

NOTE: This loader is a TEMPLATE. You must verify and adjust:
    1. The actual subfolder names after extraction
    2. Pose file format (TUM vs KITTI vs custom)
    3. Whether depth is provided directly or as disparity
    4. The depth scale factor
"""
import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


def tum_pose_to_matrix(line):
    """
    Convert a TUM-format pose line (tx ty tz qx qy qz qw) to 4x4 matrix.
    """
    parts = line.strip().split()
    if len(parts) == 8:
        # timestamp tx ty tz qx qy qz qw
        _, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
    elif len(parts) == 7:
        tx, ty, tz, qx, qy, qz, qw = map(float, parts)
    else:
        raise ValueError(f"Unexpected pose format: {line}")

    # Quaternion to rotation matrix (Hamilton convention)
    n = qx*qx + qy*qy + qz*qz + qw*qw
    s = 2.0 / n
    R = np.array([
        [1 - s*(qy*qy + qz*qz),  s*(qx*qy - qz*qw),    s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),      1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),      s*(qy*qz + qx*qw),    1 - s*(qx*qx + qy*qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


class StereoMISDataset(GradSLAMDataset):
    """
    Dataset loader for StereoMIS.

    Args:
        config_dict: from configs/data/stereomis.yaml
        basedir: root data directory (e.g., './data/StereoMIS')
        sequence: sequence name (e.g., 'P1', 'P2')
        train_or_test: 'train', 'test', or 'all'
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 540,
        desired_width: Optional[int] = 675,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        train_or_test: Optional[str] = "all",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "groundtruth.txt")
        self.mode = train_or_test
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
        """Get sorted color/depth file paths."""
        # ADJUST these paths based on actual StereoMIS structure
        color_paths = natsorted(glob.glob(f"{self.input_folder}/image_0/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        if not depth_paths:
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.tiff"))

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """Load ground-truth camera poses (TUM format assumed)."""
        poses = []
        with open(self.pose_path, "r") as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]

        # Some StereoMIS sequences have one pose per frame; others are sparse.
        # If sparse, you may need interpolation. For now, assume 1-to-1 matching.
        for i in range(self.num_imgs):
            line = lines[i]
            T = tum_pose_to_matrix(line)
            poses.append(torch.from_numpy(T).float())
        return poses

    def train_test_split(self, stride):
        """
        Same convention as C3VD: every 8th frame is test.
        Adjust this if StereoMIS has its own train/test split convention.
        """
        all_idx = set(range(self.end))
        eval_idx = set(range(self.start + 7, self.end, 8))
        train_idx = all_idx - eval_idx
        eval_idx = sorted(list(eval_idx))
        train_idx = sorted(list(train_idx))

        if self.mode == "test":
            self.color_paths = [self.color_paths[i] for i in eval_idx]
            self.depth_paths = [self.depth_paths[i] for i in eval_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in eval_idx]
            self.poses = [self.poses[i] for i in eval_idx]
            self.retained_inds = torch.arange(self.num_imgs)[eval_idx]
        elif self.mode == "train":
            self.color_paths = [self.color_paths[i] for i in train_idx]
            self.depth_paths = [self.depth_paths[i] for i in train_idx]
            if self.load_embeddings:
                self.embedding_paths = [self.embedding_paths[i] for i in train_idx]
            self.poses = [self.poses[i] for i in train_idx]
            self.retained_inds = torch.arange(self.num_imgs)[train_idx]

            self.color_paths = self.color_paths[self.start::stride]
            self.depth_paths = self.depth_paths[self.start::stride]
            if self.load_embeddings:
                self.embedding_paths = self.embedding_paths[self.start::stride]
            self.poses = self.poses[self.start::stride]
            self.retained_inds = self.retained_inds[self.start::stride]
        else:
            super().train_test_split(stride)

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)
