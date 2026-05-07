"""
StereoMIS dataset loader.

Loads PREPROCESSED StereoMIS data (via scripts/preprocess_stereomis.py).

Expected structure (after running preprocess_stereomis.py):
    basedir/sequence/
        color/000241.png        # rectified left RGB
        depth/000241.tiff       # depth in mm (uint16)
        pose.txt                # 4x4 poses, one per line, 16 comma-separated floats
        intrinsics.yaml         # auto-generated rectified intrinsics
"""
import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class StereoMISDataset(GradSLAMDataset):
    """
    Dataset loader for preprocessed StereoMIS sequences.
    
    Args:
        config_dict: from configs/data/stereomis.yaml
        basedir: e.g., './data/StereoMIS'
        sequence: e.g., 'P1', 'P2_0', 'P3'
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
        desired_height: Optional[int] = 512,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        train_or_test: Optional[str] = "all",
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "pose.txt")
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.tiff"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """
        Load 4x4 poses from pose.txt (same format as C3VD: 16 comma-separated floats per line).
        """
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            pose = list(map(float, line.split(sep=",")))
            pose = torch.Tensor(pose).reshape(4, 4).float()
            poses.append(pose)
        return poses

    def train_test_split(self, stride):
        """Same convention as C3VD: every 8th frame is test."""
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
