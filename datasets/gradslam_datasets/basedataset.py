"""
Base dataset class for GradSLAM-style datasets.

Loads one RGB-D sequence at a time, with preprocessing (resize, normalize),
pose loading, and train/test split support.

Adapted from EndoGSLAM / NICE-SLAM / GradSLAM.
"""
import abc
import glob
import os
from typing import Optional, Union

import cv2
import imageio
import PIL
import numpy as np
import torch
from natsort import natsorted

from .geometryutils import relative_transformation
from . import datautils


def to_scalar(inp: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """Convert a single-element array/tensor to scalar."""
    if isinstance(inp, float):
        return inp
    if isinstance(inp, np.ndarray):
        assert inp.size == 1
        return inp.item()
    if isinstance(inp, torch.Tensor):
        assert inp.numel() == 1
        return inp.item()


def as_intrinsics_matrix(intrinsics):
    """Convert [fx, fy, cx, cy] to 3x3 intrinsics matrix."""
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def from_intrinsics_matrix(K):
    """Extract fx, fy, cx, cy from intrinsics matrix."""
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


class GradSLAMDataset(torch.utils.data.Dataset):
    """
    Abstract base class for RGB-D datasets used in Gaussian SLAM.
    
    Subclasses must implement:
        - get_filepaths(): returns color_paths, depth_paths, embedding_paths
        - load_poses(): returns list of 4x4 pose tensors
    """

    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = True,
        preload: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.load_embeddings = load_embeddings
        self.embedding_dir = embedding_dir
        self.embedding_dim = embedding_dim
        self.relative_pose = relative_pose
        self.preload = preload

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError(f"start must be non-negative. Got {start}.")
        if not (end == -1 or end > start):
            raise ValueError(f"end ({end}) must be -1 or greater than start ({start}).")

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"])
            if "distortion" in config_dict["camera_params"]
            else None
        )
        self.crop_size = config_dict["camera_params"].get("crop_size", None)
        self.crop_edge = config_dict["camera_params"].get("crop_edge", None)

        # Load file paths and poses
        self.color_paths, self.depth_paths, self.embedding_paths = self.get_filepaths()
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("Number of color and depth images must be the same.")
        self.num_imgs = len(self.color_paths)
        self.poses = self.load_poses()

        if self.end == -1:
            self.end = self.num_imgs

        # Apply train/test split and stride
        self.train_test_split(stride)
        self.num_imgs = len(self.color_paths)

        # Stack poses and compute relative transforms
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

        # Optionally preload all data into memory
        if self.preload:
            self.prepared_data = []
            self.prepare()

    def train_test_split(self, stride):
        """Default: apply start/end/stride slicing."""
        self.color_paths = self.color_paths[self.start:self.end:stride]
        self.depth_paths = self.depth_paths[self.start:self.end:stride]
        if self.load_embeddings:
            self.embedding_paths = self.embedding_paths[self.start:self.end:stride]
        self.poses = self.poses[self.start:self.end:stride]
        self.retained_inds = torch.arange(self.num_imgs)[self.start:self.end:stride]

    def __len__(self):
        return self.num_imgs

    @abc.abstractmethod
    def get_filepaths(self):
        """Return (color_paths, depth_paths, embedding_paths) lists."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_poses(self):
        """Return list of 4x4 torch.Tensor pose matrices."""
        raise NotImplementedError

    def _preprocess_color(self, color: np.ndarray) -> np.ndarray:
        """Resize color image to desired resolution.

        NOTE: matches the original EndoGSLAM behaviour (uses `and` not `or`
        in the resize condition). Some downstream code may depend on this.
        """
        if color.shape[0] != self.desired_height and color.shape[1] != self.desired_width:
            color = cv2.resize(
                color,
                (self.desired_width, self.desired_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """Resize depth, add channel dim, scale to metric units."""
        if depth.shape[0] != self.desired_height and depth.shape[1] != self.desired_width:
            depth = cv2.resize(
                depth.astype(float),
                (self.desired_width, self.desired_height),
                interpolation=cv2.INTER_NEAREST,
            )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.png_depth_scale

    def _preprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Convert absolute poses to relative (first frame = identity)."""
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def get_cam_K(self) -> torch.Tensor:
        """Return 3x3 camera intrinsics matrix."""
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        return torch.from_numpy(K)

    def get_pose(self, index):
        """Return pose for given index."""
        return self.transformed_poses[index].to(self.device).type(self.dtype)

    def prepare_meta(self, index):
        """Load and preprocess a single frame."""
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        # Load color
        color = np.asarray(imageio.imread(color_path), dtype=float)

        # Load depth
        # IMPORTANT: matches original EndoGSLAM behaviour exactly. The original
        # code had `if ".png" or '.jpg' in depth_path:` which always evaluates
        # to True (non-empty string is truthy), so it ALWAYS used imageio.imread
        # for .tiff files. Using PIL.Image.open instead gives different scaled
        # values for 16-bit TIFFs, breaking tracking.
        if ".npy" in depth_path:
            depth = np.load(depth_path).astype(np.float64)
        else:
            depth = np.asarray(imageio.imread(depth_path), dtype=np.float64)

        # Intrinsics
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])

        # Preprocess
        color = self._preprocess_color(color)
        if self.distortion is not None:
            color = cv2.undistort(color, K, self.distortion)
        color = torch.from_numpy(color)
        K = torch.from_numpy(K)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        # Scale intrinsics
        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                embedding.to(self.device),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )

    def prepare(self):
        """Preload all frames into memory."""
        for index in range(len(self.poses)):
            self.prepared_data.append(self.prepare_meta(index))

    def __getitem__(self, index):
        if self.preload:
            return self.prepared_data[index]
        return self.prepare_meta(index)
