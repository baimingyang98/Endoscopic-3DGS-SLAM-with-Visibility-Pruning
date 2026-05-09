"""
EndoGSLAM with Innovations - Main SLAM Pipeline

This is the core RGB-D SLAM entry point implementing:
- Standard EndoGSLAM pipeline (tracking, mapping, densification)
- Innovation 1: Visibility-aware dual-mask pruning
- Innovation 2: Periodic bundle adjustment
- Innovation 3: Deformation modeling (experimental)

Usage:
    python scripts/main.py configs/c3vd/c3vd_innovations.py
"""
import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

# Project root
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.gradslam_datasets import load_dataset_config, EndoSLAMDataset, C3VDDataset, StereoMISDataset
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save
from utils.keyframe_selection import keyframe_selection_overlap, keyframe_selection_distance
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion,
)
from utils.slam_external import (
    calc_ssim, build_rotation, prune_gaussians, densify, update_three_way_classifier,
)
from utils.vis_utils import plot_video

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


# ============================================================
# Dataset factory
# ============================================================

def get_dataset(config_dict, basedir, sequence, **kwargs):
    """Create dataset based on config."""
    name = config_dict["dataset_name"].lower()
    if name in ["endoslam_unity"]:
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif name in ["c3vd"]:
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    elif name in ["stereomis"]:
        return StereoMISDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================
# Point cloud initialization
# ============================================================

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    """
    Back-project RGB-D frame to colored 3D point cloud.
    
    Args:
        color: (C, H, W) RGB image
        depth: (1, H, W) depth map
        intrinsics: (3, 3) camera intrinsics
        w2c: (4, 4) world-to-camera transform
        transform_pts: transform points to world frame
        mask: (H*W,) boolean mask for valid pixels
        compute_mean_sq_dist: compute per-point scale estimate
        mean_sq_dist_method: method for scale computation
    
    Returns:
        point_cld: (N, 6) [x, y, z, r, g, b]
        mean3_sq_dist: (N,) scale estimates (if compute_mean_sq_dist)
    """
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    x_grid, y_grid = torch.meshgrid(
        torch.arange(width).cuda().float(),
        torch.arange(height).cuda().float(),
        indexing="xy",
    )
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Camera-frame points
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)

    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Scale estimation
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian ** 2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method: {mean_sq_dist_method}")

    # Colorize
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
    point_cld = torch.cat((pts, cols), -1)

    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    return point_cld


# ============================================================
# Parameter initialization
# ============================================================

def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True):
    """
    Initialize Gaussian parameters from first-frame point cloud.
    
    Returns:
        params: dict of torch.nn.Parameter
        variables: dict of tracking variables
    """
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")

    params = {
        "means3D": means3D,
        "rgb_colors": init_pt_cld[:, 3:6],
        "unnorm_rotations": unnorm_rots,
        "logit_opacities": logit_opacities,
        "log_scales": torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None],
            (1, 1 if use_simplification else 3),
        ),
    }
    if not use_simplification:
        params["feature_rest"] = torch.zeros(num_pts, 45)

    # Camera pose parameters (per-frame quaternion + translation)
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params["cam_unnorm_rots"] = cam_rots
    params["cam_trans"] = np.zeros((1, 3, num_frames))

    # Convert to CUDA Parameters
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {
        "max_2D_radius": torch.zeros(num_pts).cuda().float(),
        "means2D_gradient_accum": torch.zeros(num_pts).cuda().float(),
        "denom": torch.zeros(num_pts).cuda().float(),
        "timestep": torch.zeros(num_pts).cuda().float(),
        "deform_mask": torch.zeros(num_pts, dtype=torch.bool, device="cuda"),
    }

    return params, variables


def initialize_optimizer(params, lrs_dict):
    """Create Adam optimizer with per-parameter learning rates."""
    param_groups = [
        {"params": [v], "name": k, "lr": lrs_dict.get(k, 0.0)}
        for k, v in params.items() if k != "feature_rest"
    ]
    if "feature_rest" in params:
        param_groups.append({
            "params": [params["feature_rest"]],
            "name": "feature_rest",
            "lr": lrs_dict.get("rgb_colors", 0.0) / 20.0,
        })
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio,
                               mean_sq_dist_method, densify_dataset=None,
                               use_simplification=True, enable_deformation=False):
    """Initialize parameters from the first frame of the dataset.

    Args:
        enable_deformation: if True, allocates Innovation 3's per-Gaussian
            position offsets. When False (default), deform_offsets is NOT
            created so it cannot be silently picked up by transform_to_frame.
            This was the source of the asymmetric mapping/tracking bug.
    """
    color, depth, intrinsics, pose = dataset[0]
    color = color.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1)
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    cam = setup_camera(
        color.shape[2], color.shape[1],
        intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(),
        use_simplification=use_simplification,
    )

    if densify_dataset is not None:
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(
            color.shape[2], color.shape[1],
            densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(),
        )
    else:
        densify_intrinsics = intrinsics

    # Create initial point cloud
    mask = (depth > 0) & energy_mask(color)
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(
        color, depth, densify_intrinsics, w2c,
        mask=mask, compute_mean_sq_dist=True,
        mean_sq_dist_method=mean_sq_dist_method,
    )

    # Initialize Gaussians
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)
    variables["scene_radius"] = torch.max(depth) / scene_radius_depth_ratio

    # Innovation 3: Initialize deformation offsets ONLY when enabled.
    # Previously this was created unconditionally, which (combined with
    # apply_deformation=True being passed unconditionally during mapping)
    # caused tracking to render against means3D while mapping rendered
    # against means3D + offset, producing systematic pose drift (ATE blowup).
    if enable_deformation:
        params["deform_offsets"] = torch.nn.Parameter(
            torch.zeros(params["means3D"].shape[0], 3, device="cuda").requires_grad_(True)
        )

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    return params, variables, intrinsics, w2c, cam


# ============================================================
# Loss computation
# ============================================================

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights,
             use_sil_for_loss, sil_thres, use_l1, ignore_outlier_depth_loss,
             tracking=False, mapping=False, do_ba=False):
    """
    Compute combined RGB + depth loss for tracking or mapping.
    
    Returns:
        loss: scalar loss
        variables: updated variables (with gauss_vis)
        weighted_losses: dict of individual loss components
    """
    losses = {}

    # Innovation 3 gating: only apply deformation when deform_offsets is
    # actually a parameter AND a non-trivial deform_mask exists.
    # CRITICAL: tracking and mapping must use the SAME geometry, otherwise
    # systematic pose drift accumulates (was the cause of the ATE blowup).
    apply_def = (
        "deform_offsets" in params
        and variables is not None
        and "deform_mask" in variables
        and variables["deform_mask"].any().item()
    )
    def_vars = variables if apply_def else None

    # Transform Gaussians to camera frame with appropriate gradients
    if tracking:
        # Tracking: deformation MUST match what mapping uses (same geometry)
        transformed_pts = transform_to_frame(
            params, iter_time_idx, gaussians_grad=False, camera_grad=True,
            apply_deformation=apply_def, variables=def_vars,
        )
    elif mapping:
        if do_ba:
            transformed_pts = transform_to_frame(
                params, iter_time_idx, gaussians_grad=True, camera_grad=True,
                apply_deformation=apply_def, variables=def_vars,
            )
        else:
            transformed_pts = transform_to_frame(
                params, iter_time_idx, gaussians_grad=True, camera_grad=False,
                apply_deformation=apply_def, variables=def_vars,
            )
    else:
        transformed_pts = transform_to_frame(
            params, iter_time_idx, gaussians_grad=True, camera_grad=False,
            apply_deformation=apply_def, variables=def_vars,
        )

    # Render RGB
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data["w2c"], transformed_pts
    )

    rendervar["means2D"].retain_grad()
    im, radius, _, gauss_vis = Renderer(raster_settings=curr_data["cam"])(**rendervar)
    variables["means2D"] = rendervar["means2D"]

    # Render depth & silhouette
    depth_sil, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = silhouette > sil_thres
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth ** 2
    uncertainty = uncertainty.detach()

    # Validity mask
    nan_mask = ~torch.isnan(depth) & ~torch.isnan(uncertainty)
    bg_mask = energy_mask(curr_data["im"])
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data["depth"] - depth) * (curr_data["depth"] > 0)
        mask = (depth_error < 20 * depth_error.mean()) & (curr_data["depth"] > 0)
    else:
        mask = curr_data["depth"] > 0
    mask = mask & nan_mask & bg_mask
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses["depth"] = torch.abs(curr_data["depth"] - depth)[mask].sum()
        else:
            losses["depth"] = torch.abs(curr_data["depth"] - depth)[mask].mean()

    # RGB loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1)).detach()
        losses["im"] = torch.abs(curr_data["im"] - im)[color_mask].sum()
    elif tracking:
        losses["im"] = torch.abs(curr_data["im"] - im).sum()
    else:
        losses["im"] = 0.8 * l1_loss_v1(im, curr_data["im"]) + 0.2 * (1.0 - calc_ssim(im, curr_data["im"]))

    # Weighted sum
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    # Innovation 3: Deformation regularization
    if mapping and "deform_offsets" in params:
        deform_mag = torch.norm(params["deform_offsets"], dim=1).mean()
        lambda_mag = 0.01
        loss = loss + lambda_mag * deform_mag

        if "prev_deform_offsets" in variables:
            prev_offsets = variables["prev_deform_offsets"]
            if prev_offsets.shape[0] == params["deform_offsets"].shape[0]:
                temporal_diff = torch.norm(params["deform_offsets"] - prev_offsets, dim=1).mean()
                lambda_temp = 0.005
                loss = loss + lambda_temp * temporal_diff

    # Update tracking variables
    seen = radius > 0
    variables["max_2D_radius"][seen] = torch.max(radius[seen], variables["max_2D_radius"][seen])
    variables["seen"] = seen
    variables["gauss_vis"] = gauss_vis.detach()
    weighted_losses["loss"] = loss

    return loss, variables, weighted_losses


# ============================================================
# Gaussian densification (add new Gaussians)
# ============================================================

def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    """Create parameters for newly densified Gaussians."""
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5

    params = {
        "means3D": means3D,
        "rgb_colors": new_pt_cld[:, 3:6],
        "unnorm_rotations": unnorm_rots,
        "logit_opacities": logit_opacities,
        "log_scales": torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None],
            (1, 1 if use_simplification else 3),
        ),
    }
    if not use_simplification:
        params["feature_rest"] = torch.zeros(num_pts, 45)

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, use_simplification=True):
    """Add new Gaussians based on silhouette/depth non-presence detection."""
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data["w2c"], transformed_pts
    )
    depth_sil, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = silhouette < sil_thres

    # Depth-based detection
    gt_depth = curr_data["depth"][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20 * depth_error.mean())
    non_presence_mask = (non_presence_sil_mask | non_presence_depth_mask).reshape(-1)

    if torch.sum(non_presence_mask) > 0:
        curr_cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        curr_cam_tran = params["cam_trans"][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran

        valid_depth_mask = (curr_data["depth"][0] > 0) & (curr_data["depth"][0] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data["im"]).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)

        new_pt_cld, mean3_sq_dist = get_pointcloud(
            curr_data["im"], curr_data["depth"], curr_data["intrinsics"],
            curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method,
        )
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)

        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

        num_pts = params["means3D"].shape[0]
        n_new = new_pt_cld.shape[0]
        variables["means2D_gradient_accum"] = torch.zeros(num_pts, device="cuda").float()
        variables["denom"] = torch.zeros(num_pts, device="cuda").float()
        variables["max_2D_radius"] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx * torch.ones(n_new, device="cuda").float()
        variables["timestep"] = torch.cat((variables["timestep"], new_timestep), dim=0)

        # FIX C: extend visibility-buffer state for new Gaussians.
        # Newly-added Gaussians have no observation history, so initialize
        # them with zero visibility but ALSO start their frame_count at 0,
        # so they are not eligible for visibility pruning until they have
        # been observed for `min_observations` frames.
        if "vis_history" in variables:
            W_buf = variables["vis_history"].shape[1]
            new_hist = torch.zeros(n_new, W_buf, device="cuda")
            variables["vis_history"] = torch.cat([variables["vis_history"], new_hist], dim=0)
        if "vis_frame_count" in variables:
            new_cnt = torch.zeros(n_new, device="cuda")
            variables["vis_frame_count"] = torch.cat([variables["vis_frame_count"], new_cnt], dim=0)
        if "deform_mask" in variables:
            new_dm = torch.zeros(n_new, dtype=torch.bool, device="cuda")
            variables["deform_mask"] = torch.cat([variables["deform_mask"], new_dm], dim=0)

        # Innovation 3: Extend deformation offsets
        if "deform_offsets" in params:
            new_deform = torch.zeros(n_new, 3, device="cuda")
            params["deform_offsets"] = torch.nn.Parameter(
                torch.cat((params["deform_offsets"], new_deform), dim=0).requires_grad_(True)
            )

    return params, variables


# ============================================================
# Camera pose initialization
# ============================================================

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    """Initialize camera pose using constant velocity model."""
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            prev_rot1 = F.normalize(params["cam_unnorm_rots"][..., curr_time_idx - 1].detach())
            prev_rot2 = F.normalize(params["cam_unnorm_rots"][..., curr_time_idx - 2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params["cam_unnorm_rots"][..., curr_time_idx] = new_rot.detach()

            prev_tran1 = params["cam_trans"][..., curr_time_idx - 1].detach()
            prev_tran2 = params["cam_trans"][..., curr_time_idx - 2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params["cam_trans"][..., curr_time_idx] = new_tran.detach()
        else:
            params["cam_unnorm_rots"][..., curr_time_idx] = params["cam_unnorm_rots"][..., curr_time_idx - 1].detach()
            params["cam_trans"][..., curr_time_idx] = params["cam_trans"][..., curr_time_idx - 1].detach()
    return params


def convert_params_to_store(params):
    """Detach and clone parameters for checkpointing."""
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in params.items()}


# ============================================================
# INNOVATION 2: Periodic Bundle Adjustment
# ============================================================

def select_ba_keyframes(keyframe_list, n_keyframes, strategy="hybrid"):
    """
    Select keyframes for periodic bundle adjustment.
    
    Strategies:
    - 'uniform': uniformly spaced
    - 'recent': most recent
    - 'hybrid': half recent + half uniform from older frames
    """
    n_available = len(keyframe_list)
    if n_available <= n_keyframes:
        return list(range(n_available))

    if strategy == "uniform":
        indices = np.linspace(0, n_available - 1, n_keyframes, dtype=int).tolist()
        return sorted(set(indices))
    elif strategy == "recent":
        return list(range(n_available - n_keyframes, n_available))
    elif strategy == "hybrid":
        n_recent = n_keyframes // 2
        n_old = n_keyframes - n_recent
        recent = list(range(n_available - n_recent, n_available))
        old_pool = list(range(0, n_available - n_recent))
        if len(old_pool) <= n_old:
            old = old_pool
        else:
            old = np.random.choice(old_pool, n_old, replace=False).tolist()
        return sorted(set(recent + old))
    else:
        indices = np.linspace(0, n_available - 1, n_keyframes, dtype=int).tolist()
        return sorted(set(indices))


def periodic_bundle_adjustment(params, variables, keyframe_list, config,
                                intrinsics, cam, first_frame_w2c):
    """
    Innovation 2: Joint optimization of camera poses and Gaussian parameters.
    
    Triggered periodically (every M frames) to reduce accumulated pose drift.
    Uses conservative learning rates for camera parameters.
    """
    innovation_cfg = config.get("innovations", {})
    ba_num_iters = innovation_cfg.get("ba_num_iters", 50)
    ba_n_keyframes = innovation_cfg.get("ba_n_keyframes", 10)
    ba_strategy = innovation_cfg.get("ba_selection", "hybrid")

    selected_kf_indices = select_ba_keyframes(keyframe_list, ba_n_keyframes, ba_strategy)
    if len(selected_kf_indices) < 2:
        return params, variables

    print(f"  BA: optimizing over {len(selected_kf_indices)} keyframes for {ba_num_iters} iterations")

    optimizer = initialize_optimizer(params, config["mapping"]["lrs"])

    for ba_iter in range(ba_num_iters):
        rand_idx = np.random.randint(0, len(selected_kf_indices))
        kf_idx = selected_kf_indices[rand_idx]
        kf = keyframe_list[kf_idx]

        iter_time_idx = kf["id"]
        iter_color = kf["color"]
        iter_depth = kf["depth"]

        iter_data = {
            "cam": cam, "im": iter_color, "depth": iter_depth, "id": iter_time_idx,
            "intrinsics": intrinsics, "w2c": first_frame_w2c, "iter_gt_w2c_list": None,
        }

        loss, variables, losses = get_loss(
            params, iter_data, variables, iter_time_idx,
            config["mapping"]["loss_weights"],
            config["mapping"]["use_sil_for_loss"],
            config["mapping"]["sil_thres"],
            config["mapping"]["use_l1"],
            config["mapping"]["ignore_outlier_depth_loss"],
            mapping=True, do_ba=True,
        )

        loss.backward()
        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    print(f"  BA: completed, final loss = {loss.item():.6f}")
    return params, variables


# ============================================================
# MAIN SLAM LOOP
# ============================================================

def rgbd_slam(config: dict):
    """
    Main RGB-D SLAM pipeline with all three innovations.
    
    Pipeline per frame:
    1. Load RGB-D data
    2. Initialize camera pose (constant velocity model)
    3. Tracking: optimize camera pose
    4. Mapping: densify Gaussians, optimize map + run innovation pruning
    5. (Periodic) Bundle adjustment
    6. Keyframe management
    """
    # Config defaults
    if "use_depth_loss_thres" not in config["tracking"]:
        config["tracking"]["use_depth_loss_thres"] = False
        config["tracking"]["depth_loss_thres"] = 100000
    if "visualize_tracking_loss" not in config["tracking"]:
        config["tracking"]["visualize_tracking_loss"] = False

    print(f"Loaded Config:\n{config}")

    # Output directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device(config["primary_device"])

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "distance_keyframe_selection" not in config:
        config["distance_keyframe_selection"] = False
    if config["distance_keyframe_selection"]:
        if "distance_current_frame_prob" not in config:
            config["distance_current_frame_prob"] = 0.5
    if "gaussian_simplification" not in config:
        config["gaussian_simplification"] = True

    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    if "train_or_test" not in dataset_config:
        dataset_config["train_or_test"] = "all"
    if "preload" not in dataset_config:
        dataset_config["preload"] = False
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True

    # Resolution settings
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        seperate_densification_res = (
            dataset_config["densification_image_height"] != dataset_config["desired_image_height"]
            or dataset_config["densification_image_width"] != dataset_config["desired_image_width"]
        )

    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        seperate_tracking_res = (
            dataset_config["tracking_image_height"] != dataset_config["desired_image_height"]
            or dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]
        )

    # Create dataset
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        train_or_test=dataset_config["train_or_test"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Evaluation dataset (test split)
    eval_dataset = None
    if dataset_config["train_or_test"] == "train":
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test="test",
        )

    # Initialize parameters
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            preload=dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"],
        )
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(
                dataset, num_frames, config["scene_radius_depth_ratio"],
                config["mean_sq_dist_method"], densify_dataset=densify_dataset,
                use_simplification=config["gaussian_simplification"],
                enable_deformation=config.get("innovations", {}).get("enable_deformation", False),
            )
    else:
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
            dataset, num_frames, config["scene_radius_depth_ratio"],
            config["mean_sq_dist_method"],
            use_simplification=config["gaussian_simplification"],
            enable_deformation=config.get("innovations", {}).get("enable_deformation", False),
        )

    # Tracking resolution dataset
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            preload=dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(
            tracking_color.shape[2], tracking_color.shape[1],
            tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(),
            use_simplification=config["gaussian_simplification"],
        )

    # --------------------------------------------------------
    # State tracking
    # --------------------------------------------------------
    keyframe_list = []
    keyframe_time_indices = []
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Checkpoint loading
    if config["load_checkpoint"]:
        checkpoint_time_idx = config["checkpoint_time_idx"]
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config["workdir"], config["run_name"], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables["max_2D_radius"] = torch.zeros(params["means3D"].shape[0]).cuda().float()
        variables["means2D_gradient_accum"] = torch.zeros(params["means3D"].shape[0]).cuda().float()
        variables["denom"] = torch.zeros(params["means3D"].shape[0]).cuda().float()
        variables["timestep"] = torch.zeros(params["means3D"].shape[0]).cuda().float()
        keyframe_time_indices = np.load(
            os.path.join(config["workdir"], config["run_name"], f"keyframe_time_indices{checkpoint_time_idx}.npy")
        ).tolist()
        for t_idx in range(checkpoint_time_idx):
            color, depth, _, gt_pose = dataset[t_idx]
            gt_w2c_all_frames.append(torch.linalg.inv(gt_pose))
            if t_idx in keyframe_time_indices:
                curr_cam_rot = F.normalize(params["cam_unnorm_rots"][..., t_idx].detach())
                curr_cam_tran = params["cam_trans"][..., t_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                keyframe_list.append({"id": t_idx, "est_w2c": curr_w2c, "color": color, "depth": depth})
    else:
        checkpoint_time_idx = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load frame
        color, depth, _, gt_pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(gt_pose)
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        iter_time_idx = time_idx

        curr_data = {
            "cam": cam, "im": color, "depth": depth, "id": iter_time_idx,
            "intrinsics": intrinsics, "w2c": first_frame_w2c, "iter_gt_w2c_list": curr_gt_w2c,
        }

        # Tracking data (potentially different resolution)
        if seperate_tracking_res:
            tracking_color_f, tracking_depth_f, _, _ = tracking_dataset[time_idx]
            tracking_color_f = tracking_color_f.permute(2, 0, 1) / 255
            tracking_depth_f = tracking_depth_f.permute(2, 0, 1)
            tracking_curr_data = {
                "cam": tracking_cam, "im": tracking_color_f, "depth": tracking_depth_f,
                "id": iter_time_idx, "intrinsics": tracking_intrinsics,
                "w2c": first_frame_w2c, "iter_gt_w2c_list": curr_gt_w2c,
            }
        else:
            tracking_curr_data = curr_data

        num_iters_mapping = config["mapping"]["num_iters"]

        # Initialize camera pose
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config["tracking"]["forward_prop"])

        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------
        tracking_start_time = time.time()
        if time_idx > 0 and not config["tracking"]["use_gt_poses"]:
            optimizer = initialize_optimizer(params, config["tracking"]["lrs"])
            candidate_cam_unnorm_rot = params["cam_unnorm_rots"][..., time_idx].detach().clone()
            candidate_cam_tran = params["cam_trans"][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            num_iters_tracking = config["tracking"]["num_iters"]
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Step: {time_idx}")
            iter = 0
            do_continue_slam = False

            while True:
                iter_start_time = time.time()
                loss, variables, losses = get_loss(
                    params, tracking_curr_data, variables, iter_time_idx,
                    config["tracking"]["loss_weights"],
                    config["tracking"]["use_sil_for_loss"],
                    config["tracking"]["sil_thres"],
                    config["tracking"]["use_l1"],
                    config["tracking"]["ignore_outlier_depth_loss"],
                    tracking=True,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params["cam_unnorm_rots"][..., time_idx].detach().clone()
                        candidate_cam_tran = params["cam_trans"][..., time_idx].detach().clone()
                    if config["report_iter_progress"]:
                        report_progress(params, tracking_curr_data, iter + 1, progress_bar,
                                       iter_time_idx, sil_thres=config["tracking"]["sil_thres"], tracking=True)
                    else:
                        progress_bar.update(1)

                tracking_iter_time_sum += time.time() - iter_start_time
                tracking_iter_time_count += 1
                iter += 1

                if iter == num_iters_tracking:
                    if losses["depth"] < config["tracking"]["depth_loss_thres"] and config["tracking"]["use_depth_loss_thres"]:
                        break
                    elif config["tracking"]["use_depth_loss_thres"] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Step: {time_idx}")
                        num_iters_tracking = 2 * num_iters_tracking
                    else:
                        break

            progress_bar.close()
            with torch.no_grad():
                params["cam_unnorm_rots"][..., time_idx] = candidate_cam_unnorm_rot
                params["cam_trans"][..., time_idx] = candidate_cam_tran

        elif time_idx > 0 and config["tracking"]["use_gt_poses"]:
            with torch.no_grad():
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                params["cam_unnorm_rots"][..., time_idx] = rel_w2c_rot_quat
                params["cam_trans"][..., time_idx] = rel_w2c_tran

        tracking_frame_time_sum += time.time() - tracking_start_time
        tracking_frame_time_count += 1

        # ----------------------------------------------------
        # MAPPING (Densification + Optimization)
        # ----------------------------------------------------
        if time_idx == 0 or (time_idx + 1) % config["map_every"] == 0:
            # Densification
            if config["mapping"]["add_new_gaussians"] and time_idx > 0:
                if seperate_densification_res:
                    densify_color_f, densify_depth_f, _, _ = densify_dataset[time_idx]
                    densify_color_f = densify_color_f.permute(2, 0, 1) / 255
                    densify_depth_f = densify_depth_f.permute(2, 0, 1)
                    densify_curr_data = {
                        "cam": densify_cam, "im": densify_color_f, "depth": densify_depth_f,
                        "id": time_idx, "intrinsics": densify_intrinsics,
                        "w2c": first_frame_w2c, "iter_gt_w2c_list": curr_gt_w2c,
                    }
                else:
                    densify_curr_data = curr_data

                params, variables = add_new_gaussians(
                    params, variables, densify_curr_data,
                    config["mapping"]["sil_thres"], time_idx,
                    config["mean_sq_dist_method"],
                    config["gaussian_simplification"],
                )

            # Keyframe selection for mapping window
            if not config["distance_keyframe_selection"]:
                with torch.no_grad():
                    curr_cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
                    curr_cam_tran = params["cam_trans"][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    num_keyframes = config["mapping_window_size"] - 2
                    selected_keyframes = keyframe_selection_overlap(
                        depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes
                    )
                    selected_time_idx = [keyframe_list[f_idx]["id"] for f_idx in selected_keyframes]
                    if len(keyframe_list) > 0:
                        selected_time_idx.append(keyframe_list[-1]["id"])
                        selected_keyframes.append(len(keyframe_list) - 1)
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Optimizer for mapping
            optimizer = initialize_optimizer(params, config["mapping"]["lrs"])

            # Mapping iterations
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Step: {time_idx}")

            actural_keyframe_ids = []
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                # Select frame for this iteration
                if not config["distance_keyframe_selection"]:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]["id"]
                        iter_color = keyframe_list[selected_rand_keyframe_idx]["color"]
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]["depth"]
                else:
                    if len(actural_keyframe_ids) == 0:
                        if len(keyframe_list) > 0:
                            curr_position = params["cam_trans"][..., time_idx].detach().cpu()
                            actural_keyframe_ids = keyframe_selection_distance(
                                time_idx, curr_position, keyframe_list,
                                config["distance_current_frame_prob"], num_iters_mapping,
                            )
                        else:
                            actural_keyframe_ids = [0] * num_iters_mapping

                    selected_keyframe_ids = actural_keyframe_ids[iter]
                    if selected_keyframe_ids == len(keyframe_list):
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        iter_time_idx = keyframe_list[selected_keyframe_ids]["id"]
                        iter_color = keyframe_list[selected_keyframe_ids]["color"]
                        iter_depth = keyframe_list[selected_keyframe_ids]["depth"]

                iter_gt_w2c = gt_w2c_all_frames[: iter_time_idx + 1]
                iter_data = {
                    "cam": cam, "im": iter_color, "depth": iter_depth, "id": iter_time_idx,
                    "intrinsics": intrinsics, "w2c": first_frame_w2c, "iter_gt_w2c_list": iter_gt_w2c,
                }

                # Mapping loss
                loss, variables, losses = get_loss(
                    params, iter_data, variables, iter_time_idx,
                    config["mapping"]["loss_weights"],
                    config["mapping"]["use_sil_for_loss"],
                    config["mapping"]["sil_thres"],
                    config["mapping"]["use_l1"],
                    config["mapping"]["ignore_outlier_depth_loss"],
                    mapping=True,
                )
                loss.backward()

                with torch.no_grad():
                    innovation_cfg = config.get("innovations", {})

                    # Fix C: update visibility classifier BEFORE prune/densify
                    # so vis_history stays index-aligned with the rasterizer's
                    # gauss_vis from this iteration. If we update after
                    # prune/densify, the Gaussian count has changed and rows
                    # get mis-assigned (leading to newly-cloned Gaussians being
                    # immediately mis-classified as floaters).
                    if innovation_cfg.get("enable_visibility_pruning", False) and "gauss_vis" in variables:
                        variables = update_three_way_classifier(
                            variables, variables["gauss_vis"], innovation_cfg
                        )

                    # Innovation 1: Enhanced pruning
                    if config["mapping"]["prune_gaussians"]:
                        prune_transformed_pts = None
                        if innovation_cfg.get("enable_visibility_pruning", False):
                            prune_transformed_pts = transform_to_frame(
                                params, iter_time_idx, gaussians_grad=False, camera_grad=False
                            )
                        params, variables = prune_gaussians(
                            params, variables, optimizer, iter,
                            config["mapping"]["pruning_dict"],
                            innovation_config=innovation_cfg,
                            curr_data=iter_data,
                            transformed_pts=prune_transformed_pts,
                        )

                    # Standard densification
                    if config["mapping"]["use_gaussian_splatting_densification"]:
                        params, variables = densify(params, variables, optimizer, iter, config["mapping"]["densify_dict"])

                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Innovation 3: Store previous deformation offsets
                    if innovation_cfg.get("enable_deformation", False) and "deform_offsets" in params:
                        variables["prev_deform_offsets"] = params["deform_offsets"].detach().clone()

                    # Progress
                    if config["report_iter_progress"]:
                        report_progress(params, iter_data, iter + 1, progress_bar,
                                       iter_time_idx, sil_thres=config["mapping"]["sil_thres"],
                                       mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)

                mapping_iter_time_sum += time.time() - iter_start_time
                mapping_iter_time_count += 1

            if num_iters_mapping > 0:
                progress_bar.close()

            mapping_frame_time_sum += time.time() - mapping_start_time
            mapping_frame_time_count += 1

            # Innovation 2: Periodic Bundle Adjustment
            innovation_cfg = config.get("innovations", {})
            if (innovation_cfg.get("enable_periodic_ba", False)
                    and time_idx > 0
                    and (time_idx + 1) % innovation_cfg.get("ba_every_m_frames", 50) == 0
                    and len(keyframe_list) >= 2):
                print(f"\n--- Periodic BA at frame {time_idx} ---")
                params, variables = periodic_bundle_adjustment(
                    params, variables, keyframe_list, config, intrinsics, cam, first_frame_w2c
                )

            # Report global progress
            if time_idx == 0 or (time_idx + 1) % config["report_global_progress_every"] == 0:
                try:
                    progress_bar = tqdm(range(1), desc=f"Mapping Result: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx,
                                       sil_thres=config["mapping"]["sil_thres"],
                                       mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except Exception:
                    save_params_ckpt(params, os.path.join(config["workdir"], config["run_name"]), time_idx)
                    print("Failed to evaluate trajectory.")

        # ----------------------------------------------------
        # KEYFRAME MANAGEMENT
        # ----------------------------------------------------
        if ((time_idx == 0)
                or ((time_idx + 1) % config["keyframe_every"] == 0)
                or (time_idx == num_frames - 2)):
            if not torch.isinf(curr_gt_w2c[-1]).any() and not torch.isnan(curr_gt_w2c[-1]).any():
                with torch.no_grad():
                    curr_cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
                    curr_cam_tran = params["cam_trans"][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    curr_keyframe = {"id": time_idx, "est_w2c": curr_w2c, "color": color, "depth": depth}
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)

        # Checkpointing
        if time_idx % config["checkpoint_interval"] == 0 and config["save_checkpoints"]:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"),
                    np.array(keyframe_time_indices))

        torch.cuda.empty_cache()

    # ========================================================
    # POST-PROCESSING
    # ========================================================

    # Runtime statistics
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1

    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count

    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000:.2f} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg:.4f} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000:.2f} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg:.4f} s")

    with open(os.path.join(output_dir, "runtimes.txt"), "w") as f:
        f.write(f"Average Tracking/Iteration Time: {tracking_iter_time_avg*1000:.2f} ms\n")
        f.write(f"Average Tracking/Frame Time: {tracking_frame_time_avg:.4f} s\n")
        f.write(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000:.2f} ms\n")
        f.write(f"Average Mapping/Frame Time: {mapping_frame_time_avg:.4f} s\n")
        f.write(f"Total Frame Time: {tracking_frame_time_avg + mapping_frame_time_avg:.4f} s\n")

    # Final evaluation
    eval_ds = [dataset, eval_dataset, "C3VD"] if dataset_config["train_or_test"] == "train" else dataset
    with torch.no_grad():
        eval_save(eval_ds, params, eval_dir, sil_thres=config["mapping"]["sil_thres"],
                  mapping_iters=config["mapping"]["num_iters"],
                  add_new_gaussians=config["mapping"]["add_new_gaussians"])

    # Save final parameters
    params["timestep"] = variables["timestep"]
    params["intrinsics"] = intrinsics.detach().cpu().numpy()
    params["w2c"] = first_frame_w2c.detach().cpu().numpy()
    params["org_width"] = dataset_config["desired_image_width"]
    params["org_height"] = dataset_config["desired_image_height"]
    params["gt_w2c_all_frames"] = np.stack(
        [gt.detach().cpu().numpy() for gt in gt_w2c_all_frames], axis=0
    )
    params["keyframe_time_indices"] = np.array(keyframe_time_indices)

    save_params(params, output_dir)
    save_means3D(params["means3D"], output_dir)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EndoGSLAM with Innovations")
    parser.add_argument("experiment", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    # Load experiment config
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set seed
    seed_everything(seed=experiment.config["seed"])

    # Create results directory
    results_dir = os.path.join(experiment.config["workdir"], experiment.config["run_name"])
    if not experiment.config["load_checkpoint"]:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    # Run SLAM
    rgbd_slam(experiment.config)

    # Create evaluation video
    plot_video(
        os.path.join(results_dir, "eval", "plots"),
        os.path.join("./experiments/", experiment.group_name, experiment.scene_name, "keyframes"),
    )
