"""
Gaussian Splatting operations for SLAM.

Includes:
- Rotation building from quaternions
- SSIM computation
- Optimizer state management (add/remove Gaussians)
- Innovation 1: Visibility-aware dual-mask pruning
- Innovation 1+3: Three-way classifier (static/deforming/floater)
- Gradient-based densification

NOTE: Parts of this code are adapted from the 3D Gaussian Splatting codebase
(Inria GRAPHDECO, non-MIT licensed). See:
https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# ============================================================
# Quaternion to rotation matrix
# ============================================================

def build_rotation(q):
    """
    Convert quaternion(s) to 3x3 rotation matrix.
    
    Args:
        q: (N, 4) quaternions (w, x, y, z)
    
    Returns:
        rot: (N, 3, 3) rotation matrices
    """
    norm = torch.sqrt(q[:, 0]**2 + q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device="cuda")
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot[:, 0, 0] = 1 - 2*(y*y + z*z)
    rot[:, 0, 1] = 2*(x*y - r*z)
    rot[:, 0, 2] = 2*(x*z + r*y)
    rot[:, 1, 0] = 2*(x*y + r*z)
    rot[:, 1, 1] = 1 - 2*(x*x + z*z)
    rot[:, 1, 2] = 2*(y*z - r*x)
    rot[:, 2, 0] = 2*(x*z - r*y)
    rot[:, 2, 1] = 2*(y*z + r*x)
    rot[:, 2, 2] = 1 - 2*(x*x + y*y)
    return rot


# ============================================================
# Image quality metrics
# ============================================================

def calc_mse(img1, img2):
    """Mean squared error between two images."""
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    """Peak signal-to-noise ratio."""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    """1D Gaussian kernel."""
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    """Compute SSIM between two images."""
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Internal SSIM computation."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2*mu1_mu2 + c1) * (2*sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ============================================================
# Optimizer state management
# ============================================================

def accumulate_mean2d_gradient(variables):
    """Accumulate 2D gradient magnitude for densification criterion."""
    variables["means2D_gradient_accum"][variables["seen"]] += torch.norm(
        variables["means2D"].grad[variables["seen"], :2], dim=-1
    )
    variables["denom"][variables["seen"]] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    """Replace parameter values in-place, resetting optimizer state."""
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.zeros_like(v)
            stored_state["exp_avg_sq"] = torch.zeros_like(v)
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
            optimizer.state[group["params"][0]] = stored_state
        else:
            # Adam state not yet initialized (no .step() has run yet);
            # just replace the parameter directly.
            group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        params[k] = group["params"][0]
    return params


def _reset_optimizer_state_subset(optimizer, param_name, mask):
    """
    Reset Adam optimizer state (exp_avg, exp_avg_sq) for a SUBSET of indices
    within a parameter, identified by a boolean mask.

    This avoids wiping momentum for ALL Gaussians when only a few had their
    opacity degenerated. Preserves optimization stability for unaffected Gaussians.

    Args:
        optimizer: Adam optimizer
        param_name: name of the parameter group (e.g., "logit_opacities")
        mask: (N,) boolean tensor, True = reset this Gaussian's state
    """
    group = [g for g in optimizer.param_groups if g["name"] == param_name]
    if not group:
        return
    group = group[0]
    stored_state = optimizer.state.get(group["params"][0], None)
    if stored_state is not None:
        # Zero out momentum only for affected indices
        stored_state["exp_avg"][mask] = 0.0
        stored_state["exp_avg_sq"][mask] = 0.0


def cat_params_to_optimizer(new_params, params, optimizer):
    """Concatenate new Gaussians to existing parameters and optimizer state."""
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            params[k] = group["params"][0]
    return params


# ============================================================
# Tensor resize helpers (for keeping innovation variables in sync)
# ============================================================

def _resize_1d(tensor, N, dtype=None):
    """Resize a 1D tensor to length N: pad with zeros or truncate."""
    old_N = tensor.shape[0]
    if old_N == N:
        return tensor
    elif old_N < N:
        pad = torch.zeros(N - old_N, device=tensor.device, dtype=dtype or tensor.dtype)
        return torch.cat([tensor, pad])
    else:
        return tensor[:N]


def _resize_to_N(tensor, N):
    """Resize a 2D tensor [old_N, D] to [N, D]: pad with zeros or truncate."""
    old_N = tensor.shape[0]
    if old_N == N:
        return tensor
    elif old_N < N:
        pad_shape = list(tensor.shape)
        pad_shape[0] = N - old_N
        pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)
    else:
        return tensor[:N]


# ============================================================
# Point removal (with innovation variable sync)
# ============================================================

def remove_points(to_remove, params, variables, optimizer):
    """
    Remove Gaussians from params, variables, and optimizer state.
    
    Keeps all innovation-related variables (vis_history, deform_mask, etc.)
    in sync with the current Gaussian count.
    """
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ["cam_unnorm_rots", "cam_trans"]]

    for k in keys:
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]

    # Standard variables
    variables["means2D_gradient_accum"] = variables["means2D_gradient_accum"][to_keep]
    variables["denom"] = variables["denom"][to_keep]
    variables["max_2D_radius"] = variables["max_2D_radius"][to_keep]
    if "timestep" in variables:
        variables["timestep"] = variables["timestep"][to_keep]

    # Innovation 1+3: Keep innovation variables in sync
    N = to_keep.shape[0]
    if "vis_history" in variables:
        variables["vis_history"] = _resize_to_N(variables["vis_history"], N)[to_keep]
    if "vis_frame_count" in variables:
        variables["vis_frame_count"] = _resize_1d(variables["vis_frame_count"], N)[to_keep]
    if "deform_mask" in variables:
        variables["deform_mask"] = _resize_1d(variables["deform_mask"], N, dtype=torch.bool)[to_keep]
    if "prev_deform_offsets" in variables:
        variables["prev_deform_offsets"] = _resize_to_N(variables["prev_deform_offsets"], N)[to_keep]

    return params, variables


def inverse_sigmoid(x):
    """Inverse of sigmoid function (logit)."""
    return torch.log(x / (1 - x))


# ============================================================
# TVS-GUIDED SOFT PRUNING: Core computation
# ============================================================

def compute_tvs(variables, params, innovation_config):
    """
    Compute the Temporal-Volumetric Significance (TVS) score for all Gaussians.

    TVS_j = mean(V_buffer[j, 0:W]) * gamma(Sigma_j)
    gamma(Sigma_j) = (V_norm_j)^beta
    V(Sigma_j) = (4/3) * pi * s_x * s_y * s_z  (anisotropic volume)

    Args:
        variables: dict with 'vis_history' (N, W) circular buffer
        params: Gaussian parameters (needs 'log_scales')
        innovation_config: dict with 'tvs_beta'

    Returns:
        tvs: (N,) Temporal-Volumetric Significance score
    """
    beta = innovation_config.get("tvs_beta", 0.1)

    # Temporal mean visibility from circular buffer
    vis_hist = variables["vis_history"]  # (N, W)
    vis_mean = vis_hist.mean(dim=1)  # (N,)

    # Volumetric penalty: penalize large Gaussians that contribute little
    log_scales = params["log_scales"]
    if log_scales.shape[1] == 1:
        # Isotropic: volume = (4/3) * pi * s^3
        s = torch.exp(log_scales.squeeze())
        volume = (4.0 / 3.0) * 3.14159 * s ** 3
    else:
        # Anisotropic: volume = (4/3) * pi * s_x * s_y * s_z
        scales = torch.exp(log_scales)  # (N, 3)
        volume = (4.0 / 3.0) * 3.14159 * scales[:, 0] * scales[:, 1] * scales[:, 2]

    # Normalize volume and apply power penalty
    # Large volume -> high v_norm -> LOW gamma -> LOW TVS -> more likely pruned
    # Small volume -> low v_norm -> HIGH gamma -> HIGH TVS -> protected
    v_max = volume.max().clamp(min=1e-10)
    v_norm = volume / v_max
    gamma = (1.0 - v_norm).clamp(min=0.01) ** beta  # (N,) in (0, 1]

    tvs = vis_mean * gamma
    return tvs


def gumbel_sigmoid_decay(tvs, tau_sig, temperature):
    """
    Compute Gumbel-Sigmoid soft degeneration factor.

    gs(TVS) = 1 / (1 + exp(-(log(TVS) - tau_sig) / temperature))

    When TVS is high -> decay_factor -> 1 (keep)
    When TVS is low  -> decay_factor -> 0 (degenerate)

    Args:
        tvs: (M,) TVS scores for floater Gaussians
        tau_sig: significance threshold (log-space)
        temperature: sharpness of transition

    Returns:
        decay_factor: (M,) in (0, 1)
    """
    # Clamp TVS to avoid log(0)
    safe_tvs = tvs.clamp(min=1e-10)
    logit = (torch.log(safe_tvs) - tau_sig) / temperature
    decay_factor = torch.sigmoid(logit)
    return decay_factor


# ============================================================
# TVS-GUIDED SOFT PRUNING: Spatial floater detection
# ============================================================

def compute_distance_mask(transformed_pts, curr_data, gamma=0.05):
    """
    Detect Gaussians floating in front of the observed surface.

    Projects each Gaussian onto the image plane and compares its depth to the
    observed depth at that pixel. Gaussians significantly in front of the
    surface are flagged as spatial floaters.

    Args:
        transformed_pts: (N, 3) Gaussian positions in camera frame
        curr_data: dict with 'depth' (1, H, W) and 'intrinsics' (3, 3)
        gamma: depth difference threshold

    Returns:
        floater_mask: (N,) boolean tensor, True = spatial floater
    """
    pts_cam = transformed_pts.detach()
    gauss_depth = pts_cam[:, 2]

    intrinsics = curr_data["intrinsics"]
    FX = intrinsics[0, 0]
    FY = intrinsics[1, 1]
    CX = intrinsics[0, 2]
    CY = intrinsics[1, 2]

    H = curr_data["depth"].shape[1]
    W = curr_data["depth"].shape[2]

    # Filter behind-camera Gaussians before projection
    valid_proj = gauss_depth > 0.01
    safe_z = torch.where(valid_proj, gauss_depth, torch.ones_like(gauss_depth))
    u = (FX * pts_cam[:, 0] / safe_z + CX).long().clamp(0, W - 1)
    v = (FY * pts_cam[:, 1] / safe_z + CY).long().clamp(0, H - 1)

    gt_depth = curr_data["depth"].squeeze()  # (H, W)
    observed_depth = gt_depth[v, u]

    # Gaussian is in front of surface by more than gamma
    depth_diff = observed_depth - gauss_depth
    floater_mask = depth_diff > gamma

    # Only flag valid measurements
    valid_depth = observed_depth > 0
    floater_mask = floater_mask & valid_depth & valid_proj

    return floater_mask


# ============================================================
# TVS-GUIDED SOFT PRUNING: Visibility buffer update
# ============================================================

def update_vis_buffer(variables, gauss_vis, innovation_config):
    """
    Update the per-Gaussian visibility circular buffer.

    Called every mapping iteration to record the latest gauss_vis from
    the CUDA rasterizer into the rolling buffer.

    Args:
        variables: dict (modified in-place)
        gauss_vis: (N,) per-Gaussian visibility from rasterizer
        innovation_config: dict with 'tvs_buffer_size'

    Returns:
        variables: updated with new visibility data
    """
    N = gauss_vis.shape[0]
    W = innovation_config.get("tvs_buffer_size", 15)

    # Initialize circular buffer if needed
    if "vis_history" not in variables:
        variables["vis_history"] = torch.zeros(N, W, device="cuda")
        variables["vis_frame_count"] = torch.zeros(N, device="cuda")
        variables["vis_write_idx"] = 0

    # Resize to match current Gaussian count (after densification)
    variables["vis_history"] = _resize_to_N(variables["vis_history"], N)
    variables["vis_frame_count"] = _resize_1d(variables["vis_frame_count"], N)

    # Write current visibility into circular buffer
    col = variables["vis_write_idx"] % W
    variables["vis_history"][:N, col] = gauss_vis[:N].detach()
    variables["vis_write_idx"] = (variables["vis_write_idx"] + 1) % W

    # Count only Gaussians that were actually rendered (gauss_vis > 0)
    variables["vis_frame_count"][:N] += (gauss_vis[:N] > 0).float()

    return variables


# ============================================================
# TVS-GUIDED SOFT PRUNING: Main pruning function
# ============================================================

def prune_gaussians(params, variables, optimizer, iter, prune_dict,
                    innovation_config=None, curr_data=None, transformed_pts=None):
    """
    Gaussian pruning with TVS-Guided Soft Pruning (when enabled) or
    standard opacity-only pruning (baseline fallback).

    TVS-Guided Soft Pruning:
    - Temporal floaters (low TVS + mature) -> Gumbel-Sigmoid degeneration
    - Spatial floaters (in front of surface) -> mild fixed decay
    - Hard removal: sigmoid(opacity) < threshold

    Ablation flags:
    - enable_tvs_pruning: master switch for TVS pruning
    - enable_spatial_mask: toggle spatial floater detection independently

    Args:
        params: Gaussian parameters
        variables: state variables (vis_history, vis_frame_count, etc.)
        optimizer: Adam optimizer
        iter: current mapping iteration
        prune_dict: pruning hyperparameters (start_after, stop_after, etc.)
        innovation_config: innovation settings (None = baseline behavior)
        curr_data: current frame data (for distance mask)
        transformed_pts: (N, 3) Gaussians in camera frame (for distance mask)
    """
    if iter <= prune_dict["stop_after"]:
        if (iter >= prune_dict["start_after"]) and (iter % prune_dict["prune_every"] == 0):
            if iter == prune_dict["stop_after"]:
                remove_threshold = prune_dict["final_removal_opacity_threshold"]
            else:
                remove_threshold = prune_dict["removal_opacity_threshold"]

            # === TVS-GUIDED SOFT PRUNING ===
            if (innovation_config is not None
                    and innovation_config.get("enable_tvs_pruning", False)):

                tau_sig = innovation_config.get("tvs_tau_sig", 0.05)
                temperature = innovation_config.get("tvs_temperature", 0.02)
                min_obs = innovation_config.get("tvs_min_obs", 50)
                eta_spatial = innovation_config.get("eta_spatial", 0.9)
                enable_spatial = innovation_config.get("enable_spatial_mask", True)

                N = params["means3D"].shape[0]

                # --- TVS degeneration is applied BETWEEN frames, not during mapping ---
                # (see tvs_degenerate_between_frames() called from main.py after mapping)
                # This avoids corrupting optimizer state within the mapping session.
                pass

            # === LEGACY: Old dual-mask pruning (for backward compat) ===
            elif (innovation_config is not None
                    and innovation_config.get("enable_visibility_pruning", False)
                    and curr_data is not None
                    and transformed_pts is not None):

                gamma = innovation_config.get("distance_gamma", 0.5)
                eta = innovation_config.get("degeneration_eta", 0.9)
                floater_mask = compute_distance_mask(transformed_pts, curr_data, gamma=gamma)

                if "vis_history" in variables and "vis_frame_count" in variables:
                    vis_threshold = innovation_config.get("vis_threshold", 0.05)
                    min_obs = innovation_config.get("min_observations", 50)
                    N = floater_mask.shape[0]
                    vis_hist = _resize_to_N(variables["vis_history"], N)
                    vis_cnt = _resize_1d(variables["vis_frame_count"], N)
                    vis_mean = vis_hist.mean(dim=1)
                    low_vis = vis_mean < vis_threshold
                    enough_obs = vis_cnt > min_obs
                    vis_floater_mask = low_vis & enough_obs
                    combined_floater = floater_mask | vis_floater_mask
                else:
                    combined_floater = floater_mask

                with torch.no_grad():
                    if combined_floater.any():
                        opacities = torch.sigmoid(params["logit_opacities"].squeeze())
                        opacities[combined_floater] = opacities[combined_floater] * eta
                        opacities = opacities.clamp(1e-6, 1 - 1e-6)
                        new_logit = torch.log(opacities / (1 - opacities)).unsqueeze(-1)
                        new_params = {"logit_opacities": new_logit}
                        params = update_params_and_optimizer(new_params, params, optimizer)

            # Standard opacity-based hard removal (always runs)
            to_remove = (torch.sigmoid(params["logit_opacities"]) < remove_threshold).squeeze()
            # Scale-based removal
            if iter >= prune_dict["remove_big_after"]:
                big_points_ws = torch.exp(params["log_scales"]).max(dim=1).values > 0.1 * variables["scene_radius"]
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()

        # Periodic opacity reset
        if iter > 0 and iter % prune_dict["reset_opacities_every"] == 0 and prune_dict["reset_opacities"]:
            new_params = {"logit_opacities": inverse_sigmoid(torch.ones_like(params["logit_opacities"]) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


# ============================================================
# TVS-GUIDED SOFT PRUNING: Between-frame degeneration
# ============================================================

def tvs_degenerate_between_frames(params, variables, innovation_config, curr_data=None, transformed_pts=None):
    """
    Apply TVS-guided opacity degeneration BETWEEN frames (after mapping completes).

    This is called AFTER the mapping optimizer is discarded, so modifying
    logit_opacities has no interaction with Adam state. The next frame will
    create a fresh optimizer that sees the degenerated values as its starting point.

    Args:
        params: Gaussian parameters (modified in-place)
        variables: state variables with vis_history, vis_frame_count
        innovation_config: dict with TVS hyperparameters
        curr_data: current frame data (for spatial mask, optional)
        transformed_pts: (N, 3) Gaussians in camera frame (for spatial mask, optional)

    Returns:
        params: with degenerated opacities
        n_degenerated: number of Gaussians that had opacity reduced
    """
    tau_sig = innovation_config.get("tvs_tau_sig", 0.05)
    temperature = innovation_config.get("tvs_temperature", 0.02)
    min_obs = innovation_config.get("tvs_min_obs", 50)
    eta_spatial = innovation_config.get("eta_spatial", 0.9)
    enable_spatial = innovation_config.get("enable_spatial_mask", True)

    # Debug: print first 5 calls to verify function behavior
    if not hasattr(tvs_degenerate_between_frames, "_call_count"):
        tvs_degenerate_between_frames._call_count = 0
    tvs_degenerate_between_frames._call_count += 1
    if tvs_degenerate_between_frames._call_count <= 5:
        has_hist = "vis_history" in variables
        print(f"  [TVS-DEBUG] call #{tvs_degenerate_between_frames._call_count}: "
              f"N={params['means3D'].shape[0]}, has_vis_hist={has_hist}, "
              f"min_obs={min_obs}, enable_spatial={enable_spatial}")

    N = params["means3D"].shape[0]
    n_degenerated = 0

    # --- Temporal floaters: low TVS + mature ---
    if "vis_history" in variables and "vis_frame_count" in variables:
        tvs = compute_tvs(variables, params, innovation_config)
        frame_count = _resize_1d(variables["vis_frame_count"], N)
        temporal_floater = (tvs < tau_sig) & (frame_count > min_obs)

        if temporal_floater.any():
            n_floaters = temporal_floater.sum().item()
            # Only degenerate, do NOT let them reach hard-removal threshold.
            # Use a gentle multiplicative decay that floors at 0.01 (sigmoid → ~0.01)
            # instead of driving to zero.
            with torch.no_grad():
                affected_logit = params["logit_opacities"].data[temporal_floater, 0]
                affected_opacity = torch.sigmoid(affected_logit)
                # Gentle decay: multiply by 0.9 (not the Gumbel-Sigmoid which goes to ~0)
                new_opacity = (affected_opacity * 0.9).clamp(0.01, 1 - 1e-6)
                new_logit_vals = torch.log(new_opacity / (1 - new_opacity))
                params["logit_opacities"].data[temporal_floater, 0] = new_logit_vals
                n_degenerated += n_floaters

    # --- Spatial floaters: mild fixed decay ---
    if (enable_spatial and curr_data is not None and transformed_pts is not None):
        gamma = innovation_config.get("distance_gamma", 0.5)
        spatial_floater = compute_distance_mask(transformed_pts, curr_data, gamma=gamma)
        if spatial_floater.any():
            with torch.no_grad():
                affected_logit = params["logit_opacities"].data[spatial_floater, 0]
                affected_opacity = torch.sigmoid(affected_logit)
                new_opacity = (affected_opacity * eta_spatial).clamp(1e-6, 1 - 1e-6)
                new_logit_vals = torch.log(new_opacity / (1 - new_opacity))
                params["logit_opacities"].data[spatial_floater, 0] = new_logit_vals
                n_degenerated += spatial_floater.sum().item()

    # --- Hard removal of dead Gaussians (opacity below threshold) ---
    # This runs without an optimizer, so we just filter params directly
    removal_threshold = 0.005
    to_remove = (torch.sigmoid(params["logit_opacities"]) < removal_threshold).squeeze()
    if to_remove.any():
        # Can't use remove_points here (needs optimizer). Just mark for removal
        # on next frame's mapping prune step. The standard prune at iter=0 will
        # catch these on the next frame.
        pass

    return params, n_degenerated


# ============================================================
# LEGACY: Three-way visibility classifier (kept for backward compat)
# ============================================================

def update_three_way_classifier(variables, gauss_vis, innovation_config):
    """
    Legacy function: Update per-Gaussian visibility history buffer.
    Replaced by update_vis_buffer() in the new TVS system.
    Kept for backward compatibility with old configs using enable_visibility_pruning.
    """
    N = gauss_vis.shape[0]
    W = innovation_config.get("vis_window_size", 15)

    if "vis_history" not in variables:
        variables["vis_history"] = torch.zeros(N, W, device="cuda")
        variables["vis_frame_count"] = torch.zeros(N, device="cuda")
        variables["vis_write_idx"] = 0

    variables["vis_history"] = _resize_to_N(variables["vis_history"], N)
    variables["vis_frame_count"] = _resize_1d(variables["vis_frame_count"], N)

    col = variables["vis_write_idx"] % W
    variables["vis_history"][:N, col] = gauss_vis[:N].detach()
    variables["vis_write_idx"] = (variables["vis_write_idx"] + 1) % W
    variables["vis_frame_count"][:N] += (gauss_vis[:N] > 0).float()

    if innovation_config.get("enable_deformation", False):
        vis_threshold = innovation_config.get("vis_threshold", 0.05)
        var_threshold = innovation_config.get("var_threshold", 0.1)
        min_obs = innovation_config.get("min_observations", 50)

        vis_mean = variables["vis_history"][:N].mean(dim=1)
        vis_var = variables["vis_history"][:N].var(dim=1)
        enough_obs = variables["vis_frame_count"][:N] > min_obs
        deform_mask = (vis_var > var_threshold) & (vis_mean > vis_threshold) & enough_obs
        variables["deform_mask"] = deform_mask

    return variables


# ============================================================
# Gradient-based densification (standard 3DGS)
# ============================================================

def _extend_innovation_vars_for_clone(variables, parent_mask):
    """
    Helper for FIX C: when densify clones Gaussians, copy the parent's
    visibility history into the new rows so newly-created Gaussians
    do not get zero-padded (which would cause them to be flagged as
    floaters within ~min_observations frames). Also extend deform_mask
    and prev_deform_offsets.
    """
    if "vis_history" in variables:
        new_rows = variables["vis_history"][parent_mask]
        variables["vis_history"] = torch.cat([variables["vis_history"], new_rows], dim=0)
    if "vis_frame_count" in variables:
        new_cnt = variables["vis_frame_count"][parent_mask]
        variables["vis_frame_count"] = torch.cat([variables["vis_frame_count"], new_cnt], dim=0)
    if "deform_mask" in variables:
        new_mask = variables["deform_mask"][parent_mask]
        variables["deform_mask"] = torch.cat([variables["deform_mask"], new_mask], dim=0)
    if "prev_deform_offsets" in variables:
        new_off = variables["prev_deform_offsets"][parent_mask]
        variables["prev_deform_offsets"] = torch.cat([variables["prev_deform_offsets"], new_off], dim=0)
    return variables


def densify(params, variables, optimizer, iter, densify_dict):
    """
    Standard gradient-based densification: clone small Gaussians, split large ones.

    Based on the original 3D Gaussian Splatting paper (Kerbl et al., 2023).
    """
    if iter <= densify_dict["stop_after"]:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict["grad_thresh"]

        if (iter >= densify_dict["start_after"]) and (iter % densify_dict["densify_every"] == 0):
            grads = variables["means2D_gradient_accum"] / variables["denom"]
            grads[grads.isnan()] = 0.0

            # Clone: small Gaussians with large gradients
            to_clone = torch.logical_and(
                grads >= grad_thresh,
                torch.max(torch.exp(params["log_scales"]), dim=1).values <= 0.01 * variables["scene_radius"],
            )
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ["cam_unnorm_rots", "cam_trans"]}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            # FIX C: extend visibility/innovation buffers for cloned Gaussians
            variables = _extend_innovation_vars_for_clone(variables, to_clone)
            num_pts = params["means3D"].shape[0]

            # Split: large Gaussians with large gradients
            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[: grads.shape[0]] = grads
            to_split = torch.logical_and(
                padded_grad >= grad_thresh,
                torch.max(torch.exp(params["log_scales"]), dim=1).values > 0.01 * variables["scene_radius"],
            )
            n = densify_dict["num_to_split_into"]
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ["cam_unnorm_rots", "cam_trans"]}
            stds = torch.exp(params["log_scales"])[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params["unnorm_rotations"][to_split]).repeat(n, 1, 1)
            new_params["means3D"] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params["log_scales"] = torch.log(torch.exp(new_params["log_scales"]) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            # FIX C: extend visibility/innovation buffers for split Gaussians.
            # Each split source produces n copies; replicate parent stats n times.
            for _ in range(n):
                variables = _extend_innovation_vars_for_clone(variables, to_split)
            num_pts = params["means3D"].shape[0]

            # Reset accumulators
            variables["means2D_gradient_accum"] = torch.zeros(num_pts, device="cuda")
            variables["denom"] = torch.zeros(num_pts, device="cuda")
            variables["max_2D_radius"] = torch.zeros(num_pts, device="cuda")

            # Remove original split points
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            # Opacity-based cleanup
            if iter == densify_dict["stop_after"]:
                remove_threshold = densify_dict["final_removal_opacity_threshold"]
            else:
                remove_threshold = densify_dict["removal_opacity_threshold"]
            to_remove = (torch.sigmoid(params["logit_opacities"]) < remove_threshold).squeeze()
            if iter >= densify_dict["remove_big_after"]:
                big_points_ws = torch.exp(params["log_scales"]).max(dim=1).values > 0.1 * variables["scene_radius"]
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()

        # Periodic opacity reset
        if iter > 0 and iter % densify_dict["reset_opacities_every"] == 0 and densify_dict["reset_opacities"]:
            new_params = {"logit_opacities": inverse_sigmoid(torch.ones_like(params["logit_opacities"]) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


# ============================================================
# Learning rate utilities
# ============================================================

def update_learning_rate(optimizer, means3D_scheduler, iteration):
    """Update learning rate for means3D using scheduler."""
    for param_group in optimizer.param_groups:
        if param_group["name"] == "means3D":
            lr = means3D_scheduler(iteration)
            param_group["lr"] = lr
            return lr


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """Create exponential learning rate decay function."""
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper
