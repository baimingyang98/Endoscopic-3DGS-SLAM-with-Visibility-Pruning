"""
Post-hoc metrics computation for EndoGSLAM evaluation.

Computes PSNR, MS-SSIM, LPIPS (RGB), depth RMSE, and ATE (pose).
Uses every 8th frame as test set for C3VD.
"""
import os
import glob

import cv2
import numpy as np
import torch
import lpips
from PIL import Image
from pytorch_msssim import ms_ssim

lpips_model = lpips.LPIPS(net="alex")


def lsFile(folder_path, ext="png"):
    """List sorted files with given extension."""
    search_pattern = os.path.join(folder_path, f"*.{ext}")
    files = glob.glob(search_pattern)
    return sorted(files)


def read_pose_file(pose_file):
    """
    Read pose file (16 comma-separated values per line -> 4x4 matrix).
    """
    with open(pose_file, "r") as f:
        lines = f.readlines()
    poses = [np.array([float(x) for x in line.split(",")]).reshape(4, 4) for line in lines]
    # Handle transposed format
    if poses[-1][:3, 3].sum() == 0:
        poses = [pose.T for pose in poses]
    return poses


def calculate_psnr(img1, img2):
    """PSNR between two numpy images (0-255 range)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255**2 / mse)


def calculate_ssim(img1, img2):
    """MS-SSIM between two numpy images."""
    if np.max(img1) > 1:
        img1 = img1 / 255.0
        img2 = img2 / 255.0
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    return ms_ssim(img1_tensor, img2_tensor, data_range=1.0).item()


def calculate_lpips(img1, img2):
    """LPIPS (AlexNet) between two numpy images."""
    if np.max(img1) > 1:
        img1 = img1 / 255.0
        img2 = img2 / 255.0
    img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)
    return distance.item()


def calculate_depth_rmse(depth1, depth2):
    """RMSE between two depth maps."""
    mse = np.mean((depth1 - depth2) ** 2)
    return np.sqrt(mse)


def align(model, data):
    """
    Align trajectories using Horn's method.
    
    Args:
        model: (3, N) first trajectory
        data: (3, N) second trajectory
    
    Returns:
        model_aligned: (3, N) aligned trajectory
        trans_error: (N,) per-point error
    """
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return model_aligned, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Compute ATE between two pose lists.
    
    Returns:
        ate: mean translational error
        gt_aligned: aligned GT positions
        est_pts: estimated positions
    """
    gt_pts = np.array([t[:3, 3] for t in gt_traj]).T
    est_pts = np.array([t[:3, 3] for t in est_traj]).T
    gt_aligned, trans_error = align(gt_pts, est_pts)
    return trans_error.mean(), gt_aligned, est_pts


# ============================================================
# High-level metric functions
# ============================================================

def rgb_metrics(gt_dir, render_dir):
    """
    Compute PSNR, SSIM, LPIPS for rendered vs GT color images.
    Uses every 8th frame (C3VD test split).
    
    Args:
        gt_dir: directory containing 'color/' subfolder
        render_dir: directory containing 'color/' subfolder
    
    Returns:
        mean_psnr, mean_ssim, mean_lpips, psnr_list, ssim_list, lpips_list
    """
    color_gt = os.path.join(gt_dir, "color")
    color_render = os.path.join(render_dir, "color")
    color_files1 = lsFile(color_gt)[7::8]
    color_files2 = lsFile(color_render)
    if "0000" in color_files2[0]:
        color_files2 = color_files2[7::8]

    color1 = [cv2.cvtColor(cv2.imread(p).astype(np.float32), cv2.COLOR_BGR2RGB) for p in color_files1]
    color2 = [cv2.cvtColor(cv2.imread(p).astype(np.float32), cv2.COLOR_BGR2RGB) for p in color_files2]

    psnr_list = [calculate_psnr(color1[i], color2[i]) for i in range(len(color1))]
    ssim_list = [calculate_ssim(color1[i], color2[i]) for i in range(len(color1))]
    lpips_list = [calculate_lpips(color1[i], color2[i]) for i in range(len(color1))]

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), psnr_list, ssim_list, lpips_list


def depth_metrics(gt_dir, render_dir):
    """
    Compute depth RMSE for rendered vs GT depth maps.
    
    Args:
        gt_dir: directory containing 'depth/' subfolder (.tiff)
        render_dir: directory containing 'depth/' subfolder (.tiff)
    
    Returns:
        mean_rmse, rmse_list
    """
    depth_gt = os.path.join(gt_dir, "depth")
    depth_render = os.path.join(render_dir, "depth")
    depth_files1 = lsFile(depth_gt, "tiff")[7::8]
    depth_files2 = lsFile(depth_render, "tiff")
    if "0000" in depth_files2[0]:
        depth_files2 = depth_files2[7::8]

    depth1 = [np.array(Image.open(p)).astype(np.uint16) / 2.55 for p in depth_files1]
    depth2 = [np.clip(np.array(Image.open(p)).astype(np.uint16) / 655.35, 0, 100) for p in depth_files2]

    rmse_list = [calculate_depth_rmse(depth1[i], depth2[i]) for i in range(len(depth1))]
    return np.mean(rmse_list), rmse_list


def pose_metrics(gt_w2c_path, est_w2c_path):
    """
    Compute ATE from saved pose files.
    
    Returns:
        ate: average translational error
        gt_pts: (3, N) ground truth positions
        est_pts: (3, N) estimated positions
    """
    gt_w2c = read_pose_file(gt_w2c_path)
    est_w2c = read_pose_file(est_w2c_path)
    ate, gt_pts, est_pts = evaluate_ate(est_w2c, gt_w2c)
    return ate, gt_pts, est_pts
