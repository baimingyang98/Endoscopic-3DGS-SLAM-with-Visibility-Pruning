"""
Evaluation helpers for EndoGSLAM.

- report_progress: live tracking/mapping progress display
- eval_save: final evaluation with rendered images and trajectory saving
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_external import build_rotation, calc_psnr
from utils.slam_helpers import (
    transform_to_frame, transform_to_frame_eval,
    transformed_params2rendervar, transformed_params2depthplussilhouette,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()


# ============================================================
# Trajectory alignment (Horn's method)
# ============================================================

def align(model, data):
    """
    Align two trajectories using Horn's closed-form method.
    
    Args:
        model: (3, N) first trajectory
        data: (3, N) second trajectory
    
    Returns:
        rot: (3, 3) rotation
        trans: (3, 1) translation
        trans_error: (N,) per-point translational error
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

    return rot, trans, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Compute Absolute Trajectory Error (ATE) RMSE.
    
    Args:
        gt_traj: list of (4, 4) tensors
        est_traj: list of (4, 4) tensors
    
    Returns:
        avg_trans_error: mean translational error
    """
    gt_traj_pts = [gt_traj[idx][:3, 3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3, 3] for idx in range(len(est_traj))]

    gt_traj_pts = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)
    return trans_error.mean()


# ============================================================
# Progress reporting during SLAM
# ============================================================

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth,
                         presence_sil_mask, diff_depth_l1, psnr, depth_l1,
                         fig_title, plot_dir=None, plot_name=None, save_plot=False):
    """Save comparison plot of GT vs rendered RGB/depth."""
    if depth.max() > 1.0:
        depth = depth / 100.0 * 2.55
    if rastered_depth.max() > 1.0:
        rastered_depth = rastered_depth / 100.0 * 2.55

    aspect_ratio = color.shape[2] / color.shape[1]
    fig_width = 14 / 1.55 * aspect_ratio
    fig, axs = plt.subplots(2, 3, figsize=(fig_width, 8))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0].cpu(), cmap="jet", vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    axs[1, 0].imshow(torch.clamp(rastered_color, 0, 1).cpu().permute(1, 2, 0))
    axs[1, 0].set_title(f"Rendered RGB, PSNR: {psnr:.2f}")
    axs[1, 1].imshow(rastered_depth[0].cpu(), cmap="jet", vmin=0, vmax=6)
    axs[1, 1].set_title(f"Rendered Depth, L1: {depth_l1:.2f}")
    axs[0, 2].imshow(presence_sil_mask, cmap="gray")
    axs[0, 2].set_title("Silhouette")
    axs[1, 2].imshow(diff_depth_l1.cpu().squeeze(0), cmap="jet", vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")
    for ax in axs.flatten():
        ax.axis("off")
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(plot_dir, f"{plot_name}.png"), bbox_inches="tight")
    plt.close()


def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres,
                    every_i=1, tracking=False, mapping=False, online_time_idx=None):
    """Report PSNR/depth/ATE metrics during tracking or mapping."""
    if i % every_i != 0 and i != 1:
        return

    if tracking:
        gt_w2c_list = data["iter_gt_w2c_list"]
        valid_gt_w2c_list = []
        latest_est_w2c_list = []
        latest_est_w2c_list.append(data["w2c"])
        valid_gt_w2c_list.append(gt_w2c_list[0])

        for idx in range(1, iter_time_idx + 1):
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(params["cam_unnorm_rots"][..., idx].detach())
            interm_cam_trans = params["cam_trans"][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c_list.append(intermrel_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])

        gt_w2c_list = valid_gt_w2c_list
        latest_est_w2c = latest_est_w2c_list[-1]
        iter_gt_w2c = gt_w2c_list[-1]
        iter_pt_error = torch.sqrt(
            (latest_est_w2c[0, 3] - iter_gt_w2c[0, 3])**2 +
            (latest_est_w2c[1, 3] - iter_gt_w2c[1, 3])**2 +
            (latest_est_w2c[2, 3] - iter_gt_w2c[2, 3])**2
        )
        if iter_time_idx > 0:
            rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
            rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
            rel_pt_error = torch.sqrt(
                (rel_gt_w2c[0, 3] - rel_est_w2c[0, 3])**2 +
                (rel_gt_w2c[1, 3] - rel_est_w2c[1, 3])**2 +
                (rel_gt_w2c[2, 3] - rel_est_w2c[2, 3])**2
            )
        else:
            rel_pt_error = torch.zeros(1).float()

        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        ate_rmse = np.round(ate_rmse, decimals=6)

    # Render current frame
    transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False, camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, data["w2c"], transformed_pts)

    depth_sil, _, _, _ = Renderer(raster_settings=data["cam"])(**depth_sil_rendervar)
    rastered_depth = depth_sil[0, :, :].unsqueeze(0)
    valid_depth_mask = (data["depth"] > 0) & (data["depth"] < 1e10)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = silhouette > sil_thres

    im, _, _, _ = Renderer(raster_settings=data["cam"])(**rendervar)

    if tracking:
        psnr = calc_psnr(im * presence_sil_mask, data["im"] * presence_sil_mask).mean()
        diff_depth_rmse = torch.sqrt(((rastered_depth - data["depth"]) * presence_sil_mask) ** 2)
        diff_depth_rmse = diff_depth_rmse * valid_depth_mask
        rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
    else:
        psnr = calc_psnr(im, data["im"]).mean()
        diff_depth_rmse = torch.sqrt((rastered_depth - data["depth"]) ** 2)
        diff_depth_rmse = diff_depth_rmse * valid_depth_mask
        rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()

    if tracking:
        progress_bar.set_postfix({
            f"Step {iter_time_idx} | ATE RMSE": f"{ate_rmse:.6f}"
        })
    elif mapping:
        progress_bar.set_postfix({
            f"Step {online_time_idx} | PSNR": f"{psnr:.4f} | RMSE: {rmse:.4f}"
        })
    progress_bar.update(every_i)


# ============================================================
# Final evaluation and saving
# ============================================================

def eval_save(dataset, final_params, eval_dir, sil_thres,
              mapping_iters, add_new_gaussians, save_renders=True):
    """
    Final evaluation: render all frames, save images, compute trajectory.
    
    For C3VD with train/test split: renders test frames using Horn-aligned GT poses.
    """
    split = False
    if type(dataset) is list:
        assert dataset[2] == "C3VD"
        train_dataset, eval_dataset = dataset[:2]
        split = True
        num_frames = len(train_dataset) + len(eval_dataset)
    else:
        num_frames = len(dataset)

    print("Evaluating Final Parameters ...")
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_renders:
        render_rgb_dir = os.path.join(eval_dir, "color")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(render_depth_dir, exist_ok=True)

    gt_w2c_list = []
    gt_position_list = []
    est_position_list = []

    def get_idx(total_time_idx):
        if split:
            if (total_time_idx + 1) % 8 != 0:
                return "train", train_dataset, total_time_idx - (total_time_idx // 8)
            else:
                return "test", eval_dataset, total_time_idx // 8
        return "train", dataset, total_time_idx

    # First pass: collect poses
    for total_time_idx in range(num_frames):
        dataset_type, ds, time_idx = get_idx(total_time_idx)
        gt_w2c_list.append(torch.linalg.inv(ds.get_pose(time_idx).cpu()))
        gt_position_list.append(gt_w2c_list[-1][:3, 3])
        if total_time_idx == 0:
            est_position_list.append(gt_position_list[0])
        elif dataset_type == "train":
            interm_cam_trans = final_params["cam_trans"][..., time_idx].detach().cpu()
            est_position_list.append(interm_cam_trans.squeeze())

    # Horn alignment for novel view synthesis on test frames
    if split:
        all_idx = set(range(num_frames))
        eval_idx = set(range(7, num_frames, 8))
        train_idx = sorted(list(all_idx - eval_idx))
        rot, trans, _ = align(
            torch.stack(gt_position_list)[train_idx].detach().cpu().numpy().T,
            torch.stack(est_position_list).detach().cpu().numpy().T,
        )
        horn_gt_position = [rot @ gp.numpy() + trans.squeeze() for gp in gt_position_list]

    # Second pass: render and save
    for total_time_idx in tqdm(range(num_frames)):
        dataset_type, ds, time_idx = get_idx(total_time_idx)
        color, depth, intrinsics, pose = ds[time_idx]
        intrinsics = intrinsics[:3, :3]
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)

        if total_time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = setup_camera(
                color.shape[2], color.shape[1],
                intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
            )

        # Get transformed points
        if dataset_type == "train":
            cam_rot = F.normalize(final_params["cam_unnorm_rots"][..., time_idx].detach())
            cam_tran = final_params["cam_trans"][..., time_idx].detach()
            transformed_pts = transform_to_frame_eval(final_params, (cam_rot, cam_tran))
        else:
            w2c = gt_w2c_list[total_time_idx]
            w2c[:3, 3] = torch.Tensor(horn_gt_position[total_time_idx])
            transformed_pts = transform_to_frame_eval(final_params, rel_w2c=w2c.cuda())

        curr_data = {
            "cam": cam, "im": color, "depth": depth,
            "id": time_idx, "intrinsics": intrinsics, "w2c": first_frame_w2c,
        }

        # Skip train frames for rendering (only render test)
        if not (dataset_type == "test" or not split):
            continue

        # Render
        rendervar = transformed_params2rendervar(final_params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(
            final_params, curr_data["w2c"], transformed_pts
        )

        depth_sil, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (curr_data["depth"] > 0) & (curr_data["depth"] < 1e10)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**rendervar)
        weighted_im = im * valid_depth_mask
        weighted_gt_im = curr_data["im"] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()

        diff_depth_l1 = torch.abs(rastered_depth - curr_data["depth"]) * valid_depth_mask
        depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if save_renders:
            viz_render_im = torch.clamp(im, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
            save_im = np.clip(viz_render_im * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(render_rgb_dir, f"color_{total_time_idx:04d}.png"),
                cv2.cvtColor(save_im, cv2.COLOR_RGB2BGR),
            )
            viz_render_depth = rastered_depth[0].detach().cpu().numpy()
            save_depth = np.clip(viz_render_depth * 655.35, 0, 65535)
            Image.fromarray(np.uint16(save_depth)).save(
                os.path.join(render_depth_dir, f"depth_{total_time_idx:04d}.tiff")
            )

        fig_title = f"Time Step: {total_time_idx}"
        plot_name = f"{total_time_idx:04d}"
        plot_rgbd_silhouette(
            color, depth, im, rastered_depth,
            presence_sil_mask.detach().cpu().numpy(), diff_depth_l1,
            psnr, depth_l1, fig_title, plot_dir,
            plot_name=plot_name, save_plot=True,
        )

    # Save trajectory for ATE evaluation
    num_frames_params = final_params["cam_unnorm_rots"].shape[-1]
    latest_est_w2c_list = []
    latest_est_w2c_list.append(first_frame_w2c)
    for idx in range(1, num_frames_params):
        interm_cam_rot = F.normalize(final_params["cam_unnorm_rots"][..., idx].detach())
        interm_cam_trans = final_params["cam_trans"][..., idx].detach()
        intermrel_w2c = torch.eye(4).cuda().float()
        intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
        intermrel_w2c[:3, 3] = interm_cam_trans
        latest_est_w2c_list.append(intermrel_w2c)

    if split:
        train_idx_list = sorted(list(set(range(num_frames)) - set(range(7, num_frames, 8))))
        with open(os.path.join(eval_dir, "gt_train_w2c.txt"), "w") as f:
            for tensor in [gt_w2c_list[idx] for idx in train_idx_list]:
                line = ",".join(str(v) for v in tensor.reshape(-1).numpy())
                f.write(line + "\n")
        with open(os.path.join(eval_dir, "est_w2c.txt"), "w") as f:
            for tensor in latest_est_w2c_list:
                line = ",".join(str(v) for v in tensor.reshape(-1).cpu().numpy())
                f.write(line + "\n")
