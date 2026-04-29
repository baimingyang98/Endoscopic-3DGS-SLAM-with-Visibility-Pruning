"""
Keyframe selection strategies for Gaussian SLAM mapping.

Two strategies:
1. Overlap-based: select keyframes with high visual overlap to current frame
2. Distance-based: CDF sampling proportional to spatial/temporal distance
"""
import numpy as np
import torch


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    """Back-project sampled pixels to 3D world-frame point cloud."""
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    xx = (sampled_indices[:, 1] - CX) / FX
    yy = (sampled_indices[:, 0] - CY) / FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True
    )
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[: len(A)]
    pts = pts[~invalid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    """
    Select keyframes with highest visual overlap to the current frame.
    
    Args:
        gt_depth: (1, H, W) ground truth depth
        w2c: (4, 4) current frame world-to-camera
        intrinsics: (3, 3) camera intrinsics
        keyframe_list: list of keyframe dicts with 'est_w2c'
        k: number of keyframes to select
        pixels: number of pixels to sample
    
    Returns:
        selected_keyframe_list: list of keyframe indices
    """
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where((gt_depth[0] > 0) & (gt_depth[0] < 1e10))
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices]

    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        est_w2c = keyframe["est_w2c"]
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]

        edge = 20
        mask = (
            (projected_pts[:, 0] < width - edge)
            * (projected_pts[:, 0] > edge)
            * (projected_pts[:, 1] < height - edge)
            * (projected_pts[:, 1] > edge)
        )
        mask = mask & (points_z[:, 0] > 0)
        percent_inside = mask.sum() / projected_pts.shape[0]
        list_keyframe.append({"id": keyframeid, "percent_inside": percent_inside})

    list_keyframe = sorted(list_keyframe, key=lambda i: i["percent_inside"], reverse=True)
    selected_keyframe_list = [
        kf["id"] for kf in list_keyframe if kf["percent_inside"] > 0.0
    ]
    selected_keyframe_list = list(
        np.random.permutation(np.array(selected_keyframe_list))[:k]
    )

    return selected_keyframe_list


def keyframe_selection_distance(time_idx, curr_position, keyframe_list,
                                distance_current_frame_prob, n_samples):
    """
    CDF-based keyframe sampling proportional to spatial + temporal distance.
    
    Args:
        time_idx: current frame index
        curr_position: (3,) camera position
        keyframe_list: list of keyframe dicts
        distance_current_frame_prob: probability of sampling current frame
        n_samples: number of samples to draw
    
    Returns:
        sample_indices: list of keyframe indices (len(keyframe_list) = current frame)
    """
    distances = []
    time_laps = []
    curr_shift = np.linalg.norm(curr_position)

    for keyframe in keyframe_list:
        est_w2c = keyframe["est_w2c"].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position)
        time_lap = time_idx - keyframe["id"]
        distances.append(distance)
        time_laps.append(time_lap)

    dis2prob = lambda x, scaler: np.log2(1 + scaler / (x + scaler / 5))
    dis_prob = [
        dis2prob(d, curr_shift) + dis2prob(t, time_idx)
        for d, t in zip(distances, time_laps)
    ]
    sum_prob = sum(dis_prob) / (1 - distance_current_frame_prob)
    norm_dis_prob = [p / sum_prob for p in dis_prob]
    norm_dis_prob.append(distance_current_frame_prob)

    cdf = np.cumsum(norm_dis_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    return sample_indices
