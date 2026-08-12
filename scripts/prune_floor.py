"""Delete floor-opacity Gaussians from a finished map and re-evaluate.

TVS soft pruning decays a Gaussian's opacity toward `tvs_opacity_floor` (0.01)
rather than deleting it, and that floor sits deliberately above the baseline
removal threshold (0.005) so a faded Gaussian stays recoverable while the
sequence is still running. Once the run is over nothing can recover them, yet
they are still counted in #G, still stored in params.npz, and still fed through
the rasterizer's per-Gaussian preprocess pass every frame.

In the mo15 10-scene run they are 36% of the final map.

This script removes them post-hoc and re-renders the eval split, so their true
cost in quality can be measured directly. No retraining is involved: it loads
params.npz, drops the Gaussians at or below the threshold, and runs the same
eval_save the SLAM script runs at the end of a session. Score the output with
calc_metrics.py exactly as usual.

Usage
-----
    python scripts/prune_floor.py \\
        --experiment_dir ./experiments/C3VD_mo15/full/sigmoid_t2_a \\
        --config configs/c3vd/c3vd_innovations.py \\
        --out_dir ./experiments/C3VD_mo15_pruned/full/sigmoid_t2_a

    # every scene in a group
    python scripts/prune_floor.py --group_dir ./experiments/C3VD_mo15/full \\
        --config configs/c3vd/c3vd_innovations.py \\
        --out_group ./experiments/C3VD_mo15_pruned/full

Then:
    python scripts/calc_metrics.py --all --group_dir ./experiments/C3VD_mo15_pruned/full
"""
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
import torch

from datasets.gradslam_datasets import load_dataset_config
from utils.common_utils import save_params, save_means3D
from utils.eval_helpers import eval_save

sys.path.insert(0, os.path.join(_BASE_DIR, "scripts"))
from main import get_dataset                                    # noqa: E402

# Per-Gaussian arrays that must be sliced together. Everything else in the npz
# (camera trajectory, intrinsics, gt poses, keyframe indices) is passed through
# untouched -- those are indexed by frame, not by Gaussian.
_GAUSSIAN_KEYS = ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities",
                  "log_scales", "feature_rest", "deform_offsets", "timestep"]


def build_datasets(config, device):
    dcfg = dict(config["data"])
    gradslam_cfg = load_dataset_config(dcfg["gradslam_data_cfg"])
    dcfg.setdefault("ignore_bad", False)
    dcfg.setdefault("use_train_split", True)

    def make(split):
        return get_dataset(
            config_dict=gradslam_cfg,
            basedir=dcfg["basedir"],
            sequence=os.path.basename(dcfg["sequence"]),
            start=dcfg["start"], end=dcfg["end"], stride=dcfg["stride"],
            desired_height=dcfg["desired_image_height"],
            desired_width=dcfg["desired_image_width"],
            device=device, relative_pose=True,
            ignore_bad=dcfg["ignore_bad"],
            use_train_split=dcfg["use_train_split"],
            train_or_test=split,
        )

    return [make("train"), make("test"), "C3VD"]


def prune_one(exp_dir, out_dir, config, threshold, device="cuda:0"):
    data = dict(np.load(os.path.join(exp_dir, "params.npz"), allow_pickle=True))
    opacity = 1.0 / (1.0 + np.exp(-data["logit_opacities"].ravel()))
    keep = opacity > threshold
    n_before, n_after = keep.shape[0], int(keep.sum())

    params = {}
    for k, v in data.items():
        if k in _GAUSSIAN_KEYS and getattr(v, "shape", (0,))[:1] == (n_before,):
            v = v[keep]
        if k in _GAUSSIAN_KEYS or k in ("cam_unnorm_rots", "cam_trans"):
            params[k] = torch.from_numpy(np.asarray(v)).float().to(device)
        else:
            params[k] = v

    scene = Path(exp_dir).name
    print(f"[{scene}] {n_before} -> {n_after} Gaussians "
          f"({100.0 * (n_after - n_before) / n_before:+.1f}%, "
          f"dropped {n_before - n_after} at alpha<={threshold})")

    # Point the config at this scene so the dataset matches the map.
    config["data"]["sequence"] = scene
    dataset = build_datasets(config, device)

    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        eval_save(dataset, params, os.path.join(out_dir, "eval"),
                  sil_thres=config["mapping"]["sil_thres"],
                  mapping_iters=config["mapping"]["num_iters"],
                  add_new_gaussians=config["mapping"]["add_new_gaussians"])

    to_save = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
               for k, v in params.items()}
    save_params(to_save, out_dir)
    save_means3D(params["means3D"], out_dir)
    with open(os.path.join(out_dir, "prune_report.txt"), "w") as f:
        f.write(f"threshold {threshold}\nbefore {n_before}\nafter {n_after}\n")
    return n_before, n_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--group_dir", help="Prune every scene subdirectory.")
    ap.add_argument("--out_group")
    ap.add_argument("--config", required=True,
                    help="Config .py whose data/mapping blocks match the run.")
    ap.add_argument("--threshold", type=float, default=0.011,
                    help="Drop Gaussians with opacity <= this. Default 0.011 "
                         "catches the 0.01 floor without touching live ones.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    config = SourceFileLoader("cfg", args.config).load_module().config

    if args.group_dir:
        if not args.out_group:
            ap.error("--group_dir requires --out_group")
        scenes = sorted(p for p in Path(args.group_dir).iterdir()
                        if (p / "params.npz").exists())
        totals = [0, 0]
        for p in scenes:
            b, a = prune_one(str(p), os.path.join(args.out_group, p.name),
                             config, args.threshold, args.device)
            totals[0] += b
            totals[1] += a
        print(f"\nTOTAL {totals[0]} -> {totals[1]} "
              f"({100.0 * (totals[1] - totals[0]) / totals[0]:+.1f}%)")
    elif args.experiment_dir:
        if not args.out_dir:
            ap.error("--experiment_dir requires --out_dir")
        prune_one(args.experiment_dir, args.out_dir, config,
                  args.threshold, args.device)
    else:
        ap.error("pass --experiment_dir or --group_dir")


if __name__ == "__main__":
    main()
