"""
Post-hoc metric evaluation script.

Computes PSNR, SSIM, LPIPS, depth RMSE, and ATE for a completed experiment.

Usage:
    python scripts/calc_metrics.py --experiment_dir ./experiments/C3VD_innovations/sigmoid_t3_a
    python scripts/calc_metrics.py --all --group_dir ./experiments/C3VD_innovations
"""
import argparse
import os
import sys
import csv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import numpy as np
from utils.metrics_helper import rgb_metrics, depth_metrics, pose_metrics


def metric_single(gt_dir, experiment_dir):
    """
    Compute all metrics for a single scene.
    
    Args:
        gt_dir: path to ground truth data (with color/ and depth/ subfolders)
        experiment_dir: path to experiment output (with eval/ subfolder)
    
    Returns:
        dict with all metric values
    """
    eval_dir = os.path.join(experiment_dir, "eval")
    
    results = {}
    
    # RGB metrics
    try:
        psnr, ssim, lpips_val, _, _, _ = rgb_metrics(gt_dir, eval_dir)
        results["PSNR"] = psnr
        results["SSIM"] = ssim
        results["LPIPS"] = lpips_val
        print(f"  PSNR:  {psnr:.4f}")
        print(f"  SSIM:  {ssim:.4f}")
        print(f"  LPIPS: {lpips_val:.4f}")
    except Exception as e:
        print(f"  RGB metrics failed: {e}")
    
    # Depth metrics
    try:
        rmse, _ = depth_metrics(gt_dir, eval_dir)
        results["RMSE"] = rmse
        print(f"  RMSE:  {rmse:.4f}")
    except Exception as e:
        print(f"  Depth metrics failed: {e}")
    
    # Pose metrics (ATE)
    try:
        gt_w2c_path = os.path.join(eval_dir, "gt_train_w2c.txt")
        est_w2c_path = os.path.join(eval_dir, "est_w2c.txt")
        if os.path.exists(gt_w2c_path) and os.path.exists(est_w2c_path):
            ate, _, _ = pose_metrics(gt_w2c_path, est_w2c_path)
            results["ATE"] = ate
            print(f"  ATE:   {ate:.6f}")
    except Exception as e:
        print(f"  Pose metrics failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EndoGSLAM metrics")
    parser.add_argument("--experiment_dir", type=str, help="Single experiment directory")
    parser.add_argument("--gt_dir", type=str, default=None, help="Ground truth data directory")
    parser.add_argument("--all", action="store_true", help="Evaluate all scenes in group")
    parser.add_argument("--group_dir", type=str, default=None, help="Group directory for --all mode")
    parser.add_argument("--data_dir", type=str, default="./data/C3VD", help="Base data directory")
    args = parser.parse_args()

    if args.all and args.group_dir:
        # Evaluate all scenes
        scenes = sorted([
            d for d in os.listdir(args.group_dir)
            if os.path.isdir(os.path.join(args.group_dir, d))
        ])
        
        all_results = []
        for scene in scenes:
            print(f"\n=== {scene} ===")
            experiment_dir = os.path.join(args.group_dir, scene)
            gt_dir = os.path.join(args.data_dir, scene)
            results = metric_single(gt_dir, experiment_dir)
            results["scene"] = scene
            all_results.append(results)
        
        # Summary CSV
        csv_path = os.path.join(args.group_dir, "metrics_summary.csv")
        if all_results:
            keys = ["scene"] + [k for k in all_results[0].keys() if k != "scene"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_results)
            
            # Print averages
            print("\n=== AVERAGES ===")
            for key in keys:
                if key == "scene":
                    continue
                vals = [r[key] for r in all_results if key in r]
                if vals:
                    print(f"  {key}: {np.mean(vals):.4f}")
            print(f"\nSaved to {csv_path}")
    
    elif args.experiment_dir:
        gt_dir = args.gt_dir or os.path.join(
            args.data_dir, os.path.basename(args.experiment_dir)
        )
        print(f"Evaluating: {args.experiment_dir}")
        print(f"GT dir:     {gt_dir}")
        metric_single(gt_dir, args.experiment_dir)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
