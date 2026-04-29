# EndoGSLAM with Innovations

**Robust Gaussian Map Management for Endoscopic SLAM: Visibility-Guided Pruning and Bundle Adjustment**

This repository extends [EndoGSLAM](https://github.com/endogslam/EndoGSLAM) with three innovations for improved map quality and camera tracking in endoscopic RGB-D SLAM:

1. **Visibility-Aware Dual-Mask Pruning** - Combines per-Gaussian visibility scores (from a modified CUDA rasterizer) with depth-based floater detection to degenerate spurious Gaussians.
2. **Periodic Bundle Adjustment** - Joint optimization of camera poses and Gaussian parameters over selected keyframes to reduce pose drift.
3. **Deformation Modeling** (experimental) - Per-Gaussian position offsets for handling tissue deformation, with three-way classification (static/deforming/floater).

## Project Structure

```
configs/         - Experiment configurations and dataset YAML
datasets/        - Dataset loaders (C3VD, EndoSLAM)
scripts/         - Entry points (main SLAM pipeline, metrics calculation)
utils/           - Core utilities (rendering, losses, pruning, evaluation)
submodules/      - Modified CUDA Gaussian rasterizer
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install modified rasterizer (with per-Gaussian visibility output)
pip install submodules/diff-gaussian-rasterization-w-depth/
```

## Usage

```bash
# Run SLAM on a C3VD sequence
python scripts/main.py configs/c3vd/c3vd_innovations.py

# Evaluate metrics
python scripts/calc_metrics.py --experiment_dir ./experiments/C3VD_innovations/sigmoid_t3_a
```

## Dataset

Download the [C3VD dataset](https://durrlab.github.io/C3VD/) and place it under `./data/C3VD/`.

## Citation

Based on EndoGSLAM (Wang et al., 2024) and SplaTAM (Keetha et al., 2024).
