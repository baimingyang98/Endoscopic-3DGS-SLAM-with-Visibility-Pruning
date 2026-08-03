"""CPU smoke test for the TVS dormancy timeout (tvs_dormancy_frames).

Checks that (1) Gaussians pinned at the opacity floor are removed after the
configured number of frames, (2) a Gaussian the optimizer lifts off the floor
resets its counter and survives, and (3) params/variables stay in sync.

Run: python docs/test_dormancy.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.slam_external import tvs_degenerate_between_frames

N = 6
FLOOR = 0.01


def make_state():
    opac = torch.full((N, 1), FLOOR)      # all start pinned at the floor
    opac[0] = 0.9                          # one healthy Gaussian
    logit = torch.log(opac / (1 - opac))
    params = {
        "means3D": torch.nn.Parameter(torch.randn(N, 3)),
        "rgb_colors": torch.nn.Parameter(torch.rand(N, 3)),
        "unnorm_rotations": torch.nn.Parameter(torch.randn(N, 4)),
        "logit_opacities": torch.nn.Parameter(logit),
        "log_scales": torch.nn.Parameter(torch.full((N, 1), -3.0)),
        "cam_unnorm_rots": torch.nn.Parameter(torch.randn(1, 4, 10)),
        "cam_trans": torch.nn.Parameter(torch.randn(1, 3, 10)),
    }
    variables = {
        "means2D_gradient_accum": torch.zeros(N),
        "denom": torch.zeros(N),
        "max_2D_radius": torch.zeros(N),
        "timestep": torch.zeros(N),
        "vis_history": torch.zeros(N),          # EMA mode
        "vis_frame_count": torch.zeros(N),      # nothing mature -> no TVS decay
        "scene_radius": torch.tensor(1.0),
    }
    return params, variables


cfg = dict(
    enable_tvs_pruning=True,
    tvs_aggregation="ema",
    tvs_opacity_floor=FLOOR,
    tvs_min_obs=50,
    tvs_degenerate_every=1,
    tvs_dormancy_frames=3,
    enable_spatial_mask=False,
)

params, variables = make_state()
for frame in range(1, 5):
    if frame == 3:
        # the mapping optimizer revives Gaussian index 1 (recoverability)
        with torch.no_grad():
            params["logit_opacities"].data[1, 0] = torch.log(
                torch.tensor(0.4 / 0.6))
    params, n_degen, n_removed = tvs_degenerate_between_frames(
        params, variables, cfg)
    n_now = params["means3D"].shape[0]
    print(f"frame {frame}: removed={n_removed} N={n_now} "
          f"dormancy={variables['dormancy_count'].tolist()}")

n_now = params["means3D"].shape[0]
assert n_now == 2, f"expected 2 survivors (healthy + revived), got {n_now}"
opac = torch.sigmoid(params["logit_opacities"][:, 0])
assert (opac > FLOOR * 1.05).all(), f"survivors should be off the floor: {opac}"
for k in ("means2D_gradient_accum", "denom", "max_2D_radius", "timestep",
          "vis_history", "vis_frame_count", "dormancy_count"):
    assert variables[k].shape[0] == n_now, f"{k} out of sync: {variables[k].shape}"

# disabled by default: tvs_dormancy_frames=0 must remove nothing
params, variables = make_state()
cfg_off = dict(cfg, tvs_dormancy_frames=0)
for _ in range(5):
    params, _, n_removed = tvs_degenerate_between_frames(params, variables, cfg_off)
    assert n_removed == 0
assert params["means3D"].shape[0] == N
assert "dormancy_count" not in variables

print("OK: dormancy timeout removes dead Gaussians, spares revived ones, "
      "and is inert when disabled")
