"""CPU test for the TVS final cleanup (tvs_final_cleanup).

The cleanup runs once, after refinement, on the same params dict main.py is
about to hand to eval_save and save_params. It must (1) remove exactly the
Gaussians at or below the threshold, (2) leave live ones untouched and in
order, (3) keep the camera trajectory intact -- it is indexed by frame, not by
Gaussian -- and (4) keep params and variables the same length, since
runtimes.txt reads its count straight off means3D afterwards.

Run: python docs/test_final_cleanup.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.slam_external import remove_points_no_optimizer

N, T = 8, 12
THRES = 0.011

# Opacities straddling the floor: 0, 2, 5, 7 are dead (<= 0.011), the rest live.
# Values sit clear of the threshold rather than exactly on it: opacities make the
# round trip through a float32 logit, so a value written as exactly 0.011 can come
# back either side of the comparison. Real floor Gaussians are clamped to 0.01,
# well clear of it, so the ambiguous case never arises in a run.
OPACITIES = [0.0100, 0.9000, 0.0050, 0.5000, 0.0120, 0.0105, 0.2000, 0.0001]
EXPECTED_SURVIVORS = [1, 3, 4, 6]      # 4 is 0.0120, just above the threshold


def make_state():
    opac = torch.tensor(OPACITIES).reshape(N, 1)
    logit = torch.log(opac / (1 - opac))
    params = {
        "means3D": torch.nn.Parameter(torch.arange(N * 3, dtype=torch.float).reshape(N, 3)),
        "rgb_colors": torch.nn.Parameter(torch.rand(N, 3)),
        "unnorm_rotations": torch.nn.Parameter(torch.randn(N, 4)),
        "logit_opacities": torch.nn.Parameter(logit),
        "log_scales": torch.nn.Parameter(torch.full((N, 1), -3.0)),
        "cam_unnorm_rots": torch.nn.Parameter(torch.randn(1, 4, T)),
        "cam_trans": torch.nn.Parameter(torch.randn(1, 3, T)),
    }
    variables = {
        "means2D_gradient_accum": torch.zeros(N),
        "denom": torch.zeros(N),
        "max_2D_radius": torch.zeros(N),
        "timestep": torch.arange(N, dtype=torch.float),   # identity tag
        "vis_history": torch.rand(N),
        "vis_frame_count": torch.arange(N, dtype=torch.float),
        "scene_radius": torch.tensor(1.0),
    }
    return params, variables


def cleanup(params, variables, threshold):
    """The block main.py runs, in isolation."""
    with torch.no_grad():
        opacity = torch.sigmoid(params["logit_opacities"]).reshape(-1)
        to_remove = opacity <= threshold
        n_removed = int(to_remove.sum().item())
        if n_removed:
            params, variables = remove_points_no_optimizer(to_remove, params, variables)
    return params, variables, n_removed


params, variables = make_state()
cam_before = params["cam_unnorm_rots"].detach().clone()
means_before = params["means3D"].detach().clone()

params, variables, n_removed = cleanup(params, variables, THRES)

assert n_removed == N - len(EXPECTED_SURVIVORS), \
    f"expected {N - len(EXPECTED_SURVIVORS)} removed, got {n_removed}"
survivors = [int(t) for t in variables["timestep"]]
assert survivors == EXPECTED_SURVIVORS, \
    f"expected survivors {EXPECTED_SURVIVORS}, got {survivors}"

# Survivors keep their values, in their original order.
assert torch.equal(params["means3D"].data, means_before[EXPECTED_SURVIVORS]), \
    "survivor means3D were reordered or altered"

# Nothing at or below the threshold may survive.
alive = torch.sigmoid(params["logit_opacities"]).reshape(-1)
assert (alive > THRES).all(), f"a below-threshold Gaussian survived: {alive}"

# Camera trajectory is indexed by frame and must not be sliced.
assert params["cam_unnorm_rots"].shape == (1, 4, T), \
    f"cam_unnorm_rots was sliced: {params['cam_unnorm_rots'].shape}"
assert params["cam_trans"].shape == (1, 3, T)
assert torch.equal(params["cam_unnorm_rots"].data, cam_before), \
    "camera trajectory values changed"

# params and variables stay the same length -- runtimes.txt counts means3D.
n_now = params["means3D"].shape[0]
for k, v in params.items():
    if k in ("cam_unnorm_rots", "cam_trans"):
        continue
    assert v.shape[0] == n_now, f"params[{k}] out of sync: {v.shape}"
for k in ("means2D_gradient_accum", "denom", "max_2D_radius", "timestep",
          "vis_history", "vis_frame_count"):
    assert variables[k].shape[0] == n_now, f"variables[{k}] out of sync: {variables[k].shape}"

print(f"OK: cleanup removed {n_removed}/{N} at alpha<={THRES}, "
      f"survivors {survivors}, trajectory intact")

# A map with nothing at the floor must come through untouched.
params, variables = make_state()
params["logit_opacities"] = torch.nn.Parameter(
    torch.full((N, 1), 2.0))                       # sigmoid(2) = 0.88
params, variables, n_removed = cleanup(params, variables, THRES)
assert n_removed == 0 and params["means3D"].shape[0] == N, \
    "cleanup removed Gaussians from a map with no floor population"

print("OK: cleanup is inert on a map with no floor-opacity Gaussians "
      "(the baseline case)")
