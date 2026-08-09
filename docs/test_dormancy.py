"""CPU smoke test for the TVS dormancy timeout (tvs_dormancy_frames).

Checks that (1) Gaussians that were degenerated once and have since gone
unrendered for K frames are removed, (2) a Gaussian rendered again before the
timeout expires resets its counter and survives, (3) a Gaussian the method
never flagged is never removed no matter how long it goes unseen, and
(4) params/variables stay in sync.

Run: python docs/test_dormancy.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.slam_external import tvs_degenerate_between_frames, update_vis_buffer

N = 6
FLOOR = 0.01
K = 3

# Gaussian roles:
#   0: healthy, rendered every frame            -> survives
#   1: dead, but never flagged by TVS           -> survives (never degenerated)
#   2: dead, flagged                            -> removed at the timeout
#   3: dead, flagged                            -> removed at the timeout
#   4: flagged, re-rendered every other frame   -> counter keeps resetting
#   5: dead, flagged                            -> removed at the timeout
VISIBLE_ALWAYS = [0]
REAPPEARS_AT = {4: (2, 4)}      # never dark for K consecutive frames

cfg = dict(
    enable_tvs_pruning=True,
    tvs_aggregation="ema",
    tvs_ema_lambda=0.18,
    tvs_opacity_floor=FLOOR,
    tvs_min_obs=50,
    tvs_degenerate_every=1,
    tvs_dormancy_frames=K,
    enable_spatial_mask=False,
)


def make_state():
    opac = torch.full((N, 1), 0.5)
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
        "vis_history": torch.zeros(N),
        "vis_frame_count": torch.zeros(N),
        # everything except index 1 has already been judged insignificant
        "tvs_degenerated": torch.tensor([True, False, True, True, True, True]),
        "scene_radius": torch.tensor(1.0),
    }
    return params, variables


params, variables = make_state()
# timestep doubles as an identity tag: it is sliced on every removal, so the
# survivors' original indices can be read back out at the end.
variables["timestep"] = torch.arange(N, dtype=torch.float)

for frame in range(1, 6):
    alive = [int(t) for t in variables["timestep"]]
    vis = torch.tensor([0.7 if (gid in VISIBLE_ALWAYS
                                or frame in REAPPEARS_AT.get(gid, ()))
                        else 0.0 for gid in alive])
    update_vis_buffer(variables, vis, cfg)
    params, n_degen, n_removed = tvs_degenerate_between_frames(params, variables, cfg)
    print(f"frame {frame}: removed={n_removed} N={params['means3D'].shape[0]} "
          f"alive={[int(t) for t in variables['timestep']]}")

survivors = sorted(int(t) for t in variables["timestep"])
assert survivors == [0, 1, 4], f"expected [0, 1, 4] to survive, got {survivors}"
n_now = params["means3D"].shape[0]
assert n_now == 3, f"expected 3 survivors, got {n_now}"
for k in ("means2D_gradient_accum", "denom", "max_2D_radius", "timestep",
          "vis_history", "vis_frame_count", "unseen_count", "tvs_degenerated"):
    assert variables[k].shape[0] == n_now, f"{k} out of sync: {variables[k].shape}"

# An unflagged Gaussian must survive unbounded dormancy.
params, variables = make_state()
variables["tvs_degenerated"] = torch.zeros(N, dtype=torch.bool)
for _ in range(20):
    update_vis_buffer(variables, torch.zeros(N), cfg)
    params, _, n_removed = tvs_degenerate_between_frames(params, variables, cfg)
    assert n_removed == 0, "unflagged Gaussians must never be removed"

# Disabled by default: tvs_dormancy_frames=0 must remove nothing.
params, variables = make_state()
cfg_off = dict(cfg, tvs_dormancy_frames=0)
for _ in range(20):
    update_vis_buffer(variables, torch.zeros(N), cfg_off)
    params, _, n_removed = tvs_degenerate_between_frames(params, variables, cfg_off)
    assert n_removed == 0
assert params["means3D"].shape[0] == N

print("OK: dormancy timeout removes flagged+unseen Gaussians, spares revived "
      "and unflagged ones, and is inert when disabled")


# ----------------------------------------------------------------------
# tvs_reset_on_spatial: does the spatial branch starve the maturation gate?
# ----------------------------------------------------------------------
def spatial_state():
    """Gaussians 1 metre in front of a surface observed at 2 m -> all floaters."""
    params, variables = make_state()
    variables["vis_frame_count"] = torch.full((N,), 40.0)   # approaching min_obs=50
    pts = torch.zeros(N, 3)
    pts[:, 2] = 1.0                                          # 1 m from camera
    curr = {
        "depth": torch.full((1, 8, 8), 2.0),                 # surface at 2 m
        "intrinsics": torch.tensor([[4.0, 0.0, 4.0],
                                    [0.0, 4.0, 4.0],
                                    [0.0, 0.0, 1.0]]),
    }
    return params, variables, pts, curr


for coupled in (True, False):
    params, variables, pts, curr = spatial_state()
    cfg_sp = dict(cfg, enable_spatial_mask=True, tvs_dormancy_frames=0,
                  tvs_reset_on_spatial=coupled)
    params, n_degen, _ = tvs_degenerate_between_frames(
        params, variables, cfg_sp, curr_data=curr, transformed_pts=pts)
    counts = variables["vis_frame_count"]
    assert n_degen == N, f"all {N} should be spatial floaters, got {n_degen}"
    if coupled:
        assert (counts == 0).all(), f"coupled: counts should be reset, got {counts}"
    else:
        assert (counts == 40).all(), f"decoupled: counts should survive, got {counts}"

print("OK: tvs_reset_on_spatial=True re-probates spatial floaters (original "
      "behaviour); False leaves their observation counts intact")
