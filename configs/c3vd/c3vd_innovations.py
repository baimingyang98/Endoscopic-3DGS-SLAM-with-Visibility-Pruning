"""
Configuration for C3VD dataset with TVS-Guided Soft Pruning.

Ablation system: each innovation is independently toggleable.
Scene is selected via the SCENE_NUM environment variable (default: sigmoid_t3_a).

Current state (this file): BASELINE-PARITY TEST CONFIG.
  - All innovations OFF (TVS, spatial mask, refinement)
  - gaussian_simplification=True (isotropic + direct RGB; matches EndoGSLAM baseline)
  - 15 tracking iters + 15 mapping iters (short schedule for fast A/B runs)
  - seed=2048, group_name="C3VD_tvs_test"

To enable innovations, flip the corresponding `enable_*` flags below.
Use this file as the control arm of any ablation; copy + edit for treatment arms.
"""
import os

# ============================================================
# Scene selection
# ============================================================
scenes = [
    "cecum_t1_b", "cecum_t2_b", "cecum_t3_a",
    "sigmoid_t1_a", "sigmoid_t2_a", "sigmoid_t3_a",
    "trans_t1_b", "trans_t2_c", "trans_t4_a", "trans_t4_b",
]

primary_device = "cuda:0"
seed = 2048  # Fixed for reproducibility across A/B runs

try:
    scene_name = scenes[int(os.environ["SCENE_NUM"])]
except (KeyError, IndexError):
    scene_name = "sigmoid_t3_a"

# ============================================================
# Pipeline parameters
# ============================================================
map_every = 1          # Map at every frame
keyframe_every = 8     # Keyframe stride; matches C3VD test split (every 8th frame is test)
tracking_iters = 15    # Tracking iterations per frame (short schedule)
mapping_iters = 15     # Mapping iterations per frame (short schedule)

group_name = "C3VD_tvs_test"
run_name = scene_name

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    distance_keyframe_selection=True,
    distance_current_frame_prob=0.1,
    mapping_window_size=-1,
    report_global_progress_every=999999,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=int(1e10),
    # True  = isotropic scales + direct RGB (simplified, EndoGSLAM baseline)
    # False = anisotropic scales + 3-degree SH (full 3DGS)
    gaussian_simplification=True,

    # --------------------------------------------------------
    # Data configuration
    # --------------------------------------------------------
    data=dict(
        basedir="./data/C3VD",
        gradslam_data_cfg="./configs/data/c3vd.yaml",
        sequence=scene_name,
        desired_image_height=1080 // 2,
        desired_image_width=1350 // 2,
        start=0, end=-1, stride=1, num_frames=-1,
        train_or_test="train",
    ),

    # --------------------------------------------------------
    # Tracking parameters
    # --------------------------------------------------------
    tracking=dict(
        use_gt_poses=False, forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True, sil_thres=0.99, use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
            logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.002, cam_trans=0.005,
        ),
    ),

    # --------------------------------------------------------
    # Mapping parameters
    # --------------------------------------------------------
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True, sil_thres=0.5,
        use_l1=True, use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=1.0, depth=1.0),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025,
            unnorm_rotations=0.001, logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0, cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0, remove_big_after=0,
            stop_after=20, prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10),
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500, remove_big_after=3000,
            stop_after=5000, densify_every=100,
            grad_thresh=0.0002, num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),

    # ============================================================
    # INNOVATIONS — all independently toggleable for ablation.
    # Current state: ALL OFF (baseline parity).
    # ============================================================
    innovations=dict(
        # --- TVS-Guided Soft Pruning ---
        enable_tvs_pruning=False,       # Master switch for TVS soft pruning
        tvs_aggregation="uniform",      # "uniform" (circular-buffer mean) or "ema" (exp. moving avg)
        tvs_buffer_size=15,             # W: circular buffer size in frames (uniform mode only)
        tvs_ema_lambda=0.0667,          # EMA rate (ema mode only); 1/15 matches uniform W=15
        tvs_opacity_floor=0.01,         # Min opacity after degeneration (prevents hard removal)
        tvs_beta=0.1,                   # Volume penalty exponent: gamma = (1 - V_norm)^beta
        tvs_tau_sig=0.05,               # Significance midpoint: TVS=tau_sig -> decay=0.5
        tvs_temperature=1.0,            # Transition width in log-space for the Gumbel-sigmoid gate
        tvs_min_obs=50,                 # Maturation gate: min observed frames before TVS-eligible
        eta_spatial=0.9,                # Spatial floater mild decay factor (multiplicative)
        enable_spatial_mask=False,      # Toggle spatial floater detection (depth-based)
        distance_gamma=0.5,             # Depth-diff threshold (meters) for spatial floater mask

        # --- Post-SLAM Refinement ---
        enable_refinement=False,        # Two-stage post-SLAM refinement (worst-first + global)
        refine_stage1_iters=20,         # Iters per keyframe in Stage 1 (worst-first)
        refine_stage2_iters=100,        # Global random iters in Stage 2
        refine_lambda_dssim=0.2,        # DSSIM weight in the refinement L1 + DSSIM loss

        # --- Legacy flags (deprecated; kept for backward compatibility) ---
        enable_visibility_pruning=False,
        enable_periodic_ba=False,
        enable_deformation=False,
    ),

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    viz=dict(
        render_mode="color", offset_first_viz_cam=True,
        show_sil=False, visualize_cams=False,
        viz_w=320, viz_h=320, viz_near=0.01, viz_far=100.0,
        view_scale=2, viz_fps=30,
        enter_interactive_post_online=True,
    ),
)
