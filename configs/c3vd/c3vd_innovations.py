"""
Configuration for C3VD dataset with TVS-Guided Soft Pruning + Optical Flow.

Ablation system: each innovation is independently toggleable.
Scene is selected via SCENE_NUM environment variable (default: sigmoid_t3_a).

Ablation configs:
  - Baseline parity:   all innovations OFF, gaussian_simplification=True, 15+25 iters
  - A1 (TVS only):     enable_tvs_pruning=True, all others OFF
  - A2 (TVS+spatial):  A1 + enable_spatial_mask=True
  - A3 (Flow init):    enable_flow_init=True, all others OFF
  - A4 (Flow loss):    enable_flow_loss=True, all others OFF
  - A5 (Full 3DGS):    gaussian_simplification=False, all innovations OFF
  - A6 (Refinement):   enable_refinement=True, all others OFF
  - Full system:       everything ON, gaussian_simplification=False, 30+50 iters
"""
import os

# ============================================================
# Scene selection
# ============================================================
scenes = [
    "cecum_t1_b",
    "cecum_t2_b",
    "cecum_t3_a",
    "sigmoid_t1_a",
    "sigmoid_t2_a",
    "sigmoid_t3_a",
    "trans_t1_b",
    "trans_t2_c",
    "trans_t4_a",
    "trans_t4_b",
]

primary_device = "cuda:0"
seed = 0

try:
    scene_name = scenes[int(os.environ["SCENE_NUM"])]
except (KeyError, IndexError):
    scene_name = "sigmoid_t3_a"

# ============================================================
# Pipeline parameters
# ============================================================
map_every = 1
keyframe_every = 8
tracking_iters = 30
mapping_iters = 50

group_name = "C3VD_tvs"
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
    gaussian_simplification=False,  # Full 3DGS: anisotropic scales + 3-degree SH

    # --------------------------------------------------------
    # Data configuration
    # --------------------------------------------------------
    data=dict(
        basedir="./data/C3VD",
        gradslam_data_cfg="./configs/data/c3vd.yaml",
        sequence=scene_name,
        desired_image_height=1080 // 2,
        desired_image_width=1350 // 2,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        train_or_test="train",
    ),

    # --------------------------------------------------------
    # Tracking parameters
    # --------------------------------------------------------
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.005,
        ),
    ),

    # --------------------------------------------------------
    # Mapping parameters
    # --------------------------------------------------------
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=1.0, depth=1.0),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=int(1e10),
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),

    # ============================================================
    # INNOVATIONS — All independently toggleable for ablation
    # ============================================================
    innovations=dict(
        # --- TVS-Guided Soft Pruning ---
        enable_tvs_pruning=True,        # Master switch for TVS pruning
        tvs_buffer_size=15,             # W: circular buffer size (frames)
        tvs_beta=0.1,                   # Volume penalty exponent
        tvs_tau_sig=0.05,               # Significance threshold (log-space)
        tvs_temperature=0.02,           # Gumbel-Sigmoid sharpness
        tvs_min_obs=50,                 # Maturation: min frames before eligible
        eta_spatial=0.9,                # Spatial floater mild decay factor
        enable_spatial_mask=True,       # Toggle spatial floater detection
        distance_gamma=0.5,             # Depth diff threshold for spatial mask

        # --- Optical Flow ---
        enable_flow_init=True,          # Flow-guided pose initialization
        enable_flow_loss=True,          # L_flow in mapping
        lambda_flow=0.1,                # Weight for flow loss
        flow_dir="flow",                # Subdirectory under scene folder
        flow_confidence_threshold=0.5,  # Min confidence to use flow init

        # --- Post-SLAM Refinement ---
        enable_refinement=True,         # Two-stage post-SLAM refinement
        refine_stage1_iters=100,        # Iters per keyframe (worst-first)
        refine_stage2_iters=500,        # Global random iters
        refine_lambda_dssim=0.2,        # DSSIM weight in refinement loss

        # --- Legacy flags (disabled, kept for backward compat) ---
        enable_visibility_pruning=False,
        enable_periodic_ba=False,
        enable_deformation=False,
    ),

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=False,
        viz_w=320,
        viz_h=320,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=30,
        enter_interactive_post_online=True,
    ),
)
