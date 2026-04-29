"""
Configuration for C3VD dataset with all three innovations.

Innovations can be toggled independently via the 'innovations' dict.
Scene is selected via SCENE_NUM environment variable (default: sigmoid_t3_a).
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
tracking_iters = 15
mapping_iters = 25

group_name = "C3VD_innovations"
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
    report_global_progress_every=2000,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=int(1e10),
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
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
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
        loss_weights=dict(
            im=1.0,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.000,
            cam_trans=0.000,
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
    # INNOVATION CONFIG - Toggle each independently
    # ============================================================
    innovations=dict(
        # --- Innovation 1: Visibility-aware dual-mask pruning ---
        enable_visibility_pruning=True,
        distance_gamma=0.05,          # Depth diff threshold for floater detection
        degeneration_eta=0.2,         # Opacity degeneration factor (multiply)
        vis_threshold=0.3,            # Min mean visibility to be considered valid
        min_observations=5,           # Min frames before visibility pruning activates
        vis_window_size=15,           # Circular buffer size for visibility history

        # --- Innovation 2: Periodic Bundle Adjustment ---
        enable_periodic_ba=True,
        ba_every_m_frames=20,         # BA trigger frequency (frames)
        ba_n_keyframes=5,             # Number of keyframes per BA session
        ba_num_iters=30,              # BA optimization iterations per session
        ba_selection="hybrid",        # Keyframe selection: 'uniform', 'recent', 'hybrid'
        ba_lrs=dict(
            means3D=0.00005,
            rgb_colors=0.001,
            unnorm_rotations=0.0005,
            logit_opacities=0.025,
            log_scales=0.0005,
            cam_unnorm_rots=0.001,    # Non-zero: camera poses get gradients during BA
            cam_trans=0.002,
        ),

        # --- Innovation 3: Deformation modeling (experimental) ---
        enable_deformation=True,
        deform_lr=0.0005,
        var_threshold=0.1,            # Visibility variance threshold for deformation
        lambda_deform_temporal=0.1,
        lambda_deform_magnitude=0.01,
        enable_deform_weighted_tracking=True,
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
