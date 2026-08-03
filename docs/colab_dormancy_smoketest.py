"""Colab cell: dormancy-timeout smoke test (tvs_dormancy_frames sweep).

Runs the TVS+Spatial arm on one dense scene at several dormancy thresholds K,
with K=0 as the control (current published behaviour). Compares #Gaussians and
quality so we can see whether the hard removal of the dead population buys real
compression without hurting PSNR/depth.

Paste into a Colab cell after the repo is cloned and CWD is the repo root.
"""
import os, re, shutil, subprocess, sys, glob, time

REPO = "/content/new_eGSLAM"          # repo root in Colab
CFG = f"{REPO}/configs/c3vd/c3vd_innovations.py"
SCENES = [4]                           # 4 = sigmoid_t2_a (densest clip)
KS = [0, 50, 100, 200]                 # 0 = control (no dormancy timeout)

# Canonical production TVS settings (the arm reported in the paper).
BASE = {
    "enable_tvs_pruning": "True",
    "enable_spatial_mask": "True",
    "enable_refinement": "False",      # off for the smoke test: saves ~5 min/run
    "tvs_aggregation": '"ema"',
    "tvs_ema_lambda": "0.18",
    "tvs_tau_sig": "0.02",
    "tvs_temperature": "1.0",
    "tvs_beta": "0.1",
    "tvs_opacity_floor": "0.01",
    "tvs_decay_floor": "0.3",
    "tvs_degenerate_every": "4",
    "tvs_min_obs": "50",
    "tvs_reset_on_degenerate": "True",
    "eta_spatial": "0.9",
    "distance_gamma": "0.5",
    "tvs_log_every": "100",            # prints n_dormant_removed and live N
}

os.chdir(REPO)
backup = CFG + ".bak"
if not os.path.exists(backup):
    shutil.copy(CFG, backup)


def patch(overrides, group_name):
    src = open(backup).read()
    for key, val in overrides.items():
        src, n = re.subn(rf"^(\s*){key}\s*=\s*[^,]+,", rf"\g<1>{key}={val},",
                         src, count=1, flags=re.M)
        assert n == 1, f"could not patch {key}"
    src, n = re.subn(r'^group_name\s*=\s*".*"', f'group_name = "{group_name}"',
                     src, count=1, flags=re.M)
    assert n == 1, "could not patch group_name"
    open(CFG, "w").write(src)


try:
    for k in KS:
        group = f"C3VD_dorm_K{k}"
        patch(dict(BASE, tvs_dormancy_frames=str(k)), group)
        for scene in SCENES:
            out = f"experiments/{group}/{scene}"
            if glob.glob(f"experiments/{group}/*/params.npz"):
                print(f"[skip] {group} scene {scene} already done")
                continue
            print(f"\n===== K={k}  scene_num={scene} =====", flush=True)
            t0 = time.time()
            env = dict(os.environ, SCENE_NUM=str(scene))
            subprocess.run([sys.executable, "scripts/main.py", CFG],
                           env=env, check=True)
            print(f"----- K={k} scene {scene} done in {(time.time()-t0)/60:.1f} min")
finally:
    shutil.copy(backup, CFG)          # always restore the config

# Score every arm
for k in KS:
    subprocess.run([sys.executable, "scripts/calc_metrics.py", "--all",
                    "--group_dir", f"experiments/C3VD_dorm_K{k}"], check=False)

print("\n=== summary ===")
for k in KS:
    for f in sorted(glob.glob(f"experiments/C3VD_dorm_K{k}/**/metrics_summary.csv",
                              recursive=True)):
        print(f"K={k}: {f}")
        print(open(f).read())
