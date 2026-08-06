"""Colab cell: dormancy-timeout smoke test (tvs_dormancy_frames sweep).

Runs the TVS+Spatial arm on one dense scene at several dormancy thresholds K,
with K=0 as the control (current published behaviour). Compares #Gaussians and
quality so we can see whether the hard removal of the dead population buys real
compression without hurting PSNR/depth.

Run the %%writefile config cell first, then paste this into a Colab cell.
"""
import os, re, shutil, subprocess, sys, glob, time

REPO = "/content/project"                       # repo root in Colab
CFG = f"{REPO}/configs/c3vd/c3vd_test.py"       # the config the writefile cell creates
SCENES = [4]                                    # 4 = sigmoid_t2_a (densest clip)
KS = [0, 50, 100, 200]                          # 0 = control (no dormancy timeout)
GROUP = "C3VD_dormv2_K"                         # v2 = unseen-based criterion

# Scene list, must match `scenes` in the config.
SCENE_NAMES = [
    "cecum_t1_b", "cecum_t2_b", "cecum_t3_a",
    "sigmoid_t1_a", "sigmoid_t2_a", "sigmoid_t3_a",
    "trans_t1_b", "trans_t2_c", "trans_t4_a", "trans_t4_b",
]

# Canonical production TVS settings (the arm reported in the paper). Keys that
# the config does not define are inserted; nothing here should differ from the
# config except tvs_dormancy_frames, which is the variable under test.
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
# Snapshot the pristine config. Refreshed every run, so re-running the
# %%writefile cell always takes effect.
ORIGINAL = open(CFG).read()
assert "innovations=dict(" in ORIGINAL, f"{CFG} does not look like a run config"


def patch(overrides, group_name):
    src = ORIGINAL
    for key, val in overrides.items():
        src, n = re.subn(rf"^(\s*){key}\s*=\s*[^,]+,", rf"\g<1>{key}={val},",
                         src, count=1, flags=re.M)
        if n == 0:                      # key absent: add it to the innovations block
            src, n = re.subn(r"^(\s*)innovations=dict\(",
                             rf"\g<1>innovations=dict(\n\g<1>    {key}={val},",
                             src, count=1, flags=re.M)
            assert n == 1, f"could not insert {key}"
    src, n = re.subn(r'^group_name\s*=\s*".*"', f'group_name = "{group_name}"',
                     src, count=1, flags=re.M)
    assert n == 1, "could not patch group_name"
    open(CFG, "w").write(src)


# Full output goes to a log file; only lines matching this reach the cell.
KEEP = re.compile(r"\[TVS f|Traceback|Error|error:|Wrote \d+ per-frame")


def run_quiet(cmd, log_path, env=None):
    """Run a subprocess, tee its output to log_path, echo only KEEP lines."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tail = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            log.write(line)
            tail = (tail + [line])[-40:]
            if KEEP.search(line):
                print("   " + line.rstrip(), flush=True)
        rc = proc.wait()
    if rc != 0:                          # only on failure do we dump context
        print("".join(tail))
        raise RuntimeError(f"{cmd[1]} failed (rc={rc}); full log: {log_path}")


try:
    for k in KS:
        group = f"{GROUP}{k}"
        patch(dict(BASE, tvs_dormancy_frames=str(k)), group)
        for scene in SCENES:
            name = SCENE_NAMES[scene]
            if os.path.exists(f"experiments/{group}/{name}/params.npz"):
                print(f"[skip] K={k} {name} already done")
                continue
            print(f"\n===== K={k}  scene={name} =====", flush=True)
            t0 = time.time()
            run_quiet([sys.executable, "scripts/main.py", CFG],
                      f"logs/{group}_{name}.log",
                      env=dict(os.environ, SCENE_NUM=str(scene),
                               TQDM_DISABLE="1"))   # silence the per-frame bars
            print(f"----- K={k} {name} done in {(time.time()-t0)/60:.1f} min")
finally:
    open(CFG, "w").write(ORIGINAL)      # always restore the config

# Score every arm
for k in KS:
    run_quiet([sys.executable, "scripts/calc_metrics.py", "--all",
               "--group_dir", f"experiments/{GROUP}{k}"],
              f"logs/metrics_{GROUP}{k}.log")

print("\n=== summary ===")
for k in KS:
    for f in sorted(glob.glob(f"experiments/{GROUP}{k}/**/metrics_summary.csv",
                              recursive=True)):
        print(f"K={k}: {f}")
        print(open(f).read())
