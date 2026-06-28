"""AMASS eval-suite Phase 1: convert + register a diverse set of AMASS G1 clips.

Serial (csv_to_npz hardcodes /tmp/motion.npz, so no parallelism here). For each clip:
  hf_to_csv.py -> repo CSV (+fps)  ->  csv_to_npz.py -> W&B motions registry (+ save npz for eval).
Writes a manifest JSON consumed by the Phase-2 train/eval driver.

Run on a free 4090, headless:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python retargeting/eval_suite/convert_all.py
"""
import json, os, re, shutil, subprocess, time, sys

REPO = "/ws/user/yzdong/src/github/whole_body_tracking"
OUT = "/tmp/amass_suite"
os.makedirs(OUT, exist_ok=True)
PY = f"{REPO}/.venv/bin/python"

# (category, repo_path_in_HF_dataset)
CLIPS = [
    ("walk",     "g1/ACCAD/Female1Walking_c3d/B1 - stand to walk_poses_120_jpos.npy"),
    ("run",      "g1/ACCAD/Female1Running_c3d/C2 - Run to stand_poses_120_jpos.npy"),
    ("jump",     "g1/BioMotionLab_NTroje/rub001/0018_jumping1_poses_120_jpos.npy"),
    ("dance",    "g1/DanceDB/20150927_VasoAristeidou/Vasso_Afraid_v1_01_poses_120_jpos.npy"),
    ("kick",     "g1/ACCAD/Male2MartialArtsKicks_c3d/G13-  cresent right_poses_120_jpos.npy"),
    ("crouch",   "g1/ACCAD/Female1General_c3d/A7 - crouch_poses_120_jpos.npy"),
    ("sidestep", "g1/ACCAD/Female1Running_c3d/C24 -  side step left_poses_120_jpos.npy"),
    ("sprint",   "g1/ACCAD/s009/Sprint1_poses_120_jpos.npy"),
    ("fight",    "g1/Eyes_Japan_Dataset/takiguchi/fighting-06-sword strong attack-takiguchi_poses_120_jpos.npy"),
    ("turn",     "g1/ACCAD/Female1Running_c3d/C11 -  run turn left (90)_poses_120_jpos.npy"),
]


def fps_of(path):
    m = re.search(r"_(\d+)_jpos", path)
    return int(m.group(1))


def main():
    manifest = []
    for cat, clip in CLIPS:
        name = f"amass_eval_{cat}"
        csv = f"{REPO}/retargeting/out/{name}.csv"
        fps = fps_of(clip)
        print(f"\n=== [{cat}] convert {clip} (fps {fps}) ===", flush=True)
        # 1) HF -> CSV
        r = subprocess.run([PY, f"{REPO}/retargeting/hf_to_csv.py", "--file", clip, "--out", csv],
                           cwd=REPO, capture_output=True, text=True)
        if not os.path.isfile(csv):
            print(f"  CONVERT FAILED: {r.stderr[-500:]}"); continue
        # 2) csv_to_npz -> registry (kill after motion.npz + upload)
        if os.path.exists("/tmp/motion.npz"):
            os.remove("/tmp/motion.npz")
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", UV_LINK_MODE="copy")
        p = subprocess.Popen([PY, f"{REPO}/scripts/csv_to_npz.py", "--input_file", csv,
                              "--input_fps", str(fps), "--output_name", name,
                              "--output_fps", "50", "--headless"],
                             cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # wait for the npz to be written (csv_to_npz loops forever after)
        for _ in range(180):
            if os.path.isfile("/tmp/motion.npz"):
                break
            if p.poll() is not None:
                break
            time.sleep(2)
        ok = os.path.isfile("/tmp/motion.npz")
        if ok:
            time.sleep(18)  # let the W&B upload finish
            shutil.copy("/tmp/motion.npz", f"{OUT}/{name}.npz")
        p.terminate()
        try: p.wait(timeout=20)
        except Exception: p.kill()
        if ok:
            print(f"  REGISTERED {name} (fps {fps})", flush=True)
            manifest.append({"category": cat, "name": name, "fps": fps,
                             "registry": f"cs224n-robustqa/wandb-registry-motions/{name}",
                             "npz": f"{OUT}/{name}.npz", "clip": clip})
        else:
            print(f"  REGISTER FAILED for {name}", flush=True)
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPHASE1_DONE: {len(manifest)}/{len(CLIPS)} registered -> {OUT}/manifest.json")


if __name__ == "__main__":
    main()
