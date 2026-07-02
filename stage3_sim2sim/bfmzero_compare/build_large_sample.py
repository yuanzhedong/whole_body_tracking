"""Build a large stratified sample of BONES-SEED artifacts for the scaled comparison.

Near-ground heavy (the interesting regime) + standing baseline for context. Verifies
each clip loads and is long enough. Writes large_sample.json (idx -> artifact + cat)
and the BFM-Zero pkl (in OMG-order-correct pose_aa via the existing converter).
"""
import json
import os
import random

import numpy as np

ARTROOT = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_FRAMES = 24

# category -> how many to sample (squat is rare -> take all available)
QUOTA = {
    "crouch": 80, "squat": 999, "sit": 80, "kneel": 40, "stoop": 30, "crawl": 30,   # near-ground
    "walk": 50, "jog": 40, "dance": 30, "turn": 25, "jump": 25, "idle": 25,          # standing
    "reach": 15, "step": 15, "kick": 10, "stand": 10, "bow": 6, "wave": 6,
    "throw": 6, "run": 6, "spin": 4, "punch": 4,
}
NEAR = {"crouch", "squat", "sit", "kneel", "stoop", "crawl"}
ORDER = list(QUOTA.keys())   # first match wins; near-ground keywords first


def cat(name):
    n = name.lower()
    for c in ORDER:
        if c in n:
            return c
    return None


def ok_length(art):
    try:
        d = np.load(f"{ARTROOT}/{art}/motion.npz", allow_pickle=True)
        return d["body_pos_w"].shape[0] >= MIN_FRAMES
    except Exception:
        return False


def main(tag="large", scale=1.0):
    random.seed(7)
    names = [n for n in os.listdir(ARTROOT) if n.endswith(":v0")]
    by_cat = {}
    for n in names:
        c = cat(n)
        if c:
            by_cat.setdefault(c, []).append(n)
    manifest = []
    for c in ORDER:
        pool = by_cat.get(c, [])
        random.shuffle(pool)
        quota = min(int(round(QUOTA[c] * scale)), len(pool))
        picked, i = [], 0
        while len(picked) < quota and i < len(pool):
            if ok_length(pool[i]):
                picked.append(pool[i])
            i += 1
        for a in picked:
            manifest.append({"idx": len(manifest), "artifact": a, "cat": c,
                             "group": "near-ground" if c in NEAR else "standing"})
        print(f"  {c:8s}: {len(picked):4d} (pool {len(pool)})")
    json.dump(manifest, open(f"{HERE}/{tag}_sample.json", "w"))
    n_near = sum(m["group"] == "near-ground" for m in manifest)
    print(f"\nTOTAL {len(manifest)} clips ({n_near} near-ground, {len(manifest)-n_near} standing)")
    print(f"wrote {HERE}/{tag}_sample.json")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "large", float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
