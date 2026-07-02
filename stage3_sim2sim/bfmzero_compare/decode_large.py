"""VAE-decode the large sample (hybrid root) -> large_decoded.pkl + idx mapping.

Decodes every large-sample clip long enough for the VAE window, in the same order,
recording which large-sample idx each decoded clip maps to (so its BFM rollout can be
compared to the original-motion rollout in large/bfm/). Run in the OMG env.
"""
import json
import os
import joblib
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, build_hybrid_qpos36
from stage3_sim2sim.decode_to_qpos36 import qpos36_to_features
from stage3_sim2sim.vae_decode_clip import load_vae, decode_features
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from stage3_sim2sim.to_bfmzero_motion import qpos36_omg_to_bfmzero_motion
from stage3_sim2sim.bfmzero_compare.build_large_sample import ARTROOT

UMT = "UniMoTok"
CFG = f"{UMT}/configs/config_g1_seed_512_fixed.yaml"
CKPT = f"{UMT}/experiments/_compare/g1_seed_512_fixed_FINAL.ckpt"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PKL = "/ws/user/yzdong/src/github/BFM-Zero/pretrained/data/large_decoded.pkl"


def main():
    from omegaconf import OmegaConf
    model, ws = load_vae(CFG, CKPT, UMT, device="cpu")
    nz = np.load(f"{str(OmegaConf.load(CFG).DATASET.data_dir)}/normalization.npz")
    mean, std = nz["mean"], nz["std"]
    man = json.load(open(f"{HERE}/large_sample.json"))

    out, mapping = {}, []   # mapping[decoded_idx] = large_idx
    for m in man:
        q = build_qpos36_from_artifact(f"{ARTROOT}/{m['artifact']}/motion.npz")
        if q.shape[0] < ws:
            continue
        rec = decode_features(model, qpos36_to_features(q, 1 / 30), mean, std, device="cpu")
        dec = qpos36_feature_to_omg(build_hybrid_qpos36(rec, q))   # hybrid root, OMG order
        name = m["artifact"].replace(":", "_")
        out[name] = qpos36_omg_to_bfmzero_motion(dec, fps=30)
        mapping.append(m["idx"])
    joblib.dump(out, OUT_PKL)
    json.dump(mapping, open(f"{HERE}/large_decoded_map.json", "w"))
    print(f"decoded {len(out)} clips (>= {ws} frames) -> {OUT_PKL}; map -> large_decoded_map.json")


if __name__ == "__main__":
    main()
