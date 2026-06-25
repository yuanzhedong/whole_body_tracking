"""VAE-decode the quant clips and emit a BFM-Zero pkl of the DECODED motion.

Closes the full pipeline BONES-SEED -> UniMoTok VAE -> decode -> qpos_36 ->
BFM-Zero. For each grounded near-ground clip: original qpos -> 41-D features ->
VAE encode/decode -> full-root qpos_36 (root-fixed FINAL ckpt) -> reorder to OMG
-> BFM-Zero pkl entry. Run in the OMG env (torch + UniMoTok via umt_root).
"""
import os
import numpy as np
import joblib

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, build_hybrid_qpos36
from stage3_sim2sim.decode_to_qpos36 import qpos36_to_features, features_to_qpos36
from stage3_sim2sim.vae_decode_clip import load_vae, decode_features
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from stage3_sim2sim.to_bfmzero_motion import qpos36_omg_to_bfmzero_motion
from stage3_sim2sim.bfmzero_compare.quant_clips import QUANT_CLIPS, ART

UMT = "UniMoTok"
CFG = f"{UMT}/configs/config_g1_seed_512_fixed.yaml"
CKPT = f"{UMT}/experiments/_compare/g1_seed_512_fixed_FINAL.ckpt"
OUT_PKL = "/ws/user/yzdong/src/github/BFM-Zero/pretrained/data/quant_decoded.pkl"


def main():
    from omegaconf import OmegaConf
    model, ws = load_vae(CFG, CKPT, UMT, device="cpu")
    data_dir = str(OmegaConf.load(CFG).DATASET.data_dir)
    nz = np.load(f"{data_dir}/normalization.npz")
    mean, std = nz["mean"], nz["std"]

    out = {}
    for idx, art, cat, label in QUANT_CLIPS:
        qpos_gt = build_qpos36_from_artifact(f"{ART}/{art}/motion.npz")    # FEATURE order
        if qpos_gt.shape[0] < ws:
            print(f"[{idx}] too short ({qpos_gt.shape[0]}<{ws}), skip"); continue
        feats = qpos36_to_features(qpos_gt, 1 / 30)
        rec = decode_features(model, feats, mean, std, device="cpu")
        # HYBRID root (decoded joints + original root): the full-root integration drifts
        # horizontally (>0.5 m on ~44% of clips, the known double-yup issue) -- decode_robustness.json.
        # Hybrid isolates the VAE's job (the joints) and matches the HoloMotion sim2sim.
        decoded = build_hybrid_qpos36(rec, qpos_gt)
        dec_omg = qpos36_feature_to_omg(decoded)
        name = art.replace(":", "_")
        out[name] = qpos36_omg_to_bfmzero_motion(dec_omg, fps=30)
        print(f"[{idx}] {cat:6s} {label}: decoded T={dec_omg.shape[0]} "
              f"root_z[{dec_omg[:,2].min():.2f},{dec_omg[:,2].max():.2f}]")
    joblib.dump(out, OUT_PKL)
    print(f"\nwrote {len(out)} decoded clips -> {OUT_PKL}")
    print("order:", {i: n for i, n in enumerate(out)})


if __name__ == "__main__":
    main()
