"""Standalone windowed-RMSE eval for a UniMoTok MldVaeBiomechanics checkpoint.
Runs in .venv_umt (has model deps). Prints a JSON line with val + train RMSE.

  .venv_umt/bin/python stage2/eval_vae_rmse.py \
      --ckpt <epoch=N.ckpt> --data_dir <g1_dataset_*> \
      --num_layers 5 --ff_size 1024 --latent 128
"""
import argparse, os, sys, glob, json
import numpy as np
import torch
import omegaconf

sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/UniMoTok")
from multimodal_tokenizers.archs.mld_vae import MldVaeBiomechanics


def windowed_rmse(vae, paths, mean, std, ws=128, stride=64):
    if not paths:
        return None
    vals = []
    for p in paths:
        m = torch.tensor(np.load(p)["motion"], dtype=torch.float32)
        mn = (m - mean) / std
        wins = [mn[i:i+ws] for i in range(0, max(1, len(mn)-ws+1), stride)]
        x = torch.stack([w if len(w) == ws else
                         torch.nn.functional.pad(w, (0, 0, 0, ws-len(w))) for w in wins])
        with torch.no_grad():
            rec = vae(x)["rec_pose"]
        jt = slice(12, 41)  # 29 joint angles
        rmse = ((x[:, :, jt]*std[jt]+mean[jt]) - (rec[:, :, jt]*std[jt]+mean[jt])).pow(2).mean().sqrt().item()
        vals.append(rmse)
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--latent", type=int, default=128)
    args = p.parse_args()

    arch = omegaconf.OmegaConf.create({
        "vae_test_dim": 41, "latent_dim": [1, args.latent], "ff_size": args.ff_size,
        "num_layers": args.num_layers, "num_heads": 8, "dropout": 0.15,
        "arch": "encoder_decoder", "normalize_before": False,
        "activation": "gelu", "pe_type": "actor", "mlp_dist": False})
    vae = MldVaeBiomechanics(arch)
    sd = {k[len("tokenizer_arch."):]: v
          for k, v in torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"].items()
          if k.startswith("tokenizer_arch.")}
    vae.load_state_dict(sd, strict=False)
    vae.eval()

    norm = np.load(os.path.join(args.data_dir, "normalization.npz"))
    mean = torch.tensor(norm["mean"]); std = torch.tensor(norm["std"])

    val_paths = sorted(glob.glob(os.path.join(args.data_dir, "val", "*.npz")))
    trn_paths = sorted(glob.glob(os.path.join(args.data_dir, "train", "*.npz")))
    epoch = int(os.path.basename(args.ckpt).replace("epoch=", "").replace(".ckpt", ""))

    out = {"ckpt": args.ckpt, "epoch": epoch,
           "val_rmse": round(windowed_rmse(vae, val_paths, mean, std), 5),
           "train_rmse": round(windowed_rmse(vae, trn_paths, mean, std), 5)}
    out["gap"] = round(out["val_rmse"] - out["train_rmse"], 5)
    print("RMSE_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
