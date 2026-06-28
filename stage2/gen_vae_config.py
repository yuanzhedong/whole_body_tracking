"""Generate a UniMoTok VAE training config variant from the base v2 config.

Usage:
  python gen_vae_config.py --name EX_T4w_big --data_dir <abs> --exp_dir <abs> \
      --out /tmp/cfg_EX_T4w_big.yaml --num_layers 9 --ff_size 1536 --latent 256 \
      --kl 5e-5 --end_epoch 20000 [--resume <exp_dir>]

Writes a yaml that logs to entity=cs224n-robustqa, project=g1-vae-ablation.
"""
import argparse, yaml, copy, os

BASE = "/ws/user/yzdong/src/github/whole_body_tracking/UniMoTok/configs/config_g1_mldvae_v2.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--ff_size", type=int, default=1024)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--kl", type=str, default="5e-5")
    p.add_argument("--end_epoch", type=int, default=20000)
    p.add_argument("--step_size", type=int, default=32)
    p.add_argument("--resume", default="")
    p.add_argument("--project", default="g1-vae-ablation")
    p.add_argument("--entity", default="cs224n-robustqa")
    args = p.parse_args()

    with open(BASE) as f:
        cfg = yaml.safe_load(f)

    cfg["NAME"] = args.name
    cfg["FOLDER_EXP"] = args.exp_dir
    cfg["TRAIN"]["END_EPOCH"] = args.end_epoch
    cfg["TRAIN"]["RESUME"] = args.resume
    cfg["TRAIN"]["LR_SCHEDULER"]["params"]["T_max"] = args.end_epoch

    # dataset
    cfg["DATASET"]["data_dir"] = args.data_dir
    cfg["DATASET"]["params"]["data_dir"] = args.data_dir
    cfg["DATASET"]["step_size"] = args.step_size

    # model arch
    arch = cfg["model"]["params"]["tokenizer_arch"]["params"]
    arch["num_layers"] = args.num_layers
    arch["ff_size"] = args.ff_size
    arch["latent_dim"] = [1, args.latent]
    cfg["MODEL"]["latent_dim"] = args.latent
    cfg["MODEL"]["vae_layer"] = args.num_layers

    # loss
    cfg["LOSS"]["LAMBDA_KL"] = float(args.kl)

    # logging -> team project, readable name
    w = cfg["LOGGER"]["WANDB"]["params"]
    w["project"] = args.project
    w["entity"] = args.entity
    w["name"] = args.name
    w["id"] = None
    cfg["LOGGER"]["WANDB"]["params"]["tags"] = ["ablation", args.name, f"L{args.num_layers}", f"kl{args.kl}"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.out}  (name={args.name} L={args.num_layers} ff={args.ff_size} "
          f"latent={args.latent} kl={args.kl} data={os.path.basename(args.data_dir)})")


if __name__ == "__main__":
    main()
