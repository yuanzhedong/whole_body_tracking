"""Load a UniMoTok G1 VAE checkpoint and reconstruct (encode->decode) 41-D features.

Used by the sim2sim decoded path: a clip's 41-D features -> VAE reconstruction ->
``features_to_qpos36`` -> HoloMotion. Kept separate from ``sim2sim.py`` because it
needs torch + the UniMoTok package; importing it lazily avoids a hard dependency.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


def load_vae(cfg_path, ckpt_path, umt_root, device="cuda"):
    """Build a BioMechanicsTokenizer from config and load checkpoint state_dict."""
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(umt_root))
    from multimodal_tokenizers.models.build_model import build_model

    cfg = OmegaConf.load(cfg_path)
    model = build_model(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=False)
    model = model.eval().to(device)
    ws = int(cfg.DATASET.window_size)
    return model, ws


def decode_features(model, features, mean, std, device="cuda", window=128):
    """Encode->decode features[T,41] window-by-window; return reconstructed features[T,41].

    Normalizes with (mean,std), runs ``model.vae(x)['rec_pose']``, denormalizes.
    Processes the leading ``window`` frames (the VAE's trained window length).
    """
    import torch
    std = np.clip(np.asarray(std, np.float32), 1e-6, None)
    mean = np.asarray(mean, np.float32)
    out = np.array(features, dtype=np.float32, copy=True)
    n = features.shape[0]
    for s in range(0, n, window):
        chunk = features[s:s + window]
        if chunk.shape[0] < 8:
            break
        w = (chunk - mean) / std
        with torch.no_grad():
            rec = model.vae(torch.tensor(w, dtype=torch.float32)[None].to(device))["rec_pose"]
        rec = rec.detach().cpu().numpy()[0][:chunk.shape[0]]
        out[s:s + chunk.shape[0]] = rec * std + mean
    return out
