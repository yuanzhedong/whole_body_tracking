"""C4 VAE-decode wrapper logic (normalization, windowing, chunking, determinism).

Uses a FAKE identity VAE so we test *our* wrapper math without needing a checkpoint:
with an identity decoder, normalize->decode->denormalize must return the input.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from stage3_sim2sim.vae_decode_clip import decode_features


class _IdentityVAE:
    """model.vae(x) -> {'rec_pose': x}; mimics the BioMechanicsTokenizer interface."""
    def vae(self, x):
        return {"rec_pose": x}


def test_decode_identity_recovers_input():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((128, 41)).astype(np.float32)
    mean = rng.standard_normal(41).astype(np.float32)
    std = (1 + np.abs(rng.standard_normal(41))).astype(np.float32)
    out = decode_features(_IdentityVAE(), feats, mean, std, device="cpu", window=128)
    assert out.shape == feats.shape
    assert np.allclose(out, feats, atol=1e-4)        # norm then denorm cancels


def test_decode_handles_zero_std():
    feats = np.ones((128, 41), dtype=np.float32)
    mean = np.zeros(41, dtype=np.float32)
    std = np.zeros(41, dtype=np.float32)             # std clipped to 1e-6 internally
    out = decode_features(_IdentityVAE(), feats, mean, std, device="cpu", window=128)
    assert np.all(np.isfinite(out))


def test_decode_chunks_longer_than_window():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((300, 41)).astype(np.float32)
    mean = np.zeros(41, dtype=np.float32); std = np.ones(41, dtype=np.float32)
    out = decode_features(_IdentityVAE(), feats, mean, std, device="cpu", window=128)
    assert out.shape == (300, 41)
    assert np.allclose(out, feats, atol=1e-4)        # every chunk recovered


def test_decode_deterministic():
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((128, 41)).astype(np.float32)
    mean = np.zeros(41, dtype=np.float32); std = np.ones(41, dtype=np.float32)
    a = decode_features(_IdentityVAE(), feats, mean, std, device="cpu", window=128)
    b = decode_features(_IdentityVAE(), feats, mean, std, device="cpu", window=128)
    assert np.array_equal(a, b)
