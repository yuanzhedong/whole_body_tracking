"""PROOF that STOCK UniMoTok supervises root orientation for the G1 41-D feature.

Claim: the loss-fix patch (`vae_patches/biomechanics_tokenizer_rootrot6d_lossfix.patch`)
is REDUNDANT for root-orientation supervision, because stock `BioMechanicsTokenizer`
already supervises the root 6D rotation at dims [0:6] via its geodesic `root_orient_loss`
term (gated by `LAMBDA_ROOT_ORIENT`, which our configs set to 5.0).

We prove it by gradient flow on the *real* stock loss:
  * with the orient term ON  -> the loss gradient reaches dims [0:6]  (supervised)
  * with the orient term OFF -> NO gradient reaches [0:6]             (only the orient
    term touches [0:6]; the main reconstruction slice for 41-D is [9:41])

Needs the UniMoTok env (torch + multimodal_tokenizers); skips otherwise.
"""
import os
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("multimodal_tokenizers")

from omegaconf import OmegaConf
from multimodal_tokenizers.models.build_model import build_model

CFG = os.path.join(os.path.dirname(__file__), "..", "..", "UniMoTok",
                   "configs", "config_g1_seed_512_fixed.yaml")


def _build():
    if not os.path.exists(CFG):
        pytest.skip(f"config not found: {CFG}")
    return build_model(OmegaConf.load(CFG))


def _grad_on_root_orient(model, lam_orient, lam_orient_vel):
    """Backprop the stock train_vae_forward loss through a stub decoder whose output
    is WRONG only on the root orientation block; return |grad| summed over dims [0:6]."""
    model.lambda_root_orient = lam_orient
    model.lambda_root_orient_velocity = lam_orient_vel
    torch.manual_seed(0)
    target = torch.randn(2, 32, 41)
    rec = target.clone().detach()
    rec[:, :, 0:6] = rec[:, :, 0:6] + 0.3          # corrupt root orientation
    rec[:, :, 12:41] = rec[:, :, 12:41] + 0.3      # and joints (so joint grad is non-zero)
    rec = rec.requires_grad_(True)
    # stub the decoder with a plain callable returning our leaf tensor; bypass
    # nn.Module.__setattr__ (which rejects non-Module assignment to the 'vae' submodule)
    object.__setattr__(model, "vae", lambda m: {"rec_pose": rec})
    rs = model.train_vae_forward({"motion": target})
    rs["recons_loss"].backward()
    return rec.grad[:, :, 0:6].abs().sum().item(), rec.grad[:, :, 12:41].abs().sum().item()


def test_stock_supervises_root_orientation(capsys):
    model = _build()
    g_orient_on, g_joints_on = _grad_on_root_orient(model, 5.0, 0.0)   # stock config
    g_orient_off, g_joints_off = _grad_on_root_orient(model, 0.0, 0.0)  # orient term removed
    with capsys.disabled():
        print(f"\n[proof] |grad on root_rot6d[0:6]|  orient=5.0: {g_orient_on:.4f}  "
              f"orient=0: {g_orient_off:.2e}")
        print(f"[proof] |grad on joints[12:41]|     orient=5.0: {g_joints_on:.4f}  "
              f"orient=0: {g_joints_off:.4f}  (always supervised by main recons)")
    # stock (LAMBDA_ROOT_ORIENT=5) DOES push gradient into the root orientation dims:
    assert g_orient_on > 1e-4, "stock loss does not supervise root orientation [0:6]"
    # and that supervision comes ONLY from the orient term (main recons slice is [9:41]):
    assert g_orient_off < 1e-9, "something other than the orient term touches [0:6]"
    # joints are supervised regardless of the orient term (main reconstruction covers [12:41]):
    assert g_joints_on > 1e-4 and g_joints_off > 1e-4


def test_main_recons_slice_excludes_root_for_41d():
    """Documents WHY the orient term is needed: for motion_dim=41 the main reconstruction
    slice is [9:41], i.e. it excludes root_rot6d[0:6] and root_lin_vel[6:9]."""
    from biomechanics_mot_utils import BIO_MOT_DIM, BIO_ROOT_LINEAR_MODEL_SLICE
    motion_dim = 41
    assert motion_dim != BIO_MOT_DIM
    recons_slice = slice(BIO_ROOT_LINEAR_MODEL_SLICE.stop, motion_dim)
    assert recons_slice == slice(9, 41)          # main recons covers ang_vel + joints only
    assert 0 not in range(*recons_slice.indices(motion_dim))   # [0:6] root orientation excluded
