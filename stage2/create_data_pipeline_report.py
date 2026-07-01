"""Publish the W&B report explaining the BONES-SEED humanoid data-generation pipeline.

Summarizes docs/DATA_PIPELINE.md (two paired tracks: robot state-action + human SMPL/SMPL-X)
with the rendered sample videos embedded. Run in an env with the reports API (.venv, wandb 0.27).
"""
import glob
import os
import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
RUN_NAME = "data-pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
md = wr.MarkdownBlock

# ── media run: the rendered robot sample videos ─────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT, name=RUN_NAME, id=RUN_NAME,
                 resume="allow", job_type="analysis", reinit=True)
media, demo_keys = {}, []
# the two hero clips first, then the full diverse gallery from docs/demo/
for fn in ["sample_Neutral_kick_trash_001__A057.mp4", "sample_jog_squat_A492.mp4"]:
    p = os.path.join(DOCS, fn)
    if os.path.exists(p):
        key = "demo_" + os.path.splitext(fn)[0].replace("sample_", "")
        media[key] = wandb.Video(p, fps=30, format="mp4"); demo_keys.append(key)
for p in sorted(glob.glob(os.path.join(DOCS, "demo", "*.mp4"))):
    key = "demo_" + os.path.splitext(os.path.basename(p))[0]
    media[key] = wandb.Video(p, fps=30, format="mp4"); demo_keys.append(key)
if media:
    run.log(media)
run.finish()
print(f"logged {len(demo_keys)} demo videos")


def runset():
    return wr.Runset(entity=ENTITY, project=PROJECT, filters=f"display_name == '{RUN_NAME}'")


blocks = [
    wr.H1(text="BONES-SEED → Humanoid Data Generation Pipeline"),
    md(text=(
        "A reproducible pipeline that turns the **BONES-SEED** motion dataset (142,220 clips) into "
        "**two paired, training-ready datasets**, keyed by the same clip name:\n\n"
        "1. **Robot state–action** — the Unitree **G1** tracking the ground-truth retarget under the "
        "**BFM-Zero** policy, logged as `(reference motion, robot state) → action` tuples.\n"
        "2. **Human SMPL / SMPL-X** — the same motions as parametric human body params (foot-corrected).\n\n"
        "Every robot clip has a matching human clip, so the two modalities are aligned for "
        "cross-modal learning. Full runnable spec: `docs/DATA_PIPELINE.md`.")),

    wr.H2(text="Design rules baked in"),
    md(text=(
        "- **No VAE anywhere.** The robot tracks the *ground-truth* retarget — a partially-trained "
        "motion VAE was tried and discarded (it smoothed away wrist 29.6°→6.6° and waist motion). "
        "BFM-Zero policy inference is the only model in the loop (it supplies the action labels).\n"
        "- **SMPL-X feet are corrected.** The raw SOMA→SMPL-X transfer plantarflexes the feet; a "
        "post-hoc per-frame ankle-alignment fix is applied (foot misalignment **35°→5°**).\n"
        "- **Ground-truth, not VAE-decoded, is the reference** — keeps full articulation.")),

    wr.H2(text="The two tracks"),
    md(text=(
        "**Track A — Robot state–action** (envs: `.venv6` prep, `bfmzero` collection)\n"
        "```\n"
        "CSV ─(1)→ G1 artifact ─(2)→ ground-truth BFM-Zero motion pkl ─(3)→ (state, action) tuples\n"
        "```\n"
        "1. `seed_to_artifacts.py`  — 36-col G1 qpos @120fps CSV → artifact `motion.npz`.\n"
        "2. `to_bfmzero_motion.py`  — CPU forward-kinematics → BFM-Zero motion-lib pkl (no VAE).\n"
        "3. `collect_state_action.py` — BFM-Zero MuJoCo rollout, two modes:\n"
        "   - **onpolicy** — closed-loop: log states BFM-Zero actually visits + the action taken.\n"
        "   - **teacher_forced** — reset onto the reference each step; log `(reference state, action)`.\n\n"
        "**Track B — Human SMPL / SMPL-X** (env: `somax`)\n"
        "```\n"
        "BVH ─→ SOMA pose ─→ SMPL & SMPL-X params ─→ foot-correction\n"
        "```\n"
        "`batch_bvh_to_smpl.py`: BVH→SOMA (FK self-check 0.78 cm) → SOMA→SMPL/SMPL-X transfer "
        "(identity setup hoisted per actor) → per-frame ankle foot-correction.")),

    wr.H2(text="Sample output — rendered robot clips"),
    md(text=(
        "`[ Reference (retarget) | BFM-Zero executed ]` on two BONES-SEED clips, rendered with the "
        "self-contained MuJoCo renderer (`stage2/render_g1_clip.py`, 960×480 @ real-time 30 fps). The "
        "left panel is the ground-truth retarget the robot tracks; the right is what BFM-Zero executes "
        "in physics — the `(state, action)` source for Track A. (The full triptych adds the SMPL-X "
        "human panel via Track B.)")),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=demo_keys, num_columns=2)]),
    md(text=(
        f"_Gallery: {len(demo_keys)} BONES-SEED clips spanning locomotion, crouch/squat/sit, jump, "
        "turn, dance, kick, punch, bow — each `[ reference retarget | BFM-Zero executed ]`._")),

    wr.H2(text="Output schema"),
    md(text=(
        "**Robot** — `state_action/{onpolicy,teacher_forced}/traj_*.npz` (per control step, len T):\n\n"
        "| key | shape | meaning |\n|---|---|---|\n"
        "| `action` | (T,29) | BFM-Zero PD-target action **(the label)** |\n"
        "| `z` | (T,256) | BFM-Zero task latent for the tracked frame |\n"
        "| `qpos`/`qvel` | (T,36)/(T,35) | robot state (MuJoCo) |\n"
        "| `obs_state`/`obs_privileged` | (T,64)/(T,P) | policy proprio / body-frame obs |\n"
        "| `ref_dof_pos`/`ref_dof_vel` | (T,29) | reference joint targets |\n"
        "| `ref_body_pos`/`ref_body_rots` | (T,B,3)/(T,B,4) | reference body poses |\n"
        "| `alive`, `reward`, `ref_frame_idx` | (T,) | not-fallen flag, tracking reward, ref index |\n\n"
        "Top level: `normalization.npz`, `manifest.json`, `dataset_stats.json`.\n\n"
        "**Human** — `smpl_smplx/uniform/<clip>.npz`: `smpl_pose`(T,24,3), `smplx_pose`(T,55,3), "
        "`*_transl`(T,3), per-vertex fit error, `foot_corrected=True`.")),

    wr.H2(text="Results (validation subset)"),
    md(text=(
        "| Dataset | Size |\n|---|---|\n"
        "| Robot state–action | **1,417,918 transitions**, mean survival **0.992** (onpolicy + teacher-forced) |\n"
        "| Human SMPL/SMPL-X | **1,420 clips**, foot-corrected, fit error ~SMPL 4 cm / SMPL-X 3.5 cm |\n\n"
        "`scale_up/loop.sh` extends both to the full **142,220-clip** corpus with the identical, "
        "resumable pipeline (every stage skips existing outputs).")),

    wr.H2(text="Gotchas / lessons"),
    md(text=(
        "- **VAE-decoded reference is bad data** — always track the ground-truth retarget for the robot side.\n"
        "- **SMPL-X feet need the ankle-alignment fix** — the default SOMA→SMPL-X transfer plantarflexes them.\n"
        "- **BFM-Zero checkpoint**: use `new_model_for_training_code_inference/` (ships the env config), not `model/`.\n\n"
        "Related: the [control-VAE distillation plan](https://wandb.ai/toddler_tracking/g1-sim2sim/"
        "reports/BFM-Zero-→-Control-VAE-distillation:-reference-motion-→-humanoid-action-(plan)--VmlldzoxNzM5MTUxNA==) "
        "consumes this robot state-action data.")),
]

report = wr.Report(
    entity=ENTITY, project=PROJECT,
    title="BONES-SEED → Humanoid Data Generation Pipeline (robot state-action + human SMPL)",
    description="Two paired training datasets from BONES-SEED: G1 robot (state,action) under BFM-Zero "
                "+ human SMPL/SMPL-X (foot-corrected). Tracks, schema, results, and sample renders.",
    blocks=blocks,
)
report.save()
url = report.url
try:
    share = report.get_share_url()
    if share:
        url = share
except Exception:
    pass
print("REPORT_URL:", report.url)
print("SHARE_URL:", url)
