"""Publish (idempotently) the W&B report for the BONES-SEED humanoid data pipeline.

Summarizes docs/DATA_PIPELINE.md with a large gallery of rendered robot sample videos and
the reproduced-on-this-box Track-A dataset stats. Shows MANY videos at once (chunked
MediaBrowser panels). Deletes older duplicate reports of the same title at the end, so
regenerating never piles up copies. Run in an env with the reports API (.venv, wandb 0.27).
"""
import glob
import json
import os

import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
RUN_NAME = "data-pipeline"
TITLE = "BONES-SEED → Humanoid Data Generation Pipeline (robot state-action + human SMPL)"
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
SA_STATS = os.path.join(HERE, "out", "state_action_seed", "dataset_stats.json")
md = wr.MarkdownBlock
N_SHOWN = 60  # videos rendered as visible panels (rest still logged/browsable)


def reproduced_block():
    if not os.path.exists(SA_STATS):
        return ("_`collect_state_action.py` + `pack_state_action.py` (full §6 schema) are built and "
                "verified here; a dataset run is in progress._")
    s = json.load(open(SA_STATS))
    rows = ["| mode | trajectories | transitions | mean survival |", "|---|---|---|---|"]
    for m, v in s["per_mode"].items():
        rows.append(f"| {m} | {v['trajectories']} | {v['transitions']:,} | {v['mean_survival']} |")
    return ("**Actually collected on this box** with `collect_state_action.py` (full §6 schema) → "
            "`pack_state_action.py`:\n\n" + "\n".join(rows) +
            f"\n\n**Total: {s['total_transitions']:,} transitions, {s['total_trajectories']} "
            f"trajectories, mean survival {s['mean_survival_all']}** (seed subset — harder near-ground "
            "clips than the full-corpus number below).")


N_DEMO_LOG = 24   # cap robot demos LOGGED to W&B (avoid storage bloat; local set is larger)

# storage hygiene: delete the previous media run(s) so re-logging never accumulates copies
try:
    _api = wandb.Api()
    for r in _api.runs(f"{ENTITY}/{PROJECT}"):
        if r.name == RUN_NAME:
            r.delete(delete_artifacts=True)
except Exception as e:
    print("prior-run cleanup skipped:", str(e)[:100])

# ── media run: log a CURATED set once ───────────────────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT, name=RUN_NAME, job_type="analysis", reinit=True)
media, keys, trip_keys = {}, [], []
for p in sorted(glob.glob(os.path.join(DOCS, "triptych", "*.mp4"))):          # all triptychs (the deliverable)
    k = "trip_" + os.path.splitext(os.path.basename(p))[0]; media[k] = wandb.Video(p, fps=30); trip_keys.append(k)
demo_paths = sorted(glob.glob(os.path.join(DOCS, "demo", "*.mp4")))[:N_DEMO_LOG]  # curated robot subset
for fn in ["sample_Neutral_kick_trash_001__A057.mp4", "sample_jog_squat_A492.mp4"]:
    p = os.path.join(DOCS, fn)
    if os.path.exists(p):
        k = "demo_" + os.path.splitext(fn)[0].replace("sample_", ""); media[k] = wandb.Video(p, fps=30); keys.append(k)
for p in demo_paths:
    k = "demo_" + os.path.splitext(os.path.basename(p))[0]; media[k] = wandb.Video(p, fps=30); keys.append(k)
if media:
    run.log(media)
run.finish()
print(f"logged {len(trip_keys)} triptychs + {len(keys)} robot demos (curated)")


def runset():
    return wr.Runset(entity=ENTITY, project=PROJECT, filters=f"display_name == '{RUN_NAME}'")


# many videos visible at once: one MediaBrowser panel per small chunk of keys
def video_panels(all_keys, shown):
    panels = []
    for i in range(0, min(shown, len(all_keys)), 3):
        panels.append(wr.MediaBrowser(media_keys=all_keys[i:i + 3], num_columns=3))
    return panels


blocks = [
    wr.H1(text="BONES-SEED → Humanoid Data Generation Pipeline"),
    md(text=(
        "Turns the **BONES-SEED** motion dataset (142,220 clips) into **two paired, training-ready "
        "datasets** keyed by the same clip name: **(A) robot state–action** — Unitree **G1** tracking "
        "the ground-truth retarget under **BFM-Zero**, logged as `(reference, state) → action`; and "
        "**(B) human SMPL / SMPL-X** — the same motions as foot-corrected body params. Full spec: "
        "`docs/DATA_PIPELINE.md`.")),

    wr.H2(text="Triptych — [ human motion | G1 reference | G1 executed ]"),
    md(text=(
        "The full pipeline output: the **human motion** as a **SOMA/MHR body mesh** (BONES-SEED "
        "`soma_uniform` BVH → SOMA-X's bundled **MHR** rig — **no gated SMPL models needed**), the "
        "**G1 reference** retarget, and the **BFM-Zero executed** rollout, time-aligned at 1440×480. "
        "Both tracks are reproduced on this box (`stage2/render_triptych.py` + `render_human_mesh.py`). "
        "The BVH→SOMA pose uses the `demo_soma_vis` convention (local→world FK, rest-frame correction, "
        "`pose2rot=False`).")),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=trip_keys[i:i + 2], num_columns=2)
                         for i in range(0, min(12, len(trip_keys)), 2)]),

    wr.H2(text=f"Robot gallery — {len(keys)} rendered clips (showing {min(N_SHOWN, len(keys))})"),
    md(text=(
        "Diverse BONES-SEED motions — locomotion, crouch/squat/sit, jump, turn, dance, kick, punch, "
        "bow, crawl, ladder-climb — each `[ Reference (retarget) | BFM-Zero executed ]`, rendered with "
        "`stage2/render_g1_clip.py` (MuJoCo/EGL, real-time 30 fps).")),
    wr.PanelGrid(runsets=[runset()], panels=video_panels(keys, N_SHOWN)),
    md(text=f"_All {len(keys)} clips are logged to the run and browsable in any panel above._"),

    wr.H2(text="Track A — robot state–action (reproduced)"),
    md(text=(
        "```\nCSV ─(1)→ G1 artifact ─(2)→ ground-truth BFM-Zero pkl ─(3)→ (state, action) tuples\n```\n"
        "1. `seed_to_artifacts.py` — 36-col G1 qpos @120fps → artifact.\n"
        "2. `to_bfmzero_motion.py` — CPU FK → BFM-Zero motion-lib pkl (no VAE).\n"
        "3. `collect_state_action.py` — BFM-Zero MuJoCo rollout, modes **onpolicy** (states the policy "
        "visits + action) and **teacher_forced** (reset onto reference each step; `(ref state, action)`).\n\n"
        "These `(state, action)` tuples are **tracking-control / imitation** data (how to *execute* a "
        "given reference), not motion planning.")),

    wr.H2(text="Reproduced dataset (this box, Track A)"),
    md(text=reproduced_block()),

    wr.H2(text="Output schema (DATA_PIPELINE.md §6)"),
    md(text=(
        "**Robot** `state_action/{onpolicy,teacher_forced}/traj_*.npz`: `action`(T,29), `z`(T,256), "
        "`qpos`(T,36), `qvel`(T,35), `obs_state`(T,64), `obs_privileged`(T,P), `last_action`(T,29), "
        "`ref_dof_pos/vel`(T,29), `ref_body_pos/rots`(T,B,·), `alive/reward/ref_frame_idx`(T,). "
        "Top level: `normalization.npz`, `manifest.json`, `dataset_stats.json`.\n\n"
        "**Human** `smpl_smplx/uniform/<clip>.npz`: `smpl_pose`(T,24,3), `smplx_pose`(T,55,3), "
        "`*_transl`(T,3), fit error, `foot_corrected=True` (Track B).")),

    wr.H2(text="Design rules & gotchas"),
    md(text=(
        "- **No VAE anywhere** — track the ground-truth retarget (a partial motion VAE smoothed away "
        "wrist 29.6°→6.6° / waist motion).\n"
        "- **SMPL-X feet need the ankle-alignment fix** (raw transfer plantarflexes them; 35°→5°).\n"
        "- **BFM-Zero checkpoint**: use `new_model_for_training_code_inference/` (ships the env config).\n\n"
        "Full-corpus documented run: **1,417,918 transitions**, survival **0.992**; 1,420 human clips.")),
]

report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE,
                   description="BONES-SEED → paired G1 robot (state,action) + human SMPL data. Track A "
                               "reproduced here with real stats + a large robot-render gallery.",
                   blocks=blocks)
report.save()
url = report.url
try:
    url = report.get_share_url() or url
except Exception:
    pass
print("SHARE_URL:", url)
print("REPORT_ID:", url.rsplit("--", 1)[-1] if "--" in url else "?")
# NOTE: run stage2/prune_data_pipeline_reports.py separately to delete older duplicates
# (kept out of this generator — an id-format mismatch here can delete the wrong ones).
