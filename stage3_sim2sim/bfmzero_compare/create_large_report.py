"""Standalone W&B report for the scaled HoloMotion-vs-BFM-Zero comparison.

Dedicated home for the large-N sweep (survival/joint/depth by category + scale
findings + depth-floor demo). Re-run after a bigger sweep to refresh.
    OMG/.venv-cu128/bin/python stage3_sim2sim/bfmzero_compare/create_large_report.py
"""
import json
import os
import wandb
import wandb.apis.reports as wr

from stage3_sim2sim.bfmzero_compare.large_section import large_blocks

ENTITY, PROJECT = "toddler_tracking", "g1-sim2sim"
RUN_NAME = "scaled-tracker-comparison"
HERE = os.path.dirname(os.path.abspath(__file__))

s = json.load(open(f"{HERE}/large_survival.json"))
o = s["overall"]

run = wandb.init(entity=ENTITY, project=PROJECT, name=RUN_NAME, id="scaled-tracker-comparison",
                 resume="allow", job_type="analysis", reinit=True)
media = {}
for k in ("depth_floor", "pipeline"):
    p = f"{HERE}/{k}.png" if k == "pipeline" else f"{HERE}/{k}.mp4"
    if os.path.exists(p):
        media[k] = (wandb.Image(p) if p.endswith(".png") else wandb.Video(p, fps=30, format="mp4"))
if media:
    run.log(media)
run.finish()


def runset():
    return wr.Runset(entity=ENTITY, project=PROJECT, filters=f"display_name == '{RUN_NAME}'")


blocks = [
    wr.H1(text="Scaled tracker comparison: HoloMotion vs BFM-Zero"),
    wr.MarkdownBlock(text=(
        f"A large-N test of the two G1 motion trackers as physics validators. We ran **both** on a "
        f"**{o['n']}-clip stratified sample** of the 142k-clip BONES-SEED dataset (near-ground heavy + a "
        f"standing baseline), full-clip rollouts in identical MuJoCo physics, scored with the same "
        f"`rollout_metrics`. **Headline: reference-relative survival HoloMotion {o['holo_rel']:.2f} vs "
        f"BFM-Zero {o['bfm_rel']:.2f}**, BFM-Zero lower joint error on **{o['bfm_wins_joint']:.0%}** of "
        f"clips. Full root-cause analysis, per-clip videos, and the deployment/compute breakdown are in "
        f"the [companion report](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzOTg2Mg==).")),
    *large_blocks(HERE, wr, runset=runset),
    wr.H2(text="Method"),
    wr.MarkdownBlock(text=(
        "Stratified sample via `build_large_sample.py`; BFM-Zero via a batched, GPU-shardable inference "
        "(`batch_tracking_inference.py`, env built once); HoloMotion via the OMG tracker-only pipeline "
        "(`run_holo_large.py`); scored by `score_large.py`. BFM-Zero ingest is our G1 qpos → robot "
        "axis-angle `pose_aa` (no SMPL / no reverse-retarget). Both trackers scored identically; "
        "survival_rel = fraction of frames the executed pelvis stays within 0.15 m of the reference.")),
]

report = wr.Report(entity=ENTITY, project=PROJECT,
                   title="Scaled tracker comparison: HoloMotion vs BFM-Zero (large sample)",
                   description=f"Both trackers on {o['n']} clips: survival_rel "
                               f"{o['holo_rel']:.2f} vs {o['bfm_rel']:.2f}.",
                   blocks=blocks)
report.save()
print("REPORT:", report.url)
