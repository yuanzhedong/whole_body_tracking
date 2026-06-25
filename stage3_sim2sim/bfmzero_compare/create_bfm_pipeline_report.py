"""Build the report: G1 motion pipeline — BONES-SEED -> UniMoTok VAE -> BFM-Zero sim2sim.

Sibling of the HoloMotion pipeline report, validated with BFM-Zero (which also
executes near-ground motion). Sections: pipeline diagram, sim2sim executability
(VAE-decoded vs original, both through BFM-Zero), near-ground coverage vs
HoloMotion (from the quantitative analysis), method. Re-run to refresh.
"""
import json
import os
import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
RUN_NAME = "bfm-pipeline-diagram"
HERE = os.path.dirname(os.path.abspath(__file__))

dec = json.load(open(f"{HERE}/decoded_analysis.json"))
quant = json.load(open(f"{HERE}/quant_analysis.json"))

# ── media run ─────────────────────────────────────────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT, name=RUN_NAME,
                 id="bfm-pipeline-diagram", resume="allow", job_type="analysis", reinit=True)
media = {"pipeline_bfm": wandb.Image(f"{HERE}/pipeline_bfm.png")}
# Result-1 pipeline demos: Reference | BFM-Zero on original | BFM-Zero on VAE-decoded
PIPE = [("pipe_crouch", "pipeline_crouch.mp4"), ("pipe_squat_1", "demo_1.mp4"),
        ("pipe_squat_2", "demo_3.mp4"), ("pipe_sit_down", "demo_5.mp4"),
        ("pipe_stand_up", "demo_6.mp4")]
# Result-2 near-ground coverage: Reference | HoloMotion | BFM-Zero (existing triptychs)
COV = [("cov_crouch", "triptych_clip0.mp4"), ("cov_squat", "triptych_clip3.mp4"),
       ("cov_sit", "triptych_clip2.mp4")]
for key, fn in PIPE + COV:
    if os.path.exists(f"{HERE}/{fn}"):
        media[key] = wandb.Video(f"{HERE}/{fn}", fps=30, format="mp4")
run.log(media)
run.finish()
pipe_keys = [k for k, fn in PIPE if os.path.exists(f"{HERE}/{fn}")]
cov_keys = [k for k, fn in COV if os.path.exists(f"{HERE}/{fn}")]
print("media run:", run.url)

# ── tables ────────────────────────────────────────────────────────────────────
do = dec["overall"]
dec_tbl = ("| clip | cat | original survival_rel | **decoded survival_rel** | orig joint° | decoded joint° |\n"
           "|---|---|---|---|---|---|\n")
for r in dec["rows"]:
    dec_tbl += (f"| `{r['clip']}` | {r['cat']} | {r['orig_surv_rel']:.2f} | "
                f"**{r['dec_surv_rel']:.2f}** | {r['orig_joint']:.1f} | {r['dec_joint']:.1f} |\n")

qo = quant["overall"]
ng_tbl = ("| group | n | HoloMotion survival_rel | **BFM-Zero survival_rel** | Holo joint° | **BFM joint°** |\n"
          "|---|---|---|---|---|---|\n"
          f"| **all** | {qo['n']} | {qo['holo_rel_mean']:.2f} | **{qo['bfm_rel_mean']:.2f}** | "
          f"{qo['holo_joint_mean']:.1f} | **{qo['bfm_joint_mean']:.1f}** |\n")
for c, a in quant["by_category"].items():
    ng_tbl += (f"| {c} | {a['n']} | {a['holo_rel_mean']:.2f} | **{a['bfm_rel_mean']:.2f}** | "
               f"{a['holo_joint_mean']:.1f} | **{a['bfm_joint_mean']:.1f}** |\n")

ng_per_clip = ("| clip | cat | ref pelvis min | HoloMotion surv/rel/joint° | **BFM-Zero surv/rel/joint°** |\n"
               "|---|---|---|---|---|\n")
for r in quant["rows"]:
    ng_per_clip += (f"| `{r['clip']}` | {r['cat']} | {r['ref_z_min']:.2f} m | "
                    f"{r['holo_surv']:.2f} / {r['holo_rel']:.2f} / {r['holo_joint']:.1f} | "
                    f"**{r['bfm_surv']:.2f} / {r['bfm_rel']:.2f} / {r['bfm_joint']:.1f}** |\n")


def runset():
    return wr.Runset(entity=ENTITY, project=PROJECT, filters=f"display_name == '{RUN_NAME}'")


from stage3_sim2sim.bfmzero_compare.seed_section import seed_survival_blocks
seed_blocks = seed_survival_blocks(HERE, wr)


blocks = [
    wr.H1(text="G1 motion pipeline: BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim"),
    wr.MarkdownBlock(text=(
        "Stage-1 of the omni-modal effort: learn a **UniMoTok VAE** motion latent on BONES-SEED G1 "
        "data and verify that **VAE-decoded motion is physically executable** — here validated with "
        "**BFM-Zero** (LeCAR-Lab Forward-Backward foundation model) in MuJoCo. This is the sibling of "
        "the [HoloMotion-validated pipeline](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/"
        "x--VmlldzoxNzMwNzgxMw==); the difference is the physics validator. BFM-Zero matters here "
        "because it **also executes near-ground motion (crouch / sit / squat)** that HoloMotion "
        "collapses on — so a BFM-Zero-validated pipeline covers more of the motion distribution.")),

    wr.H2(text="Pipeline"),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=["pipeline_bfm"], num_columns=1)]),

    wr.H2(text="Result 1 — VAE-decoded motion stays executable in BFM-Zero"),
    wr.MarkdownBlock(text=(
        f"For each grounded near-ground clip, we VAE-encode→decode the motion (lat-512 root-fixed "
        f"FINAL ckpt, full-root decode) and run both the **original** and the **decoded** motion "
        f"through BFM-Zero, scoring each against its own reference. Across {do['n']} clips, decoded "
        f"motion survives essentially as well as the original — **survival_rel "
        f"{do['orig_surv_rel_mean']:.2f} (orig) → {do['dec_surv_rel_mean']:.2f} (decoded)**, joint "
        f"tracking {do['orig_joint_mean']:.1f}° → {do['dec_joint_mean']:.1f}°, decoded survival_rel ≥ "
        f"0.9 on {do['dec_rel_ge_0.9']}/{do['n']}. The VAE latent preserves physical executability.")),
    wr.MarkdownBlock(text=dec_tbl),
    wr.MarkdownBlock(text=(
        "**Demo gallery** — each video is `Reference | BFM-Zero on original | BFM-Zero on VAE-decoded` "
        "(crouch, squat ×2, sit-down, stand-up). In every case BFM-Zero executes the VAE-reconstructed "
        "motion just like the original — the two right panels track together:")),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=pipe_keys, num_columns=1)]),

    wr.H2(text="Result 2 — BFM-Zero extends physics coverage to near-ground motion"),
    wr.MarkdownBlock(text=(
        "The validator choice matters. On every grounded near-ground seed clip, BFM-Zero holds the "
        "posture where HoloMotion collapses (full comparison: "
        "[Tracker comparison report](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/"
        "x--VmlldzoxNzMzNDI0MA==)). Aggregate (survival_rel = reference-relative survival):")),
    wr.MarkdownBlock(text=ng_tbl),
    wr.MarkdownBlock(text=(
        f"BFM-Zero has lower joint error on **{qo['bfm_wins_joint']}/{qo['n']}** clips and holds the "
        f"posture (survival_rel ≥ 0.9) on **{qo['bfm_rel_ge_0.9']}/{qo['n']}** vs HoloMotion's "
        f"**{qo['holo_rel_ge_0.9']}/{qo['n']}**. Per-clip detail (every grounded near-ground seed clip, "
        "both trackers, original references):")),
    wr.MarkdownBlock(text=ng_per_clip),
    wr.MarkdownBlock(text=(
        "`Reference | HoloMotion | BFM-Zero` on crouch, squat and sit (original references). "
        "HoloMotion collapses to the floor; BFM-Zero holds each posture:")),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=cov_keys, num_columns=1)]),
    wr.MarkdownBlock(text=(
        "So validating the VAE with BFM-Zero (vs HoloMotion) confirms decoded **crouch/squat/sit** "
        "motion is executable — motion the HoloMotion-validated pipeline could not certify. The "
        "hardest cases for both remain exotic floor postures (cross-legged sit, crawl).")),

    *seed_blocks,

    wr.H2(text="Method"),
    wr.MarkdownBlock(text=(
        "VAE: UniMoTok lat-512 continuous, root-fixed FINAL ckpt; decode = encode→decode 41-D features "
        "→ full-root `qpos_36`. BFM-Zero ingest: `qpos_36` → robot-axis-angle `pose_aa` (no SMPL / no "
        "reverse-retarget; `Humanoid_Batch` is robot FK) → FB backward-map → z. Both trackers scored "
        "with the same `rollout_metrics` (reference-relative survival). Code: "
        "`stage3_sim2sim/bfmzero_compare/` (`decode_to_bfm.py`, `score_decoded.py`, `score_quant.py`, "
        "`make_bfm_pipeline_diagram.py`, this script).")),
]

report = wr.Report(entity=ENTITY, project=PROJECT,
                   title="G1 motion pipeline: BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim",
                   description="Stage-1 VAE pipeline validated with BFM-Zero — decoded motion stays "
                               "executable, and BFM-Zero extends coverage to near-ground motion.",
                   blocks=blocks)
report.save()
open("/tmp/bfm_pipeline_report_url.txt", "w").write(report.url)
print("REPORT:", report.url)
