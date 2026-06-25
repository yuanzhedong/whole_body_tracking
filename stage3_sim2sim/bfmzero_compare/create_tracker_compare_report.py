"""Build the HoloMotion vs BFM-Zero tracker-comparison W&B report (toddler_tracking).

Per-clip layout: one combined 2x2 video (top = Reference|HoloMotion, bottom =
Reference|BFM-Zero) so the contrast is obvious at a glance. Re-run to refresh.
    OMG/.venv-cu128/bin/python stage3_sim2sim/bfmzero_compare/create_tracker_compare_report.py
"""
import json
import os
import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
RUN_ID = "tracker-compare-bfm-holo-v2"
HERE = os.path.dirname(os.path.abspath(__file__))

rows = json.load(open(os.path.join(HERE, "bfmzero_vs_holomotion.json")))


def key(r):
    return f"{r['cid']}__{r['motion'].replace(' ', '_').replace('(', '').replace(')', '')}"


# ── 1. media + table run ──────────────────────────────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT,
                 name="tracker-compare-bfmzero-vs-holomotion",
                 id=RUN_ID, resume="allow", job_type="analysis", reinit=True)

tbl = wandb.Table(columns=[
    "clip", "motion", "ref pelvis min (m)",
    "HoloMotion survival", "HoloMotion survival_rel", "HoloMotion joint°",
    "BFM-Zero survival", "BFM-Zero survival_rel", "BFM-Zero joint°"])
media = {}
for r in rows:
    tbl.add_data(r["clip"], r["motion"], r["ref_z_min"],
                 r["holo_survival"], r["holo_survival_rel"], r["holo_joint_deg"],
                 r["bfm_survival"], r["bfm_survival_rel"], r["bfm_joint_deg"])
    mp4 = os.path.join(HERE, f"combined_{r['cid']}.mp4")
    if os.path.exists(mp4):
        media[key(r)] = wandb.Video(mp4, fps=30, format="mp4")

run.log({"comparison_table": tbl, **media})
run.finish()
print(f"media run: {run.url}")

# ── 2. report ─────────────────────────────────────────────────────────────────
md_table = (
    "| clip | motion | ref pelvis min | HoloMotion survival* | **BFM-Zero survival*** | "
    "Holo joint° | **BFM joint°** |\n"
    "|---|---|---|---|---|---|---|\n"
)
for r in rows:
    md_table += (f"| `{r['clip']}` | {r['motion']} | {r['ref_z_min']:.2f} m | "
                 f"{r['holo_survival']:.2f} / {r['holo_survival_rel']:.2f} | "
                 f"**{r['bfm_survival']:.2f} / {r['bfm_survival_rel']:.2f}** | "
                 f"{r['holo_joint_deg']:.1f} | **{r['bfm_joint_deg']:.1f}** |\n")


def runset():
    return wr.Runset(entity=ENTITY, project=PROJECT,
                     filters=f"display_name == 'tracker-compare-bfmzero-vs-holomotion'")


blocks = [
    wr.H1(text="Tracker comparison: HoloMotion vs BFM-Zero on near-ground G1 motion"),
    wr.MarkdownBlock(text=(
        "**Question.** Our BONES-SEED → UniMoTok-VAE → tracker → MuJoCo pipeline uses **HoloMotion** "
        "(generalist G1 tracker) as the physics validator. It executes walk/run/dance robustly but "
        "**collapses on near-ground motion** (crouch / sit / squat) — the robot face-plants to ~0.07 m. "
        "Is that a *setup/metric* problem on our side, or a *HoloMotion* capability gap? We test by "
        "running the **exact same reference clips** through a second, independent tracker — **BFM-Zero** "
        "(LeCAR-Lab promptable Forward-Backward foundation model, arXiv 2511.04131) — in the same "
        "MuJoCo physics.\n\n"
        "**Answer: it is HoloMotion-specific.** On identical clips, BFM-Zero survives 0.80–1.00 with "
        "3–4× lower joint error. Watch any clip below: same reference (left), HoloMotion falls (top "
        "right), BFM-Zero holds the posture (bottom right).")),

    wr.H2(text="Summary"),
    wr.MarkdownBlock(text=md_table + (
        "\n*survival shown as **absolute / reference-relative** (see “About the metric” below). "
        "`joint°` = RMS joint tracking error vs reference. Both survival notions agree HoloMotion "
        "fails — so this is a real collapse, not a metric artifact.")),

    wr.H2(text="Per-clip videos"),
    wr.MarkdownBlock(text=(
        "Each video is one clip, 2×2: **top row = Reference | HoloMotion**, "
        "**bottom row = Reference | BFM-Zero**. Same reference motion on both left panels.")),
]

for i, r in enumerate(rows, 1):
    holo = "collapses to the floor" if r["holo_survival_rel"] < 0.3 else "loses the posture"
    bfm = "tracks faithfully" if r["bfm_survival_rel"] >= 0.99 else "stays stable on its feet"
    blocks += [
        wr.H3(text=f"{i}. {r['motion']}  —  HoloMotion {holo}; BFM-Zero {bfm}"),
        wr.MarkdownBlock(text=(
            f"survival (abs / ref-rel): HoloMotion **{r['holo_survival']:.2f} / {r['holo_survival_rel']:.2f}** "
            f"vs BFM-Zero **{r['bfm_survival']:.2f} / {r['bfm_survival_rel']:.2f}**  ·  "
            f"joint error: **{r['holo_joint_deg']:.1f}°** vs **{r['bfm_joint_deg']:.1f}°**  ·  "
            f"reference pelvis dips to {r['ref_z_min']:.2f} m")),
        wr.PanelGrid(runsets=[runset()],
                     panels=[wr.MediaBrowser(media_keys=[key(r)], num_columns=1)]),
    ]

blocks += [
    wr.H2(text="About the metric (fixed)"),
    wr.MarkdownBlock(text=(
        "Our original survival metric was **absolute**: fraction of frames the pelvis stays above a "
        "fixed 0.4 m. That is **too crude for legitimately-low motions** — a correct deep squat/crouch "
        "reference pelvis dips to 0.27–0.58 m, so the metric flags a perfectly-tracked low posture as "
        "“fallen.” We added a **reference-relative** survival: fraction of frames the executed pelvis "
        "stays within 0.15 m *below the reference* pelvis (`executed_z > reference_z − 0.15`). It credits "
        "holding the intended posture and only penalizes an actual collapse. For standing motions the two "
        "agree; they diverge exactly where the absolute metric misleads. Implemented in "
        "`stage3_sim2sim/sim2sim.py::rollout_metrics` (`survival_rel`), with unit tests. Note both metrics "
        "still rank HoloMotion as failing here — the near-ground collapse is genuine.")),

    wr.H2(text="Method — running our G1 clips through BFM-Zero"),
    wr.MarkdownBlock(text=(
        "BFM-Zero's motion-lib (`Humanoid_Batch`) is **robot forward-kinematics**, not SMPL mesh FK "
        "(despite the “mesh_parser” name). It needs only `root_trans_offset`, `pose_aa[T,30,3]` "
        "(row 0 = root rotvec; rows 1..29 = `dof[j] · joint_axis`) and `fps`; it derives dof, velocities "
        "and body FK itself. So **no reverse-retargeting / no SMPL** was needed — we convert our "
        "already-retargeted G1 qpos directly (joints reordered FEATURE→OMG to match BFM-Zero's XML). "
        "Both trackers are scored with the same `rollout_metrics`. All videos use the same OMG MuJoCo "
        "renderer. Code: `stage3_sim2sim/to_bfmzero_motion.py`, `bfmzero_compare/` "
        "(`render_holomotion_compare.py`, `render_bfm_omg.py`, `rescore.py`, this script), "
        "`bfmzero_tracking_inference.patch`.")),
]

report = wr.Report(entity=ENTITY, project=PROJECT,
                   title="Tracker comparison: HoloMotion vs BFM-Zero (near-ground G1)",
                   description="Our exact crouch/sit/squat clips run through both trackers — the "
                               "near-ground failure is HoloMotion-specific, not our setup.",
                   blocks=blocks)
report.save()
open("/tmp/tracker_compare_report_url.txt", "w").write(report.url)
print(f"REPORT: {report.url}")
