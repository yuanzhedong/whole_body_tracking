"""Build the HoloMotion vs BFM-Zero tracker-comparison W&B report (toddler_tracking).

Layout: pipeline diagram (each tracker's input) -> summary table -> per-clip
triptych videos (Reference | HoloMotion | BFM-Zero, single reference). Re-run to
refresh.
    OMG/.venv-cu128/bin/python stage3_sim2sim/bfmzero_compare/create_tracker_compare_report.py
"""
import json
import os
import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
RUN_ID = "tracker-compare-bfm-holo-v4"
RUN_NAME = "tracker-compare-bfmzero-vs-holomotion"
HERE = os.path.dirname(os.path.abspath(__file__))

rows = json.load(open(os.path.join(HERE, "bfmzero_vs_holomotion.json")))


def vkey(r):
    return f"{r['cid']}__{r['motion'].replace(' ', '_').replace('(', '').replace(')', '')}"


# ── 1. media + table run ──────────────────────────────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT, name=RUN_NAME,
                 id=RUN_ID, resume="allow", job_type="analysis", reinit=True)

tbl = wandb.Table(columns=[
    "clip", "motion", "ref pelvis min (m)",
    "HoloMotion survival", "HoloMotion survival_rel", "HoloMotion joint°",
    "BFM-Zero survival", "BFM-Zero survival_rel", "BFM-Zero joint°"])
media = {"pipeline": wandb.Image(os.path.join(HERE, "pipeline.png"))}
for r in rows:
    tbl.add_data(r["clip"], r["motion"], r["ref_z_min"],
                 r["holo_survival"], r["holo_survival_rel"], r["holo_joint_deg"],
                 r["bfm_survival"], r["bfm_survival_rel"], r["bfm_joint_deg"])
    mp4 = os.path.join(HERE, f"triptych_{r['cid']}.mp4")
    if os.path.exists(mp4):
        media[vkey(r)] = wandb.Video(mp4, fps=30, format="mp4")
for _dfk in ("deep_flexion_squat", "deep_flexion_crouch", "depth_floor"):
    _df = os.path.join(HERE, f"{_dfk}.mp4")
    if os.path.exists(_df):
        media[_dfk] = wandb.Video(_df, fps=30, format="mp4")

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
                     filters=f"display_name == '{RUN_NAME}'")


# ── optional quantitative-analysis section (all grounded near-ground clips) ────
quant_blocks = []
QPATH = os.path.join(HERE, "quant_analysis.json")
if os.path.exists(QPATH):
    qa = json.load(open(QPATH))
    o = qa["overall"]
    agg_tbl = (
        "| group | n | HoloMotion survival_rel | **BFM-Zero survival_rel** | "
        "HoloMotion joint° (mean/med) | **BFM-Zero joint° (mean/med)** |\n"
        "|---|---|---|---|---|---|\n"
        f"| **all** | {o['n']} | {o['holo_rel_mean']:.2f} | **{o['bfm_rel_mean']:.2f}** | "
        f"{o['holo_joint_mean']:.1f} / {o['holo_joint_med']:.1f} | "
        f"**{o['bfm_joint_mean']:.1f} / {o['bfm_joint_med']:.1f}** |\n")
    for c, a in qa["by_category"].items():
        agg_tbl += (f"| {c} | {a['n']} | {a['holo_rel_mean']:.2f} | **{a['bfm_rel_mean']:.2f}** | "
                    f"{a['holo_joint_mean']:.1f} / {a['holo_joint_med']:.1f} | "
                    f"**{a['bfm_joint_mean']:.1f} / {a['bfm_joint_med']:.1f}** |\n")
    per_clip = (
        "| clip | cat | ref pelvis min | HoloMotion surv/rel/joint° | **BFM-Zero surv/rel/joint°** |\n"
        "|---|---|---|---|---|\n")
    for r in qa["rows"]:
        per_clip += (f"| `{r['clip']}` | {r['cat']} | {r['ref_z_min']:.2f} m | "
                     f"{r['holo_surv']:.2f} / {r['holo_rel']:.2f} / {r['holo_joint']:.1f} | "
                     f"**{r['bfm_surv']:.2f} / {r['bfm_rel']:.2f} / {r['bfm_joint']:.1f}** |\n")
    quant_blocks = [
        wr.H2(text=f"Quantitative analysis — all {qa['n_clips']} grounded near-ground clips"),
        wr.MarkdownBlock(text=(
            f"Beyond the {len(rows)} featured clips above, we ran **every grounded** near-ground clip "
            "in the seed set (crouch / squat / sit / crawl — references with feet on the floor, "
            "floating retargets excluded) through **both** trackers under identical physics and scored "
            "them with the same `rollout_metrics`. Aggregate (survival_rel = reference-relative survival; "
            "joint° = RMS tracking error):")),
        wr.MarkdownBlock(text=agg_tbl),
        wr.MarkdownBlock(text=(
            f"**BFM-Zero has lower joint error on {o['bfm_wins_joint']}/{o['n']} clips**, and holds the "
            f"posture (survival_rel ≥ 0.9) on **{o['bfm_rel_ge_0.9']}/{o['n']}** clips vs HoloMotion's "
            f"**{o['holo_rel_ge_0.9']}/{o['n']}**. The gap is largest on crouch/squat; the hardest cases "
            "for *both* are exotic floor postures (cross-legged sit, crawl), where BFM-Zero stays "
            "upright/seated but still can't match the exact pose. Per-clip detail:")),
        wr.MarkdownBlock(text=per_clip),
    ]

from stage3_sim2sim.bfmzero_compare.large_section import large_blocks
seed_blocks = large_blocks(HERE, wr, runset=runset)   # 569-clip scaled comparison (supersedes the 40-clip seed sample)
from stage3_sim2sim.bfmzero_compare.rootcause_section import rootcause_blocks
rc_blocks = rootcause_blocks(HERE, wr, runset=runset)
from stage3_sim2sim.bfmzero_compare.compute_section import compute_blocks
comp_blocks = compute_blocks(wr)


blocks = [
    wr.H1(text="Tracker comparison: HoloMotion vs BFM-Zero on near-ground G1 motion"),
    wr.MarkdownBlock(text=(
        "**TL;DR**\n"
        "- **HoloMotion collapses on near-ground motion; BFM-Zero holds it.** Across a **569-clip** "
        "stratified sample of the 142k-clip dataset, reference-relative survival is **0.70 → 0.98** "
        "overall (near-ground **0.54 → 0.98**; crouch 0.15→0.96, squat 0.35→1.00, kneel 0.59→0.99), with "
        "BFM-Zero lower joint error on **97%** of clips.\n"
        "- **Root cause = the policy, not data or physics.** HoloMotion initializes correctly and tracks "
        "standing motion faithfully through the same pipeline, but its policy **under-commands deep knee "
        "flexion** (commands ~49° when ~143° is needed; the knee *achieves more than commanded*, so it's "
        "not torque-limited). **BFM-Zero commands and reaches the deep flexion (~130°).**\n"
        "- **Not a size issue:** HoloMotion is the *larger* model (408 M sparse-MoE vs 32 M); both run "
        "far above the 50 Hz control loop.\n"
        "- **BFM-Zero isn't an oracle either:** at scale it shows a pelvis **depth floor** (~0.35–0.40 m; "
        "can't reach floor-sitting) and, like HoloMotion, fails inverted poses (handstands).\n"
        "- **Method:** we run our exact clips through BFM-Zero with no SMPL / no reverse-retarget, and "
        "score both trackers with the same metric.")),
    wr.MarkdownBlock(text=(
        "**Question.** Our BONES-SEED → UniMoTok-VAE → tracker → MuJoCo pipeline uses **HoloMotion** "
        "(generalist G1 tracker) as the physics validator. It executes walk/run/dance robustly but "
        "**collapses on near-ground motion** (crouch / sit / squat) — the robot face-plants to ~0.07 m. "
        "Is that a *setup/metric* problem on our side, or a *HoloMotion* capability gap? We test by "
        "running the **exact same reference clips** through a second, independent tracker — **BFM-Zero** "
        "(LeCAR-Lab promptable Forward-Backward foundation model, arXiv 2511.04131) — in the same "
        "MuJoCo physics.\n\n"
        "**Answer: it is HoloMotion-specific.** On identical clips, BFM-Zero survives 0.80–1.00 with "
        "3–4× lower joint error.")),

    wr.H2(text="Pipeline — what each tracker receives"),
    wr.MarkdownBlock(text=(
        "Both trackers are driven by the **same** reference G1 clip (`qpos_36` = root pose + 29 joint "
        "angles); only the *input format* differs. HoloMotion consumes it as its 522-d tracking "
        "observation (FEATURE joint order); BFM-Zero encodes it into its Forward-Backward latent `z` "
        "(reference dof → robot-axis-angle `pose_aa`, OMG joint order). Same MuJoCo G1 physics, same "
        "scoring.")),
    wr.PanelGrid(runsets=[runset()],
                 panels=[wr.MediaBrowser(media_keys=["pipeline"], num_columns=1)]),

    wr.H2(text="Summary"),
    wr.MarkdownBlock(text=md_table + (
        "\n*survival shown as **absolute / reference-relative** (see “About the metric” below). "
        "`joint°` = RMS joint tracking error vs reference. Both survival notions agree HoloMotion "
        "fails — so this is a real collapse, not a metric artifact.")),

    *rc_blocks,
    *seed_blocks,
    *quant_blocks,
    *comp_blocks,

    wr.H2(text="Per-clip videos"),
    wr.MarkdownBlock(text=(
        "Each video is one clip, three panels: **Reference | HoloMotion | BFM-Zero** (single "
        "reference, no duplication). All four references are **physically grounded** (lowest body "
        "within a few cm of the floor — verified by forward kinematics), so the reference itself is a "
        "valid pose the robot could hold; only the trackers differ.")),
]

for i, r in enumerate(rows, 1):
    holo = "collapses to the floor" if r["holo_survival_rel"] < 0.45 else "loses the posture"
    if r["bfm_joint_deg"] < 13:
        bfm = "tracks faithfully"
    elif r["bfm_joint_deg"] < 22:
        bfm = "holds the posture (stable, tracks the low pose well)"
    else:
        bfm = "stays stable on its feet but tracks the exact pose only loosely"
    blocks += [
        wr.H3(text=f"{i}. {r['motion']}  —  HoloMotion {holo}; BFM-Zero {bfm}"),
        wr.MarkdownBlock(text=(
            f"survival (abs / ref-rel): HoloMotion **{r['holo_survival']:.2f} / {r['holo_survival_rel']:.2f}** "
            f"vs BFM-Zero **{r['bfm_survival']:.2f} / {r['bfm_survival_rel']:.2f}**  ·  "
            f"joint error: **{r['holo_joint_deg']:.1f}°** vs **{r['bfm_joint_deg']:.1f}°**  ·  "
            f"reference pelvis dips to {r['ref_z_min']:.2f} m")),
        wr.PanelGrid(runsets=[runset()],
                     panels=[wr.MediaBrowser(media_keys=[vkey(r)], num_columns=1)]),
    ]

blocks += [
    wr.H2(text="About the metric (fixed)"),
    wr.MarkdownBlock(text=(
        "Our original survival metric was **absolute**: fraction of frames the pelvis stays above a "
        "fixed 0.4 m. That is **too crude for legitimately-low motions** — a correct deep squat/crouch "
        "reference pelvis dips to 0.20–0.42 m, so the metric flags a perfectly-tracked low posture as "
        "“fallen.” We added a **reference-relative** survival: fraction of frames the executed pelvis "
        "stays within 0.15 m *below the reference* pelvis (`executed_z > reference_z − 0.15`). It credits "
        "holding the intended posture and only penalizes an actual collapse. For standing motions the two "
        "agree; they diverge exactly where the absolute metric misleads. Implemented in "
        "`stage3_sim2sim/sim2sim.py::rollout_metrics` (`survival_rel`), with unit tests. Note both metrics "
        "still rank HoloMotion as failing here — the near-ground collapse is genuine.\n\n"
        "*One known limitation:* for **floor-level references** (pelvis < ~0.15 m, e.g. cross-legged "
        "floor sitting) the relative threshold `reference_z − 0.15` goes near/below zero, so survival_rel "
        "saturates to ~1.0 for *any* non-collapsing pose — there a tracker that barely moves can score "
        "high while tracking poorly. **Joint error is the honest discriminator at floor level** (we report "
        "it alongside survival), which is why the scaled tables include it. This is the one case where "
        "BFM-Zero's survival_rel dips below HoloMotion's despite BFM tracking the pose better (lower joint "
        "error).")),

    wr.H2(text="Method — running our G1 clips through BFM-Zero"),
    wr.MarkdownBlock(text=(
        "BFM-Zero's motion-lib (`Humanoid_Batch`) is **robot forward-kinematics**, not SMPL mesh FK "
        "(despite the “mesh_parser” name). It needs only `root_trans_offset`, `pose_aa[T,30,3]` "
        "(row 0 = root rotvec; rows 1..29 = `dof[j] · joint_axis`) and `fps`; it derives dof, velocities "
        "and body FK itself. So **no reverse-retargeting / no SMPL** was needed — we convert our "
        "already-retargeted G1 qpos directly (joints reordered FEATURE→OMG to match BFM-Zero's XML). "
        "Both trackers are scored with the same `rollout_metrics`; all videos use the same OMG MuJoCo "
        "renderer. Code: `stage3_sim2sim/to_bfmzero_motion.py`, `bfmzero_compare/` "
        "(`render_holomotion_compare.py`, `render_bfm_omg.py`, `render_triptych.py`, "
        "`make_pipeline_diagram.py`, `rescore.py`, this script), `bfmzero_tracking_inference.patch`.")),
]

report = wr.Report(entity=ENTITY, project=PROJECT,
                   title="Tracker comparison: HoloMotion vs BFM-Zero (near-ground G1)",
                   description="Our exact crouch/sit/squat clips run through both trackers — the "
                               "near-ground failure is HoloMotion-specific, not our setup.",
                   blocks=blocks)
report.save()
open("/tmp/tracker_compare_report_url.txt", "w").write(report.url)
print(f"REPORT: {report.url}")
