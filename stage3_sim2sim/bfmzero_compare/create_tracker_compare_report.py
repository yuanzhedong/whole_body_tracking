"""Build the HoloMotion vs BFM-Zero tracker-comparison W&B report (toddler_tracking).

Logs the side-by-side (expert|policy) BFM-Zero videos + key frames + a comparison
table to a run, then assembles a Report. Re-run to refresh.
    OMG/.venv-cu128/bin/python stage3_sim2sim/bfmzero_compare/create_tracker_compare_report.py
"""
import json
import os
import wandb
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"
HERE = os.path.dirname(os.path.abspath(__file__))

rows = json.load(open(os.path.join(HERE, "bfmzero_vs_holomotion.json")))

# map clip name -> (motion_id, short label)
LABEL = {
    "crouch_idle_right_R_003__A247_M": ("clip0", "deep crouch (idle)"),
    "crouch_ff_start_270_R_001__A197_M": ("clip1", "crouch + turn"),
    "sit_on_chair_stop_R_001__A047": ("clip2", "sit down"),
    "squat_001__A360": ("clip3", "squat"),
}

# ── 1. media + table run ──────────────────────────────────────────────────────
run = wandb.init(entity=ENTITY, project=PROJECT,
                 name="tracker-compare-bfmzero-vs-holomotion",
                 id="tracker-compare-bfm-holo", resume="allow",
                 job_type="analysis", reinit=True)

tbl = wandb.Table(columns=[
    "clip", "motion", "HoloMotion survival", "BFM-Zero survival",
    "HoloMotion joint°", "BFM-Zero joint°", "pelvis z-MAE (m)", "ref-rel survival", "frames"])
media = {}
for r in rows:
    cid, desc = LABEL.get(r["clip"], (f"clip{r['mid']}", ""))
    tbl.add_data(r["clip"], desc, round(r["holo_surv"], 2), round(r["bfm_surv"], 2),
                 round(r["holo_trk"], 1), round(r["bfm_jt"], 1),
                 round(r["z_mae"], 3), round(r["ref_rel"], 2), r["n"])
    mp4 = os.path.join(HERE, f"tracking_{cid}.mp4")
    if os.path.exists(mp4):
        media[f"{cid}_{r['clip'][:24]}"] = wandb.Video(mp4, fps=50, format="mp4")
for fr in ("frame_clip0_60.png", "frame_clip3_60.png"):
    p = os.path.join(HERE, fr)
    if os.path.exists(p):
        media[fr.replace(".png", "")] = wandb.Image(p, caption="left = expert reference | right = BFM-Zero policy")

run.log({"comparison_table": tbl, **media})
run.finish()
print(f"media run: {run.url}")

# ── 2. report ─────────────────────────────────────────────────────────────────
md_table = (
    "| clip | motion | HoloMotion survival | **BFM-Zero survival** | Holo joint° | **BFM joint°** | pelvis z-MAE |\n"
    "|---|---|---|---|---|---|---|\n"
)
for r in rows:
    _, desc = LABEL.get(r["clip"], ("", ""))
    md_table += (f"| `{r['clip']}` | {desc} | {r['holo_surv']:.2f} | **{r['bfm_surv']:.2f}** | "
                 f"{r['holo_trk']:.1f} | **{r['bfm_jt']:.1f}** | {r['z_mae']*100:.1f} cm |\n")

blocks = [
    wr.H1(text="Tracker comparison: HoloMotion vs BFM-Zero on near-ground G1 motion"),
    wr.MarkdownBlock(text=(
        "**Question.** Our BONES-SEED → UniMoTok-VAE → tracker → MuJoCo pipeline uses **HoloMotion** "
        "(generalist G1 tracker) as the physics validator. It robustly executes walk/run/dance but "
        "**collapses on near-ground motion** (crouch / sit / squat) — survival 0.08–0.35, the robot "
        "face-plants to ~0.07 m. Is that a *setup/metric* problem on our side, or a *HoloMotion* "
        "capability gap? We test by running the **exact same reference clips** through a second, "
        "independent tracker — **BFM-Zero** (LeCAR-Lab promptable Forward-Backward foundation model, "
        "arXiv 2511.04131) — in the same MuJoCo physics.\n\n"
        "**Answer: it is HoloMotion-specific.** On the identical clips, BFM-Zero survives 0.80–1.00 "
        "with 3–4× lower joint error. The robot, the physics, and the reference motions are fine; "
        "HoloMotion's policy simply cannot hold these low postures.")),

    wr.H2(text="Results (same clips, same MuJoCo physics)"),
    wr.MarkdownBlock(text=md_table),
    wr.MarkdownBlock(text=(
        "- **survival** = fraction of frames pelvis stays above 0.4 m (our original metric — note it is "
        "*too crude* for legitimately-low references, see caveat below).\n"
        "- **joint°** = RMS joint tracking error vs reference (lower = better).\n"
        "- **pelvis z-MAE** = mean abs error between executed and reference pelvis height (BFM-Zero tracks "
        "the *low* reference within 3.6–6.7 cm).")),

    wr.H2(text="Videos — left = expert reference, right = BFM-Zero policy"),
    wr.MarkdownBlock(text=(
        "Squat and sit are tracked faithfully (policy ≈ expert). The deepest crouch (clip0) stays a bit "
        "more upright than the reference but remains **stable on its feet** — vs HoloMotion which falls.")),
    wr.PanelGrid(
        runsets=[wr.Runset(entity=ENTITY, project=PROJECT,
                           filters="display_name == 'tracker-compare-bfmzero-vs-holomotion'")],
        panels=[wr.MediaBrowser(media_keys=[k for k in media if "frame" not in k], num_columns=2),
                wr.MediaBrowser(media_keys=[k for k in media if "frame" in k], num_columns=2)],
    ),

    wr.H2(text="Two findings, both real"),
    wr.MarkdownBlock(text=(
        "1. **HoloMotion genuinely fails near-ground motion.** A different tracker handles the identical "
        "references with high survival and low error, so the failure is HoloMotion's policy, not our "
        "reference construction or the physics. (BFM-Zero also tracks LAFAN `fallAndGetUp` — pelvis to "
        "the floor — which is *deeper* than any of these.)\n\n"
        "2. **Our survival metric is too crude for low motions.** `root_z > 0.4 m` cannot distinguish a "
        "legitimate deep squat/crouch (reference pelvis dips to 0.27–0.44 m) from a fall. The "
        "**reference-relative** survival (executed pelvis within 0.15 m of the *reference* height) is the "
        "fair metric — BFM-Zero scores 0.93–1.00 on it. We should adopt ref-relative survival / tracking "
        "error going forward.")),

    wr.H2(text="Method — running our G1 clips through BFM-Zero"),
    wr.MarkdownBlock(text=(
        "BFM-Zero's motion-lib (`Humanoid_Batch`) is **robot forward-kinematics**, not SMPL mesh FK "
        "(despite the “mesh_parser” name). It needs only `root_trans_offset`, `pose_aa[T,30,3]` "
        "(row 0 = root rotvec; rows 1..29 = `dof[j] · joint_axis`) and `fps`; it derives dof, "
        "velocities and body FK itself. So **no reverse-retargeting / no SMPL** was needed — we convert "
        "our already-retargeted G1 qpos directly. Joints are reordered FEATURE→OMG (BFM-Zero's XML "
        "order). Converter + scoring + the instrumentation patch are committed under "
        "`stage3_sim2sim/` (`to_bfmzero_motion.py`, `bfmzero_vs_holomotion.json`, "
        "`bfmzero_tracking_inference.patch`). Both trackers scored with the same `rollout_metrics`.")),
]

report = wr.Report(entity=ENTITY, project=PROJECT,
                   title="Tracker comparison: HoloMotion vs BFM-Zero (near-ground G1)",
                   description="Our exact crouch/sit/squat clips run through both trackers — the "
                               "near-ground failure is HoloMotion-specific, not our setup.",
                   blocks=blocks)
report.save()
open("/tmp/tracker_compare_report_url.txt", "w").write(report.url)
print(f"REPORT: {report.url}")
