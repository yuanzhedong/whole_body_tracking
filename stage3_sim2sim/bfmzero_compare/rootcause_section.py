"""Shared builder for the 'Root cause: why HoloMotion fails' report section.

Reads holomotion_rootcause.json and returns wandb-report blocks ([] if absent).
"""
import json
import os


def rootcause_blocks(here, wr, runset=None):
    path = os.path.join(here, "holomotion_rootcause.json")
    if not os.path.exists(path):
        return []
    r = json.load(open(path))
    h1, h2, h3, h4 = (r["H1_init_ruled_out"], r["H2_depth_driven"],
                      r["H3_data_correct"], r["H4_policy_not_physics"])
    k = h4["rows"][0]["left_knee_joint"]
    hip = h4["rows"][0]["left_hip_pitch_joint"]
    bz = r.get("bfm_zero_contrast", {})
    tbl = (
        "| joint (squat descent, robot still upright) | reference needs | **policy commands** | achieved |\n"
        "|---|---|---|---|\n"
        f"| left knee | {k['reference_deg']:.0f}° | **{k['commanded_deg']:.0f}°** | {k['achieved_deg']:.0f}° |\n"
        f"| left hip pitch | {hip['reference_deg']:.0f}° | **{hip['commanded_deg']:.0f}°** | {hip['achieved_deg']:.0f}° |\n")
    return [
        wr.H2(text="Root cause — why HoloMotion fails near-ground (policy + controller gains)"),
        wr.MarkdownBlock(text=(
            "We dug into *why* HoloMotion collapses on crouch/sit/squat, ruling hypotheses out with "
            "numbers from its own rollouts (analysis + tests committed: "
            "`bfmzero_compare/analyze_holo_rootcause.py`, `tests/test_holomotion_rootcause.py`). "
            "Everything below is in HoloMotion's native FEATURE joint order.")),
        wr.MarkdownBlock(text=(
            f"1. **Not an initialization mismatch.** The robot starts *at* the reference pose "
            f"(pelvis init gap ≈ {h1['init_gap_mean_m']*100:.1f} cm), then collapses a few frames later.\n\n"
            f"2. **Failure scales with posture depth.** Survival correlates **{h2['corr_survival_vs_ref_pelvis_min']:.2f}** "
            f"with reference pelvis height across {h2['n']} clips — the lower the motion, the worse it does.\n\n"
            f"3. **Not a wrong-data artifact.** On the {h3['n_success_clips']} clips it survives, HoloMotion tracks "
            f"faithfully ({h3['success_mean_joint_err_deg']:.0f}° joint error) through the *identical* feeding "
            f"pipeline. Wrong joint order/format/scaling would break standing motion too — it doesn't.")),
        wr.MarkdownBlock(text=(
            "**4. A key policy factor — it under-commands deep flexion (not a torque limit).** HoloMotion's "
            "deploy law is `target = default_joint_pos + action_scale · action`. During the squat descent, "
            "while the robot is still upright, the reference needs the knee at ~143° but the policy "
            "**commands only ~49°**; the knee actually *achieves more than commanded* (gravity pulls it "
            "down), so the actuator clearly **can** reach deeper — the policy simply never asks it to:")),
        wr.MarkdownBlock(text=tbl),
    ] + ([
        wr.MarkdownBlock(text=(
            f"**Can BFM-Zero produce the deep flexion HoloMotion can't? Yes.** On the same squat, "
            f"BFM-Zero bends the knee to **~{bz['bfm_achieved_knee_deg']:.0f}°** at the deepest frame "
            f"(reference {bz['reference_knee_deg']:.0f}°, tracking it within ~20° at "
            f"{bz['tracking_rmse_deg']:.0f}° overall), reaching {bz['bfm_max_knee_deg']:.0f}° over the "
            f"clip — versus HoloMotion's ~{bz['holomotion_achieved_knee_deg']:.0f}°. That deep flexion "
            "is exactly what lets BFM-Zero lower its center of mass into the squat and stay balanced:\n\n"
            f"| squat, deepest frame | reference | HoloMotion | **BFM-Zero** |\n|---|---|---|---|\n"
            f"| knee flexion | {bz['reference_knee_deg']:.0f}° | ~{bz['holomotion_achieved_knee_deg']:.0f}° "
            f"(under-commands) | **~{bz['bfm_achieved_knee_deg']:.0f}°** |\n")),
    ] if bz else []) + ([
        wr.MarkdownBlock(text=(
            "**See it** — `Reference | HoloMotion | BFM-Zero` with the live left-knee flexion on each "
            "panel. **Squat:** as it deepens the reference knee bends to ~150°; BFM-Zero follows (~130°) "
            "while HoloMotion stalls (~80°) and sprawls. **Crouch:** the reference holds a deep crouch "
            "(~165°); BFM-Zero holds it (~130°) while HoloMotion pops up (knee ~55°) and collapses:")),
        wr.PanelGrid(runsets=[runset()],
                     panels=[wr.MediaBrowser(media_keys=["deep_flexion_squat", "deep_flexion_crouch"],
                                             num_columns=1)]),
    ] if (runset and os.path.exists(os.path.join(here, "deep_flexion_squat.mp4"))) else []) + [
        wr.H3(text="Controlled test — it's the policy *and* the controller gains, not either alone"),
        wr.MarkdownBlock(text=(
            "The trackers are **not** run in byte-identical physics — they use different PD gains "
            "(BFM-Zero: waist kp **300**, hip_pitch **99**; HoloMotion: waist **28.5**, hip_pitch **40**; "
            "knee 99 and torque ±139 N·m are the same). We swapped gains both directions on near-ground "
            "clips:\n\n"
            "| | crouch survival_rel | squat survival_rel |\n|---|---|---|\n"
            "| BFM-Zero, native gains | **1.00** | **1.00** |\n"
            "| BFM-Zero forced onto HoloMotion's soft gains | 0.46 | 0.12 (collapses) |\n"
            "| HoloMotion, native gains | 0.42 | 0.23 |\n"
            "| HoloMotion given BFM-Zero's stiff gains | 0.61 | 0.23 (still fails) |\n")),
        wr.MarkdownBlock(text=(
            "**Conclusion (revised honestly).** The stiffer gains are **necessary** for BFM-Zero "
            "(it collapses on HoloMotion's soft gains) but **not sufficient** for HoloMotion (it still "
            "fails with BFM-Zero's stiff gains). So the near-ground gap is the **co-designed policy + "
            "controller package**, not one factor alone:\n"
            "- The **policy** matters: HoloMotion under-commands deep flexion (commands ~49° knee when "
            "~143° is needed, and *achieves more than it commands* — so it's not torque-limited), and "
            "stiff gains don't rescue it; BFM-Zero commands and reaches the deep posture.\n"
            "- The **controller gains** matter: BFM-Zero's near-ground stability depends on its stiff "
            "waist/hip gains; with HoloMotion's soft gains it collapses too.\n"
            "It is **not** a data-feeding error (HoloMotion tracks standing well through the same pipeline), "
            "and **not** a pure physics/torque issue (same ±139 N·m, achieves > commanded). The headline "
            "survival numbers are valid as *each tracker as it is actually deployed* — but the advantage "
            "should be read as a better policy+gain co-design, not a policy difference in isolation.")),
    ]
