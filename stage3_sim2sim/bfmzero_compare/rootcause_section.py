"""Shared builder for the 'Root cause: why HoloMotion fails' report section.

Reads holomotion_rootcause.json and returns wandb-report blocks ([] if absent).
"""
import json
import os


def rootcause_blocks(here, wr):
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
        wr.H2(text="Root cause — why HoloMotion fails near-ground (it's the policy)"),
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
            "**4. Decisive — the policy under-commands deep flexion (not a torque limit).** HoloMotion's "
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
    ] if bz else []) + [
        wr.MarkdownBlock(text=(
            "**Conclusion.** The near-ground failure is an **out-of-distribution policy capability gap**: "
            "HoloMotion never learned to output deep near-ground joint targets, so it under-commands "
            "flexion, can't lower its center of mass into the crouch/squat, and loses balance. It is "
            "**not** a data-feeding error and **not** a passive physics/torque sag. BFM-Zero, trained as a "
            "behavior-space foundation model, does command and hold these postures — which is why it "
            "succeeds on the same references.")),
    ]
