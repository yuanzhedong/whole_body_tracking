"""Report section for the large 569-clip survival comparison + scale findings.

Reads large_survival.json + large_issues.json. Returns wandb-report blocks ([] if absent).
"""
import json
import os


def large_blocks(here, wr, runset=None, tag="large"):
    sp, ip = os.path.join(here, f"{tag}_survival.json"), os.path.join(here, f"{tag}_issues.json")
    if not os.path.exists(sp):
        return []
    s = json.load(open(sp))
    iss = json.load(open(ip)) if os.path.exists(ip) else {}
    o = s["overall"]

    grp = ("| group | n | HoloMotion survival_rel | **BFM-Zero survival_rel** | Holo joint° | **BFM joint°** |\n"
           "|---|---|---|---|---|---|\n"
           f"| **all** | {o['n']} | {o['holo_rel']:.2f} | **{o['bfm_rel']:.2f}** | {o['holo_joint']:.0f} | "
           f"**{o['bfm_joint']:.0f}** |\n")
    for g, a in s.get("by_group", {}).items():
        grp += (f"| {g} | {a['n']} | {a['holo_rel']:.2f} | **{a['bfm_rel']:.2f}** | {a['holo_joint']:.0f} | "
                f"**{a['bfm_joint']:.0f}** |\n")

    cats = s.get("by_category", {})
    catt = "| category | n | HoloMotion survival_rel | **BFM-Zero survival_rel** |\n|---|---|---|---|\n"
    for c in sorted(cats, key=lambda c: cats[c]["holo_rel"]):
        a = cats[c]
        catt += f"| {c} | {a['n']} | {a['holo_rel']:.2f} | **{a['bfm_rel']:.2f}** |\n"

    n_deep = len(iss.get("bfm_deep_undertrack", []))
    n_bothfail = len(iss.get("both_fail", []))

    return [
        wr.H2(text=f"Scaled comparison — {o['n']} clips across the seed distribution"),
        wr.MarkdownBlock(text=(
            f"We ran **both trackers on a stratified {o['n']}-clip sample** of the 142k-clip BONES-SEED "
            "dataset (near-ground heavy + a standing baseline), full-clip rollouts (each tracker in its native MuJoCo G1 config — same robot, ±139 N·m limits; PD gains differ, see root cause), same "
            "metric. The result is unambiguous at scale:")),
        wr.MarkdownBlock(text=grp),
        wr.MarkdownBlock(text=(
            f"**BFM-Zero survives {o['bfm_rel']:.2f} (reference-relative) vs HoloMotion {o['holo_rel']:.2f}**, "
            f"and has lower joint error on **{o['bfm_wins_joint']:.0%}** of clips. The gap is concentrated "
            "near-ground but BFM-Zero is at-or-better than HoloMotion in **every** category. Survival_rel "
            "by category (sorted worst-HoloMotion first):")),
        wr.MarkdownBlock(text=catt),
        wr.H3(text="What the scale-up surfaced (issues for both)"),
        wr.MarkdownBlock(text=(
            "- **HoloMotion's weakness is broader than near-ground.** Beyond the catastrophic crouch "
            "(0.15) / squat (0.35) / kneel (0.59), it also drops on dynamic or low-reaching motion — "
            "run, punch, reach, bow — wherever the pose leaves a stable upright stance.\n"
            f"- **BFM-Zero has a pelvis *depth floor* (~0.35–0.40 m).** On the {n_deep} deepest clips "
            "(floor-sitting / extreme crouch, reference pelvis 0.05–0.20 m) it stays balanced but sits "
            "**0.13–0.44 m higher** than the reference — it doesn't reach the very lowest postures. This "
            "is a **genuine stability-vs-depth tradeoff of the stable checkpoint**, robust to settings: "
            "disabling domain-randomization/obs-noise doesn't move it, scaling the FB latent `z` makes it "
            "*shallower* (and degrades tracking), and the alternative released checkpoint reaches the "
            "floor only by **collapsing** (survival 0.44, joint 38° vs the stable model's 0.96 / 18°). "
            "Reaching controlled floor-level postures would need a different/retrained policy.\n"
            f"- **Both trackers fail inverted / extreme poses** (handstands: {n_bothfail} both-fail "
            "clips) — out-of-distribution for either policy.\n"
            "- Net: BFM-Zero is the better near-ground validator by a wide margin, but it is not a "
            "universal oracle — extreme-low and inverted postures remain open.")),
    ] + ([
        wr.MarkdownBlock(text=(
            "**The depth floor, visualized** — `Reference (floor sit) | BFM-Zero`, live pelvis height. "
            "The reference sits on the floor (~0.09 m); BFM-Zero stays balanced in a crouch (~0.5 m) and "
            "never reaches the floor — stable, but not the lowest postures:")),
        wr.PanelGrid(runsets=[runset()],
                     panels=[wr.MediaBrowser(media_keys=["depth_floor"], num_columns=1)]),
    ] if (runset and os.path.exists(os.path.join(here, "depth_floor.mp4"))) else [])
