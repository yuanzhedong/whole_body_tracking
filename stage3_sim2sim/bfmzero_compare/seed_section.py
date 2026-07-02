"""Shared builder for the 'Seed-dataset survival rate' report section.

Reads seed_survival.json (HoloMotion vs BFM-Zero on the representative seed sample)
and returns a list of wandb-report blocks. Returns [] if the json is absent.
"""
import json
import os


def seed_survival_blocks(here, wr, n_full=100, holo_full=0.83):
    path = os.path.join(here, "seed_survival.json")
    if not os.path.exists(path):
        return []
    s = json.load(open(path))
    o = s["overall"]

    overall_tbl = (
        "| group | n | HoloMotion survival (abs / ref-rel) | **BFM-Zero survival (abs / ref-rel)** |\n"
        "|---|---|---|---|\n"
        f"| **whole sample** | {o['n']} | {o['holo_surv']:.2f} / {o['holo_rel']:.2f} | "
        f"**{o['bfm_surv']:.2f} / {o['bfm_rel']:.2f}** |\n")
    for g, a in s.get("by_group", {}).items():
        overall_tbl += (f"| {g} | {a['n']} | {a['holo_surv']:.2f} / {a['holo_rel']:.2f} | "
                        f"**{a['bfm_surv']:.2f} / {a['bfm_rel']:.2f}** |\n")

    cat_tbl = "| category | n | HoloMotion survival | **BFM-Zero survival** |\n|---|---|---|---|\n"
    for c, a in s.get("by_category", {}).items():
        cat_tbl += f"| {c} | {a['n']} | {a['holo_surv']:.2f} | **{a['bfm_surv']:.2f}** |\n"

    return [
        wr.H2(text="Seed-dataset survival rate (HoloMotion vs BFM-Zero)"),
        wr.MarkdownBlock(text=(
            f"Zooming out from near-ground to the **whole motion distribution**: we ran both trackers "
            f"on a seeded **{o['n']}-clip representative sample** of the BONES-SEED dataset (walk / jog / "
            f"dance / turn / squat / sit / … — the same clips, same MuJoCo physics, same "
            f"`rollout_metrics`). Survival is shown as **absolute** (pelvis > 0.4 m) and "
            f"**reference-relative** (within 0.15 m of the reference pelvis — fair for low motions). "
            f"For context, HoloMotion over the full {n_full}-clip sweep averages **{holo_full:.2f}** "
            f"absolute survival.")),
        wr.MarkdownBlock(text=overall_tbl),
        wr.MarkdownBlock(text=(
            "**Both trackers are strong on standing motion** (walk/jog/dance/turn) — that is the bulk of "
            "the dataset and where HoloMotion already does well. **BFM-Zero's advantage is concentrated "
            "in the near-ground slice**, which drags HoloMotion's overall number down. Split by category:")),
        wr.MarkdownBlock(text=cat_tbl),
    ]
