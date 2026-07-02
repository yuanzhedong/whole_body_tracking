"""Create (or overwrite) a root W&B Report that links to every sub-report and project.

Run any time to refresh:
    .venv/bin/python stage2/create_root_report.py
"""
import wandb
import wandb.apis.reports as wr

ENTITY  = "cs224n-robustqa"
PROJECT = "beyondmimic-tracking"   # root report lives in the main project

# ── Known reports (reportID from URL: /reports/title--<ID>) ──────────────────
REPORTS = [
    {
        "id":    "VmlldzoxNzI3NzE3NA==",
        "title": "BONES-SEED → UniMoTok VAE → HoloMotion sim2sim (CURRENT EFFORT)",
        "desc":  "Living tracking page for the active effort: train UniMoTok VAE on BONES-SEED (142k G1 clips, "
                 "288h) and validate via the HoloMotion generalist tracker (MuJoCo, Blackwell) instead of "
                 "per-clip teachers. Covers pipeline, datasets, what we've tried, and what's planned.",
    },
    {
        "id":    "VmlldzoxNzE5NTQ2Ng==",
        "title": "Gated pipeline: stage-1 survival gate + uniform VAE sim2sim (LATEST)",
        "desc":  "Clean pipeline (no per-clip weights): survival gate per motion (8/9 LAFAN pass "
                 ">=0.95) -> uniform VAE EX_gated8 (8 clips) -> sim2sim with gated teachers. "
                 "4/8 PASS (walk/run/dance). Confound-free: with confirmed-good teachers the VAE "
                 "still FAILS dynamic/contact motion (sprint/fight/fightAndSports/fallAndGetUp) -> "
                 "the dynamic-motion gap is the VAE, not stage-1. Fix = uniform loss change "
                 "(per-joint norm + velocity loss).",
    },
    {
        "id":    "VmlldzoxNzA2NzE1NQ==",
        "title": "Walk policy (30k) — final metrics + video",
        "desc":  "walk1_subject1 full 30k training. 98.95% survival, E_mpbpe 40.5mm. "
                 "Gallery: lafan_final_30k/gonzc0cq.",
    },
    {
        "id":    "VmlldzoxNzA3NTI1OA==",
        "title": "Walk policy — paper segment reproduction ([0,33]s + [81,87]s)",
        "desc":  "BeyondMimic Table I reproduction. [0,33]s=94.1%/31.9mm, [81,87]s=75%/38.9mm. "
                 "Phase-0 rollout videos in galleries seg0_33_final, seg81_87_final.",
    },
    {
        "id":    "VmlldzoxNzA2ODMxMw==",
        "title": "AMASS → G1 retargeting — verification log",
        "desc":  "AMASS SFU steady-walk 30k: 94.9% survival, E_mpbpe 45.9mm. "
                 "Quality on par with LAFAN1 walk. Pipeline: HF pre-retargeted data + +0.793m fix.",
    },
]

# ── Projects ─────────────────────────────────────────────────────────────────
PROJECTS = [
    {
        "name":  "g1-teachers",
        "desc":  "SAVED per-clip tracking-policy (teacher) artifacts — one `teacher_<clip>` per motion "
                 "(model_*.pt + reloadable agent/env configs + gate_survival metadata). "
                 "Source of truth for teacher coverage; reload for the later VAE/distillation stage.",
    },
    {
        "name":  "beyondmimic-tracking",
        "desc":  "All G1 tracking policy training runs (walk, run, sprint, dance, jumps, …). "
                 "Metrics: Episode_Reward/*, eval/survival_rate, eval/E_mpbpe_mm.",
    },
    {
        "name":  "g1-motion-tokenizer",
        "desc":  "UniMoTok MLD-VAE training on G1 motion exports. "
                 "v1 (22.4M params, 12 clips, epoch 564 best val=3.481), "
                 "v2 (7M params, smallreg config).",
    },
    {
        "name":  "g1-vae-ablation",
        "desc":  "VAE data-scaling ablation: T1 (walk) → T2 (locomotion) → T3 (+dance) → T4 (all). "
                 "Tracks Phase-0 RMSE and Phase-2 sim2sim survival per tier. "
                 "Summary run: ablation_summary.",
    },
]

# ── Build report blocks ───────────────────────────────────────────────────────
def report_link(entity, report_id, title):
    url = f"https://wandb.ai/{entity}/reports/{report_id}"
    return f"[{title}]({url})"


def project_link(entity, name):
    return f"https://wandb.ai/{entity}/{name}"


blocks = [
    wr.H1(text="WBT: Whole-Body Tracking Master Index"),
    wr.MarkdownBlock(text=(
        "Root report for the BeyondMimic whole-body tracking + motion-VAE pipeline. "
        "Links to all sub-reports, projects, and ablation results. "
        "Update any time by re-running `stage2/create_root_report.py`."
    )),

    wr.H2(text="Pipeline Overview"),
    wr.MarkdownBlock(text=(
        "1. Retarget motion data (LAFAN1 CSV / AMASS HF) → G1 joint format via `csv_to_npz.py`\n"
        "2. Upload `motion.npz` to W&B registry (`cs224n-robustqa/wandb-registry-Motions`)\n"
        "3. Train per-clip G1 tracking policy (RSL-RL PPO, 30k iters, 2048 envs, 50Hz)\n"
        "4. Export motion corpus → `stage2/export_g1_motion.py` → 41-D y-up features at 20fps\n"
        "5. Train UniMoTok MLD-VAE (128-frame windows, latent [1,128])\n"
        "6. Phase-0 offline RMSE eval + Phase-2 sim-to-sim closed-loop survival\n"
        "7. Hand VAE checkpoint to OmniMM for diffusion stage (blocked on Yao He's repo)"
    )),

    wr.H2(text="Reports"),
]

for r in REPORTS:
    url = f"https://wandb.ai/{ENTITY}/reports/{r['id']}"
    blocks.append(wr.H3(text=r["title"]))
    blocks.append(wr.MarkdownBlock(text=f"{r['desc']}\n\n[Open report]({url})"))

blocks.append(wr.H2(text="W&B Projects"))
for proj in PROJECTS:
    url = project_link(ENTITY, proj["name"])
    blocks.append(wr.H3(text=proj["name"]))
    blocks.append(wr.MarkdownBlock(text=f"{proj['desc']}\n\n[Open project]({url})"))

blocks += [
    wr.H2(text="VAE Ablation: Data Scaling"),
    wr.MarkdownBlock(text=(
        "Tiers: T1=walk, T2=locomotion, T3=+dance, T4=all categories.\n\n"
        "Key metric: Phase-0 joint angle RMSE (target < 0.10 rad). "
        "Phase-2 sim2sim only for tiers passing Phase-0 gate.\n\n"
        f"[Live ablation results](https://wandb.ai/{ENTITY}/g1-vae-ablation)"
    )),
    wr.H2(text="Teacher Coverage (one RL tracking policy per motion clip)"),
    wr.MarkdownBlock(text=(
        "**134 teachers trained & SAVED to W&B** (`cs224n-robustqa/g1-teachers`, as of 2026-06-18) — "
        "one `teacher_<clip>` artifact per motion (checkpoint + reloadable configs).\n\n"
        "| source | motions | covered |\n"
        "|---|---|---|\n"
        "| **LAFAN1** (G1-retargeted) | 40 / 40 | ✅ complete (all saved) |\n"
        "| **AMASS** (HF pre-retargeted) | 94 / 17,717 | churning ~25/day, continuous queue |\n"
        "| **full LAFAN1 (77) via GMR** | +37 missing | in progress (BVH→G1 retarget) |\n\n"
        "**LAFAN1 gate (original survival ≥0.95):** 23/40 PASS. Fails = all 5 fallAndGetUp (0.51–0.91, "
        "the holdout) + fast run/sprint (0.92–0.94).\n\n"
        "**Convergence (31-clip sample):** median iters-to-gate ≈ easy 1.8k / medium 3.4k / hard 6.0k — "
        "far below the 30k convention. Per-clip ~0.7–2.4 h on a 4090; teachers trained at 12k iters.\n\n"
        "**Pipeline scale:** per-clip teachers don't scale to all AMASS (~2 yr at current rate) — the "
        "queue runs incrementally & is stoppable; full coverage is not the goal, broad saved coverage is."
    )),
    wr.H2(text="Key findings (2026-06)"),
    wr.MarkdownBlock(text=(
        "- **sim2sim slowness was CPU contention**, not PhysX/reset/thrashing — never co-locate eval with training.\n"
        "- **RMSE ≠ trackability** — fallAndGetUp & dance1 have ~equal RMSE (0.26) but decoded survival 0.16 vs 0.87. "
        "Closed-loop survival, not the 0.10-rad RMSE target, is the right VAE gate.\n"
        "- **Confound-free verdict (gated pipeline):** with confirmed-good teachers the uniform VAE passes "
        "locomotion+dance (4/4) but FAILS dynamic/contact motion → the gap is the VAE, not stage-1.\n"
        "- **Fix proven:** up-weighting hard clips moved 3/4 FAILs to PASS; productionize via a UNIFORM "
        "loss change (per-joint normalization + velocity loss), not per-clip weights.\n"
        "- **VAE gen-readiness:** decoder healthy off-manifold; latent not N(0,I) but fixed by shipping "
        "per-dim standardization stats (KL weight can't normalize it)."
    )),
    wr.H2(text="Outstanding Blockers"),
    wr.MarkdownBlock(text=(
        "- VAE fails dynamic/contact motion (sprint/fight/fallAndGetUp) — fix: uniform loss change (in progress)\n"
        "- OmniMM diffusion wiring — blocked on Yao He's repo (handoff = VAE ckpt + normalization + latent-standardization stats)\n"
        "- Blackwell GPUs (0,3) — no PyTorch kernels on driver 595\n"
        "- RTX renderer segfaults — driver 595 vs required ~535; use `--headless` for all training"
    )),
]

# ── Create / upsert report ────────────────────────────────────────────────────
# Update the EXISTING bookmarked root report in place (same URL) rather than spawning a new one.
ROOT_URL = ("https://wandb.ai/cs224n-robustqa/beyondmimic-tracking/reports/"
            "WBT-Master-Index:-all-reports,-projects,-ablation--VmlldzoxNzE0MjQ0MA==")
try:
    report = wr.Report.from_url(ROOT_URL)
    report.blocks = blocks
    report.title = "WBT Master Index: all reports, projects, ablation"
    report.description = "Root report linking all WBT sub-reports, projects, and ablation results."
except Exception as e:
    print(f"[warn] could not load existing root ({e}); creating new")
    report = wr.Report(entity=ENTITY, project=PROJECT,
                       title="WBT Master Index: all reports, projects, ablation",
                       description="Root report linking all WBT sub-reports, projects, and ablation results.",
                       blocks=blocks)
report.save()
print(f"Report saved: {report.url}")
