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
    wr.H2(text="Tracking Policy Status (stage-1, survival gate ≥0.95)"),
    wr.MarkdownBlock(text=(
        "All 9 LAFAN clips have a gate-passing teacher except jumps1 (0.914, retraining).\n\n"
        "**Pass gate (original survival):** walk 1.00, sprint1 1.00, fightAndSports1 1.00, dance2 0.99, "
        "run1 0.98, fight1 0.97, dance1 0.96, fallAndGetUp 0.95.\n\n"
        "**Finding:** stage-1 converges far before 30k (walk ~6k/~2h, fallAndGetUp ~12k); 30k is "
        "overkill. Use the survival gate as the stop criterion, not a fixed iteration budget."
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
