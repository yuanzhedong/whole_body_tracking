"""Create/refresh the W&B page tracking the BONES-SEED -> UniMoTok VAE -> HoloMotion sim2sim effort.
Living doc: current pipeline, datasets, what we've tried, what's planned. Re-run to update in place.
"""
import wandb
import wandb.apis.reports as wr

ENTITY = "cs224n-robustqa"; PROJECT = "beyondmimic-tracking"

blocks = [
    wr.H1(text="BONES-SEED → UniMoTok VAE → HoloMotion sim2sim"),
    wr.MarkdownBlock(text=(
        "Living tracking page for the current effort: train the UniMoTok motion-VAE on the **BONES-SEED** "
        "G1 corpus and validate it in closed-loop physics using a **generalist tracker (HoloMotion)** "
        "instead of per-clip teachers. Re-run `stage2/create_bones_seed_report.py` to refresh.\n\n"
        "_Status 2026-06-18: BeyondMimic per-clip teacher training ON HOLD (134 teachers saved to W&B). "
        "Pivoted to BONES-SEED + HoloMotion. BONES-SEED format verified; HoloMotion integration in progress._"
    )),

    wr.H2(text="Current pipeline"),
    wr.MarkdownBlock(text=(
        "1. **Data**: BONES-SEED G1 CSVs (already retargeted to Unitree G1).\n"
        "2. **Convert** → repo features: skip header/Frame col; root pos cm→m; root euler-deg→quat XYZW; "
        "joints deg→rad (block order matches our convention); **resample 120→20 fps**; build 41-D y-up features.\n"
        "3. **VAE**: UniMoTok MLD-VAE, **latent 512 / 5 layers / ff 1024 / KL 5e-5** (= the winning `laA_lat512` "
        "arch), **128-frame windows @20 fps (6.4 s)**.\n"
        "4. **RMSE eval** (`paper_metrics.py`) — kinematic recon, no tracker needed.\n"
        "5. **sim2sim validation via HoloMotion** (MuJoCo): pretrained generalist G1 tracker tracks the "
        "VAE-decoded motion → survival / tracking error. **No per-clip teachers.**"
    )),

    wr.H2(text="Datasets"),
    wr.MarkdownBlock(text=(
        "| dataset | clips | duration | retargeted to G1? | use |\n"
        "|---|---|---|---|---|\n"
        "| **BONES-SEED** (bones-studio/seed) | 142,220 (71k+71k mirror) | ~288 h @120fps | ✅ G1 MuJoCo CSV | **primary VAE corpus** (this effort) |\n"
        "| LAFAN1 (lvhaidong G1) | 40 | ~ | ✅ | prior VAE + 40 teachers |\n"
        "| LAFAN1 FULL via GMR | 77 | 4.6 h | ✅ (we retargeted, corr 0.90) | +37 new motions (obstacles, ground, aiming…) |\n"
        "| AMASS (HF pre-retarg) | 17,717 | ~ | ✅ | 94 teachers trained (queue ON HOLD) |\n\n"
        "**BONES-SEED specifics:** local at `/scratch/user/yzdong/OMG-Data/raw/bones_seed/` (71 GB). "
        "G1 CSV = `Frame + root_translate XYZ(cm) + root_rotate XYZ(euler°) + 29 joints(°)`, block order. "
        "Rich **language captions + temporal segments** (`seed_metadata_*`) — ideal for the OmniMM diffusion stage. "
        "SOMA BVH also available (not needed; G1 provided)."
    )),

    wr.H2(text="Validator: HoloMotion (generalist tracker)"),
    wr.MarkdownBlock(text=(
        "Why a tracker at all: sim2sim is a **closed-loop physics test** — a kinematic motion isn't executable "
        "on a physics robot without a controller; the tracker IS that controller. RMSE ≠ trackability "
        "(fallAndGetUp & dance1 had equal RMSE 0.26 but survival 0.16 vs 0.87).\n\n"
        "**HoloMotion** (HorizonRobotics, Apache-2.0; the tracker OMG uses): reference-conditioned foundation "
        "tracker, v1.3 = 0.4B params / 2000+h, ~300 FPS, **G1 29-DoF**, **MuJoCo sim2sim** "
        "(`eval_mujoco_sim2sim.sh`: ONNX + G1 mjcf + reference npz). Ships **cu128** env → runs on the "
        "**Blackwell RTX PRO 6000** GPUs (the Blackwell block only affected our Isaac torch-2.5.1 venv). "
        "One generalist validates ANY clip → no per-clip teachers ever.\n\n"
        "Integration note: HoloMotion's ref npz needs **FK body poses** (`ref_global_translation/rotation/...`) "
        "+ `ref_dof_pos` — so feeding decoded motion requires a forward-kinematics step + joint-order map."
    )),

    wr.H2(text="What we've TRIED (log)"),
    wr.MarkdownBlock(text=(
        "- **Per-clip teachers**: 134 trained & saved to W&B (`g1-teachers`) — 40 LAFAN1 + 94 AMASS. "
        "Convergence far below 30k (easy ~1.8k, hard ~6k iters). LAFAN1 gate: 23/40 pass survival ≥0.95 "
        "(fails = all 5 fallAndGetUp + fast run/sprint).\n"
        "- **VAE data×capacity**: `laA_lat512` (12.1 h corpus) = best handoff VAE. **Gating to 23 clips HURT** "
        "(data starvation; helps dynamic, loses coverage). **lat1024 WORSE than lat512** (512 is the sweet spot). "
        "Dynamic-motion recon (~0.2–0.35 rad) is the frontier — architectural (1-token over-compression).\n"
        "- **sim2sim methodology**: 12k teachers pass the original-motion gate but are too **brittle for decoded "
        "eval** → need robust 30k teachers, OR a generalist (→ HoloMotion).\n"
        "- **Full LAFAN1 via GMR**: retargeted all 77 BVH→G1 (corr 0.90 vs lvhaidong); +37 new motions.\n"
        "- **OMG (arxiv 2606.10340)**: 1174 h via generalist tracker (HoloMotion) + MuJoCo fall-filter + DiT, "
        "**no per-clip RL** — the scaling blueprint we're now adopting."
    )),

    wr.H2(text="What we PLAN to try"),
    wr.MarkdownBlock(text=(
        "**Phase 0** — BONES-SEED ingest: verify license; download/extract subset (~200 clips, diversity-sampled "
        "by captions); convert G1 CSV → 41-D features @20fps; FK sanity-check the euler convention.\n"
        "**Phase 1** — Train UniMoTok VAE (latent 512) on the subset; RMSE vs LAFAN/AMASS baselines.\n"
        "**Phase 2a** (in progress) — Integrate HoloMotion: cu128 deploy env on Blackwell, pretrained G1 tracker "
        "weights + mjcf, ref-npz builder (FK + joint map), confirm MuJoCo sim2sim runs.\n"
        "**Phase 2b** — Validate-the-validator: HoloMotion tracks ORIGINAL clips at high survival (baseline).\n"
        "**Phase 2c** — Validate VAE: HoloMotion tracks VAE-DECODED motion → survival vs original.\n"
        "**Phase 3** — Analyze; decide whether BONES-SEED becomes the primary VAE corpus (288 h + captions → "
        "OmniMM diffusion handoff).\n\n"
        "**Locked decisions:** 20 fps / 128-frame windows; latent 512; generalist tracker (not per-clip teachers); "
        "Blackwell for the MuJoCo/cu128 stack; AMASS teacher queue paused (resumable)."
    )),

    wr.H2(text="Open questions / risks"),
    wr.MarkdownBlock(text=(
        "- HoloMotion pretrained **G1 ONNX weights** location + license — confirming.\n"
        "- BONES-SEED **euler convention** (XYZ intrinsic vs extrinsic) — validate via FK.\n"
        "- HoloMotion ref npz **FK + joint-order** mapping from our G1 motion.\n"
        "- Validation runs in **HoloMotion's MuJoCo**, not our Isaac `bench_earlyfreeze` (acceptable trade for the generalist)."
    )),
]

URL_FILE = "/tmp/bones_seed_report_url.txt"
try:
    report = wr.Report(entity=ENTITY, project=PROJECT,
                       title="BONES-SEED → UniMoTok VAE → HoloMotion sim2sim",
                       description="Living tracking page: pipeline, datasets, tried & planned.",
                       blocks=blocks)
    report.save()
    open(URL_FILE, "w").write(report.url)
    print(f"Report saved: {report.url}")
except Exception as e:
    print(f"[error] {e}")
