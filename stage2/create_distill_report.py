"""Publish the W&B design report for the BFM-Zero -> control-VAE distillation pipeline.

Text/design report (no results yet) describing the plan in stage2/BFMZERO_DISTILL_PLAN.md:
distill BFM-Zero's (backward_map, act) into a Gaussian control VAE (reference -> action),
trained on (state, reference, action, z) pairs under domain randomization, validated sim2sim.

Run in an env with the reports API (wandb 0.27, e.g. .venv). Prints a shareable URL.
"""
import wandb  # noqa: F401
import wandb.apis.reports as wr

ENTITY = "toddler_tracking"
PROJECT = "g1-sim2sim"

md = wr.MarkdownBlock

blocks = [
    wr.H1(text="BFM-Zero → Control-VAE distillation: reference motion → humanoid action"),
    md(text=(
        "**TL;DR.** We are building the faithful BeyondMimic **Fig-7 control VAE**, but with "
        "**BFM-Zero** (the Forward-Backward foundation-model tracker) as the single generalist "
        "teacher instead of per-clip RL policies. The VAE maps **reference motion → action**, "
        "distilled from BFM-Zero, with an `N(0,I)` latent, trained on **(state, reference, action)** "
        "triples collected under **domain randomization**, and validated **sim2sim** in MuJoCo. "
        "This fuses the current two-stage *generate-then-track* pipeline into one latent→action "
        "controller with a diffusion-ready latent. *Design report — experiments are launching now; "
        "results will be appended.*")),

    wr.H2(text="Why distill BFM-Zero (vs the validated generate-then-track pipeline)"),
    md(text=(
        "The current pipeline is two stages: a motion-VAE **reconstructs** a clip, then a "
        "**separate** tracker (BFM-Zero / HoloMotion) executes it. This plan **fuses** them:\n\n"
        "1. **One deployable controller** — `a = D(z, proprio)` replaces decode-then-track.\n"
        "2. **A diffusion-ready latent** — BFM-Zero's `z` lives on a `√d`-sphere (`project_z`), not a "
        "Gaussian; the KL term gives an `N(0,I)` action-latent, which downstream latent-diffusion "
        "needs (raw actions have torque spikes that break diffusion — the VAE smooths them).\n"
        "3. **Inherit BFM-Zero's coverage** — including near-ground crouch/sit/squat that HoloMotion "
        "collapses on — in a compact student.")),

    wr.H2(text="Key insight — BFM-Zero already factorizes as a control VAE"),
    md(text=(
        "Verified in the BFM-Zero source (`humanoidverse/agents/fb/model.py`):\n\n"
        "```python\n"
        "z = model.backward_map(ref_obs)     # reference/goal motion -> latent z   (E)\n"
        "a = model.act(proprio_obs, z)       # z + current state     -> action     (D)\n"
        "```\n\n"
        "So we distill the pair `(backward_map, act)` into a VAE with a **Gaussian bottleneck**. "
        "Interface (dims verified from a live rollout of the pretrained model):\n\n"
        "| symbol | source | dim / semantics |\n"
        "|---|---|---|\n"
        "| `r` reference obs (encoder in) | `get_backward_observation` | **556 / frame** (ref body "
        "pos/rot/vel/ang-vel, local-root) |\n"
        "| `s` proprio obs (decoder in) | `_get_g1env_observation` | **929** (joint pos/vel, base, "
        "history, last action) |\n"
        "| `a*` target action | `model.act(s, z)` | **29-D scaled joint-position targets** (offset "
        "from default pose, `action_scale=0.25` → PD `kp=50`); **not torque** |\n"
        "| `z` teacher latent | `project_z(backward_map(r))` | **256**, on a `√d`-sphere — **not** "
        "`N(0,I)` |\n\n"
        "The action being **position targets** (not torque) means behaviour-cloning is well-behaved "
        "(no spikes) — a good reconstruction target.")),

    wr.H2(text="Model, loss, and the encoder horizon H"),
    md(text=(
        "`MotionVAE` (reuse `stage2/vae_model.py`): encoder `E(ref_window[H]) → (μ,logσ)`, "
        "`z~N(μ,σ)`; decoder `D(z, proprio) → â` (state-conditioned — non-negotiable: under DR/noise "
        "the same reference needs different corrections in different states).\n\n"
        "`L = ‖â − a*‖² + β·KL(q(z)‖N(0,I))` (+ optional `α·‖adapt(z) − z_bfm‖²` to keep the "
        "Gaussian latent semantically tied to BFM-Zero's `z`).\n\n"
        "**Encoder horizon `H`** is a first-class sweepable knob — short for control, long for "
        "generation:\n\n"
        "| `H` | regime | note |\n"
        "|---|---|---|\n"
        "| 1 | per-step causal (Fig-7 exact) | most reactive |\n"
        "| ~8–16 | **short horizon (recommended)** | 16 aligns with the Fig-7 diffusion horizon |\n"
        "| ≥64 | long / whole-clip | good for generation, **too long for control** |\n\n"
        "`H` is a **training-time** parameter: the collector logs the per-step reference, so any `H` "
        "window is a cheap train-time slice (sweep without re-collecting). Do **not** confuse this "
        "with the motion-reconstruction VAE's 128-frame window (that length is right for generation).")),

    wr.H2(text="The hard problem — covariate shift, and the two roles of domain randomization"),
    md(text=(
        "The student **is** the tracker now, so its own errors feed back; naive offline BC drifts to "
        "states BFM-Zero never visited → falls. Escalating fixes:\n\n"
        "1. **DART / noise-injection BC (start here):** step the env with a *noised* action to fatten "
        "the visited-state tube, but **label** each state with the *clean* teacher action `a*`. "
        "Offline; no teacher in the train loop.\n"
        "2. **DAgger (faithful Fig-7):** roll out the *student*, query `BFM.act(s_student, z)` at "
        "student states, aggregate, retrain.\n"
        "3. **On-policy + OU noise + DR** (the full Phase-2 recipe) if needed.\n\n"
        "**DR has two distinct roles:** (a) *robustness* for sim2sim/sim2real (randomize "
        "mass/friction/motor-strength/latency + obs noise — already exposed in the env config); "
        "(b) *covariate coverage* — DR widens the collected-state distribution, a partial substitute "
        "for DAgger. **Subtlety:** BFM-Zero was trained with DR, so match its ranges first, then "
        "widen — a mismatch caps the student.")),

    wr.H2(text="Phased plan"),
    md(text=(
        "| phase | deliverable | status |\n"
        "|---|---|---|\n"
        "| 0 confirm BFM-Zero I/O | action = 29-D pos-targets; z on √d-sphere; dims verified | ✅ done |\n"
        "| 1 data collector | `stage2/collect_bfmzero_pairs.py`: per-step `(s, r, a*, z)` with DR + "
        "DART noise | ✅ built · **collecting now** |\n"
        "| 2 offline VAE-BC | `stage2/train_control_vae.py`: train `MotionVAE`, **sweep `H`**; G1 "
        "recon + G3 active-dims/z-ablation | ✅ built · **sweep launching** |\n"
        "| 3 closed-loop gate | run the distilled VAE **as a policy** in MuJoCo; survival vs BFM-Zero "
        "(**hard gate G2**) | next |\n"
        "| 4 DAgger (if G2 red) | student-rollout + BFM-Zero relabel | conditional |\n"
        "| 5 sim2sim | HV/Isaac-trained VAE → MuJoCo across DR seeds; survival/tracking vs BFM-Zero on "
        "the near-ground suite | next |\n\n"
        "**Experiments launching now:** phase-1 collection on `large_sample` (569 motions, DART=0.1, "
        "DR on) and `lafan_29dof` (40, DART=0.0) → then a phase-2 grid over "
        "`H∈{1,8,16,32} × latent∈{16,32,64} × β∈{1e-3,1e-2,1e-1} × align∈{0,0.5}` on the RTX PRO 6000 "
        "Blackwell GPUs. Reported metric to watch: **G3 z-ablation ≥ 0.1** (latent is used) and G1 "
        "recon RMSE, then the **G2 closed-loop survival** in phase 3.")),

    wr.H2(text="Verification gates (teacher = BFM-Zero)"),
    md(text=(
        "| gate | proves | pass |\n"
        "|---|---|---|\n"
        "| **G1** reconstruction | decoder learned BFM-Zero's map | `‖â − a*‖²` small |\n"
        "| **G2** closed-loop *(hard)* | the VAE *policy* controls the robot | survival ≥ 0.9× "
        "BFM-Zero, E_mpjpe ≤ 1.3× |\n"
        "| **G3** latent structure | latent smooth, used, ≈`N(0,I)` | z-ablation ≥ 0.1, ≥2 active |\n"
        "| **G4** robustness | recovery basin under DR/OU noise | recovers |\n\n"
        "Do not proceed to diffusion until **G2 + G3** pass.")),

    wr.H2(text="Status & links"),
    md(text=(
        "**Design + code:** `stage2/BFMZERO_DISTILL_PLAN.md`, `collect_bfmzero_pairs.py`, "
        "`train_control_vae.py`, `vae_model.py` — on the "
        "[whole_body_tracking repo](https://github.com/yuanzhedong/whole_body_tracking) "
        "(`main`).\n\n"
        "**Sibling pipeline reports:** "
        "[BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim](https://wandb.ai/toddler_tracking/"
        "g1-sim2sim/reports/x--VmlldzoxNzMzOTg2Mg==) · "
        "[HoloMotion-validated pipeline](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/"
        "x--VmlldzoxNzMwNzgxMw==) · "
        "[Tracker comparison](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/"
        "x--VmlldzoxNzMzNDI0MA==).")),
]

report = wr.Report(
    entity=ENTITY, project=PROJECT,
    title="BFM-Zero → Control-VAE distillation: reference motion → humanoid action (plan)",
    description="Design report for the fused latent→action controller distilled from BFM-Zero: "
                "model, (s,r,a*,z) data, DART/DR, encoder horizon H, gates, phased plan.",
    blocks=blocks,
)
report.save()
url = report.url
try:
    share = report.get_share_url()
    if share:
        url = share
except Exception:
    pass
open("/tmp/distill_report_url.txt", "w").write(url or "")
print("REPORT_URL:", report.url)
print("SHARE_URL:", url)
