"""Collate all BFM-Zero distillation experiments into one consolidated report.

Reads whatever result files exist and prints (a) the dual/single-head sweep leaderboard,
(b) the closed-loop G2 survival table, (c) the DAgger survival curve, (d) collection
counts. Safe to run any time (skips missing pieces). Writes a markdown summary.
"""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stage2" / "out"


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def sweep_table(sweep_dir):
    rows = []
    for md in glob.glob(str(sweep_dir / "*" / "metrics.json")):
        m = load_json(md)
        if not m:
            continue
        a = m.get("args", {})
        rows.append(dict(tag=Path(md).parent.name,
                         head="dual" if (a.get("motion_coef") or 0) > 0 else "single",
                         motion=a.get("motion_coef"), H=a.get("horizon"), L=a.get("latent"),
                         a_rmse=round(m.get("val_rec_rmse_raw", 9), 4),
                         z_abl=round(m.get("z_ablation", 0), 4),
                         mo_rmse=round(m.get("motion_rmse", 0), 3),
                         mo_zabl=round(m.get("motion_z_ablation", 0), 3)))
    return rows


def fmt_sweep(rows):
    if not rows:
        return "_(no sweep results yet)_\n"
    rows.sort(key=lambda r: (-max(r["z_abl"], r["mo_zabl"]), r["a_rmse"]))
    out = ["| tag | head | a_rmse | z_abl | mo_rmse | mo_zabl |", "|---|---|---|---|---|---|"]
    for r in rows[:15]:
        out.append(f"| {r['tag']} | {r['head']} | {r['a_rmse']} | {r['z_abl']} | {r['mo_rmse']} | {r['mo_zabl']} |")
    # single vs dual summary
    sh = [r for r in rows if r["head"] == "single"]
    dh = [r for r in rows if r["head"] == "dual"]
    def best_zu(rs): return max((max(r["z_abl"], r["mo_zabl"]) for r in rs), default=0)
    out.append("")
    out.append(f"**single-head:** {len(sh)} runs, best latent-usage {best_zu(sh):.3f} | "
               f"**dual-head:** {len(dh)} runs, best latent-usage {best_zu(dh):.3f}")
    return "\n".join(out) + "\n"


def fmt_g2():
    out = ["| ckpt | n | surv bfm | surv vae_mu | surv vae_zero | jerr bfm | jerr vae_mu |",
           "|---|---|---|---|---|---|---|"]
    any_ = False
    for gd in sorted(glob.glob(str(OUT / "g2_*" / "g2_results.json"))):
        d = load_json(gd)
        if not d:
            continue
        any_ = True
        a = d["agg"]
        name = Path(gd).parent.name.replace("g2_", "")
        out.append(f"| {name} | {a['n']} | {a['survival_bfm']} | {a['survival_vae_mu']} | "
                   f"{a['survival_vae_zero']} | {a['jerr_bfm']} | {a['jerr_vae_mu']} |")
    return "\n".join(out) + "\n" if any_ else "_(no G2 results yet)_\n"


def fmt_dagger():
    d = load_json(OUT / "dagger_H16_seed40" / "dagger_curve.json")
    if not d:
        return "_(no DAgger results yet)_\n"
    out = ["| iter | student survival | buffer clips |", "|---|---|---|"]
    for c in d["curve"]:
        out.append(f"| {c['iter']} | {c['student_survival']} | {c['buffer_clips']} |")
    return "\n".join(out) + "\n"


def fmt_collections():
    out = ["| dataset | pairs |", "|---|---|"]
    for d in sorted(glob.glob(str(OUT / "bfmpairs_*"))):
        out.append(f"| {Path(d).name} | {len(list(Path(d).glob('*.npz')))} |")
    return "\n".join(out) + "\n"


def main():
    md = []
    md.append("# BFM-Zero distillation — consolidated results\n")
    md.append("## ① Dual vs single-head sweep (latent quality)\n")
    md.append(fmt_sweep(sweep_table(OUT / "sweep_dualhead_seed40")))
    md.append("\n## ② DAgger survival curve (closed-loop control fix)\n")
    md.append(fmt_dagger())
    md.append("\n## ③ G2 closed-loop eval (BC vs BFM)\n")
    md.append(fmt_g2())
    md.append("\n## ④ Seed collections\n")
    md.append(fmt_collections())
    text = "\n".join(md)
    (OUT.parent / "CONSOLIDATED_RESULTS.md").write_text(text)
    print(text)
    print(f"\n-> written to stage2/CONSOLIDATED_RESULTS.md")


if __name__ == "__main__":
    main()
