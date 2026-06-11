# Fix: standardize feature ROOT convention (raw-root), clean incremental scaling (2026-06-11)
ISSUE: existing LAFAN features (g1_dataset_T4/gated8) use FK-derived root (csv_to_npz Isaac pass);
the direct AMASS/LAFAN1 converter uses RAW root. JOINTS are EXACT-identical (verified 0.000); only
root6d/velocity differ (frame-dependent, NOT a constant offset -> can't cheaply correct). Doesn't
affect joint-RMSE or sim2sim (both joint-only), but mixing conventions in one corpus is a real bug
(g1_dataset_la mixed FK-LAFAN + raw-AMASS -> KILLED those runs).
FIX: standardize the WHOLE scaling pipeline on RAW-root (direct converter, self-consistent, joint-exact).
- LAFAN1: stage2/out/lafan1_feats (40 raw-root clips, incl all 8 eval clips). full LAFAN1.
- AMASS: stage2/out/amass_feats (raw-root, growing to 10h via fetch_amass_10h.py).
- Build corpora train+val raw-root; recompute norm; sim2sim decodes RAW-root eval clips (joints spliced
  into original npz as before -> convention-independent -> survival comparable to old FK baseline).
INCREMENTAL LADDER (user wants small steps, scale data gradually, download in parallel):
  step2: g1_dataset_lafan1 (40 LAFAN1, raw) x {lat128, lat256} -- does full same-distribution LAFAN help?
  step3: + AMASS (as it downloads) x {lat256, lat512} -- scale capacity with data.
EVAL: run_gated_sim2sim_raw.sh (raw-root clip source + corpus norm). Gated teachers unchanged.
