# Future work: port G1 tracking eval (sim2sim) to mjlab/MJX

**Why:** Isaac eval crawls on untrackable motion due to reset-on-fall overhead (~50x; measured).
MJX reset is ~free (benchmarked: stage2/bench_mjx_reset.py — reset adds <=9% even at 4096 envs;
MJX step ~0.36ms vs Isaac ~40ms). mjlab would make eval fast AND robust, especially while the VAE
still produces failing motion (the regime where Isaac is slowest).

**What "port" means (3 pieces):**
1. G1 robot in MJCF — mostly available (Unitree publishes G1 MJCF; MuJoCo Menagerie has G1).
2. Tracking env — translate the Isaac manager-based env (obs: anchor pos/ori error + proprio;
   reward/termination; motion command w/ RSI + reference playback; joint-target actions) to mjlab's
   (Isaac-Lab-like) API. Translation, not from-scratch.
3. Policy — THE CATCH: current tracking policy was trained in Isaac/PhysX. Running it in MuJoCo adds
   a real physics sim-to-sim gap that would CONFOUND the VAE measurement.

**Two levels:**
- (A) QUICK/confounded: run existing Isaac-trained policy in mjlab eval. Lower effort, but a fall
  could be the physics gap rather than the VAE. Murky.
- (B) CLEAN: retrain the tracking policy in mjlab, then run the VAE eval there. More work; MuJoCo-
  native, no gap, fast resets. The right long-term move if committing to mjlab.

**Decision deferred.** Revisit when: (a) the VAE is good enough that eval-speed matters at scale, or
(b) we want a MuJoCo-native Stage-0/Stage-2 pipeline anyway. Benchmark + env already set up in
.venv_mjx (mujoco 3.9 + mujoco-mjx + jax[cuda12]).

**Env to reproduce the benchmark:** `.venv_mjx/bin/python stage2/bench_mjx_reset.py`
