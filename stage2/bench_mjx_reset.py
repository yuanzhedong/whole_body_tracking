"""Confirm the architectural claim: in MJX (what mjlab uses), is RESET ~as cheap as STEP?
Mirrors the Isaac reset benchmark: time (a) batched step, (b) batched step+reset-ALL-envs.
If (b) ~= (a), reset is essentially free -> confirms why mjlab avoids Isaac's reset-on-fall blowup.

  CUDA_VISIBLE_DEVICES=4 .venv_mjx/bin/python -u stage2/bench_mjx_reset.py
"""
import os, time, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx

def log(*a): print(*a, flush=True)

XML = os.path.join(os.path.dirname(mujoco.__file__),
                   "mjx/test_data/humanoid/01_humanoids.xml")
ENV_COUNTS = [128, 1024, 4096]
STEPS = 200

def main():
    log(f"jax {jax.__version__} | device {jax.devices()[0]} | model {os.path.basename(XML)}")
    mj_model = mujoco.MjModel.from_xml_path(XML)
    mjx_model = mjx.put_model(mj_model)
    log(f"humanoid: nq={mj_model.nq} nv={mj_model.nv} nbody={mj_model.nbody}")

    # one reference (init) data on device
    d0 = mjx.make_data(mjx_model)

    log(f"\n=== MJX STEP vs STEP+RESET-ALL (ms/step, {STEPS} steps) ===")
    log(f"{'envs':>6} | {'step only':>10} | {'step+reset-all':>14} | {'reset overhead':>14}")

    for ne in ENV_COUNTS:
        # batch initial data across ne envs
        batch = jax.vmap(lambda _: d0)(jnp.arange(ne))
        init_batch = jax.tree_util.tree_map(lambda x: x, batch)  # keep a copy for reset target

        step_v = jax.vmap(lambda d: mjx.step(mjx_model, d))
        n = STEPS  # static, baked into the jitted scans below

        @jax.jit
        def rollout_step(data):
            def body(d, _):
                return step_v(d), None
            d, _ = jax.lax.scan(body, data, None, length=n)
            return d

        @jax.jit
        def rollout_step_reset(data, init, done):
            # every step: step, then reset envs where done==True via where-SELECT over the
            # full state pytree. `done` is a RUNTIME array (all True here = worst case: every
            # env falls+resets every step) so XLA cannot dead-code-eliminate the step — both
            # the stepped state and the init state are required by the select.
            def sel(stepped, ini):
                m = done.reshape((-1,) + (1,) * (stepped.ndim - 1))
                return jnp.where(m, ini, stepped)
            def body(d, _):
                d2 = step_v(d)
                d2 = jax.tree_util.tree_map(sel, d2, init)
                return d2, None
            d, _ = jax.lax.scan(body, data, None, length=n)
            return d

        done_all = jnp.ones((ne,), dtype=bool)  # runtime array, every env resets every step

        # warmup (compile) — first call discarded
        jax.block_until_ready(rollout_step(batch))
        jax.block_until_ready(rollout_step_reset(batch, init_batch, done_all))

        t0 = time.time()
        jax.block_until_ready(rollout_step(batch))
        ms_step = (time.time() - t0) / STEPS * 1000

        t0 = time.time()
        jax.block_until_ready(rollout_step_reset(batch, init_batch, done_all))
        ms_reset = (time.time() - t0) / STEPS * 1000

        log(f"{ne:>6} | {ms_step:>10.3f} | {ms_reset:>14.3f} | {ms_reset-ms_step:>+14.3f}")

    log("\n=== READ ===")
    log("If 'step+reset-all' ~= 'step only' -> reset is essentially FREE in MJX (vectorized select).")
    log("Contrast Isaac measured earlier: clean step ~40ms; step WITH reset-on-fall ~2000ms (~50x).")
    log("That asymmetry is the root cause of slow Isaac eval on untrackable motion; MJX avoids it.")

if __name__ == "__main__":
    main()
