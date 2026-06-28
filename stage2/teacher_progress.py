"""Track per-clip tracking-policy (stage-1 teacher) training progress + convergence.
Parses RSL-RL stdout logs for per-iter 'Mean episode length' (the gate proxy: ep length saturates
at the 500-step cap when the policy tracks without early termination) and 'Mean reward'.
Reports: current iter, target, s/iter (from file mtime span), ETA, and iters-to-90%/95% of the
episode-length asymptote (the convergence/gate estimate). Run:
  .venv/bin/python stage2/teacher_progress.py /tmp/train_faithful_*.log
"""
import sys, os, re, glob, time
import numpy as np


def parse(f):
    it, el, itime = [], [], []
    cur = None; tgt = 0
    for line in open(f, errors="ignore"):
        m = re.search(r"Learning iteration (\d+)/(\d+)", line)
        if m:
            cur = int(m.group(1)); tgt = int(m.group(2))
        m = re.search(r"Mean episode length:\s*([\d.]+)", line)
        if m and cur is not None:
            it.append(cur); el.append(float(m.group(1)))
        m = re.search(r"Iteration time:\s*([\d.]+)s", line)
        if m:
            itime.append(float(m.group(1)))
    return np.array(it), np.array(el), tgt, np.array(itime)


def conv_iter(it, el, frac):
    if len(el) < 10:
        return -1
    asym = np.median(el[int(len(el) * 0.8):])
    thr = frac * asym
    for i in np.where(el >= thr)[0]:
        w = el[i:i + 20]
        if len(w) >= 5 and np.median(w) >= 0.9 * thr:
            return int(it[i])
    idx = np.where(el >= thr)[0]
    return int(it[idx[0]]) if len(idx) else -1


def main():
    files = []
    for a in (sys.argv[1:] or ["/tmp/train_faithful_*.log"]):
        files += glob.glob(a)
    files = sorted(set(files))
    print(f"{'log':34s} {'iter':>12} {'curEL':>6} {'s/it':>5} {'conv90':>7} {'conv95':>7} {'ETA(h)':>7}")
    for f in files:
        it, el, tgt, itime = parse(f)
        if len(it) == 0:
            print(f"{os.path.basename(f):34s}  (no iters yet)"); continue
        cur = it[-1]
        spit = float(np.median(itime[-30:])) if len(itime) else float("nan")
        eta = (tgt - cur) * spit / 3600 if spit == spit else float("nan")
        c90, c95 = conv_iter(it, el, 0.90), conv_iter(it, el, 0.95)
        live = "" if (time.time() - os.path.getmtime(f)) < 180 else " (stale)"
        print(f"{os.path.basename(f):34s} {cur:6d}/{tgt:<5d} {el[-1]:6.1f} {spit:5.2f} {c90:7d} {c95:7d} {eta:7.1f}{live}")


if __name__ == "__main__":
    main()
