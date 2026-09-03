"""Two checks:
1. the split-mask bytes really do cap at one copy of the take array;
2. read amplification for scattered takes -- a pre-existing effect that dominates
   any plan-memory concern.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-pynbody/7111840c-72b2-5c5f-836a-cf1664b7611d/scratchpad")
from plansize import build_plan, plan_bytes, PTYPE
from pynbody.snapshot.gadgethdf import _max_buf

N = 100_000_000
print("check 1: does the split-mask cost cap at one copy of the take array?")
print(f"{'n_files':>8} {'f':>5} {'fresh MB':>10} {'take array MB':>14} {'ratio':>7}")
for nfiles, f in ((4096, 1.0), (4096, 0.5), (512, 1.0), (64, 1.0)):
    take = np.arange(0, N, int(round(1 / f)))
    lc, plan = build_plan(N, nfiles, take)
    st = plan_bytes(lc, plan)
    print(f"{nfiles:8d} {f:5.2f} {st['fresh_array_bytes']/1e6:10.1f} "
          f"{take.nbytes/1e6:14.1f} {st['fresh_array_bytes']/take.nbytes:7.2f}")
    del lc, plan, take

print("\ncheck 2: read amplification -- bytes touched vs bytes kept")
print(f"{'take pattern':30s} {'kept':>12s} {'span read':>12s} {'amplification':>14s}")
rng = np.random.default_rng(1)
NN = 20_000_000
for label, take in (
        ("contiguous 5% block",        np.arange(0, NN // 20)),
        ("random 0.1% of particles",   np.unique(rng.choice(NN, NN // 1000, replace=False))),
        ("random 1% of particles",     np.unique(rng.choice(NN, NN // 100, replace=False))),
        ("every 1000th particle",      np.arange(0, NN, 1000)),
):
    lc, plan = build_plan(NN, 16, take)
    kept = span = 0
    for _, sel, _, _ in plan:
        if isinstance(sel, np.ndarray):
            if len(sel) == 0:
                continue
            kept += len(sel)
            span += int(sel[-1]) - int(sel[0]) + 1
        else:
            kept += sel.stop - sel.start
            span += sel.stop - sel.start
    print(f"{label:30s} {kept:12,d} {span:12,d} {span/max(kept,1):13.1f}x")
    del lc, plan
