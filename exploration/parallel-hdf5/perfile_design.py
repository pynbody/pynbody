"""Check the per-file formulation reproduces the current mapping exactly.

Claim: for a file covering family-relative disk positions [lo, hi),
    k_lo = searchsorted(family_ids, lo)
gives *both* the memory write offset for that file and the start of the id view --
so files become independent with no walk over their predecessors, and the ids for
each file are a numpy view rather than a copy.
"""
import sys
import numpy as np
from pynbody import chunk
from pynbody.snapshot.gadgethdf import _max_buf

PTYPE = "PartType1"


def mapping_via_current_path(n_disk, per_file, take):
    """Walk iterate_with_interrupts and record global-disk-pos -> memory-index."""
    lc = chunk.LoadControl({PTYPE: slice(0, n_disk)}, _max_buf, take)
    interrupts = list(np.cumsum(per_file))
    n_files = len(per_file)
    state = {"fi": 0, "off": 0, "exhausted": False}

    def select_file(_):
        if state["fi"] + 1 >= n_files:
            state["exhausted"] = True
        else:
            state["fi"] += 1
        state["off"] = 0

    file_start = np.concatenate([[0], np.cumsum(per_file)[:-1]])
    out = {}
    for readlen, sel, mem in lc.iterate_with_interrupts(
            PTYPE, PTYPE, interrupts, select_file):
        if mem is None or state["exhausted"]:
            state["off"] += readlen
            continue
        base = file_start[state["fi"]] + state["off"]
        if isinstance(sel, np.ndarray):
            for j, d in enumerate(sel):
                out[base + int(d)] = mem.start + j
        else:
            for j in range(sel.stop - sel.start):
                out[base + sel.start + j] = mem.start + j
        state["off"] += readlen
    return lc, out


def mapping_via_per_file(lc, n_disk, per_file, take):
    """Independent per-file formulation: one searchsorted per file, no predecessors."""
    file_start = np.concatenate([[0], np.cumsum(per_file)[:-1]])
    out = {}
    n_copies = 0
    if take is None:
        for fi, (lo, n) in enumerate(zip(file_start, per_file)):
            for j in range(n):                      # contiguous: mem == disk
                out[int(lo) + j] = int(lo) + j
        return out, n_copies

    ids = lc._family_ids[PTYPE]                     # sorted, family-relative
    for fi, (lo, n) in enumerate(zip(file_start, per_file)):
        hi = lo + n
        k_lo, k_hi = np.searchsorted(ids, [lo, hi])
        view = ids[k_lo:k_hi]                       # a view, not a copy
        assert view.base is not None or len(view) == 0, "expected a view"
        n_copies += 0
        for j, d in enumerate(view):
            out[int(d)] = k_lo + j                  # k_lo *is* the memory offset
    return out, n_copies


N = 3_000_000
rng = np.random.default_rng(7)
# deliberately uneven files, including an empty one and one shorter than a chunk
per_file = [1_100_000, 0, 300, 900_000, 524_288, 475_412]
assert sum(per_file) == N

cases = [
    ("full load",                     None),
    ("contiguous block mid-file",     np.arange(1_050_000, 1_150_000)),
    ("random 1%",                     np.unique(rng.choice(N, N // 100, replace=False))),
    ("random 60%",                    np.unique(rng.choice(N, int(N * 0.6), replace=False))),
    ("strided by 7",                  np.arange(0, N, 7)),
    ("single particle",               np.array([2_000_001])),
    ("first and last only",           np.array([0, N - 1])),
    ("empty",                         np.array([], dtype=np.int64)),
]
print(f"N_disk = {N:,}, files = {per_file}\n")
print(f"{'take pattern':28s} {'selected':>10s}  agree?")
for label, take in cases:
    lc, ref = mapping_via_current_path(N, per_file, take)
    got, _ = mapping_via_per_file(lc, N, per_file, take)
    agree = (ref == got)
    print(f"{label:28s} {lc.mem_num_particles:10,d}  "
          f"{'yes' if agree else 'NO -- ' + str(len(ref)) + ' vs ' + str(len(got))}")
    if not agree:
        d = {k: (ref.get(k), got.get(k)) for k in set(ref) ^ set(got)}
        print("   symmetric difference sample:", dict(list(d.items())[:5]))
