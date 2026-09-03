"""How big does a materialised load plan get, and how does it depend on the take pattern?

LoadControl needs no files: it works purely from a family-slice dict and the take array,
so plan sizes for arbitrarily large snapshots can be measured directly.
"""
import sys
import numpy as np
from pynbody import chunk
from pynbody.snapshot.gadgethdf import _max_buf

PTYPE = "PartType1"


def build_plan(n_disk, n_files, take):
    """Materialise the plan for one particle type, exactly as parallel_proto does."""
    lc = chunk.LoadControl({PTYPE: slice(0, n_disk)}, _max_buf, take)
    per_file = n_disk // n_files
    interrupts = [min((i + 1) * per_file, n_disk) for i in range(n_files)]
    interrupts[-1] = n_disk

    plan = []
    state = {"file_index": 0, "offset": 0, "exhausted": False}

    def select_file(_):
        if state["file_index"] + 1 >= n_files:
            state["exhausted"] = True
        else:
            state["file_index"] += 1
        state["offset"] = 0

    i0 = 0
    for readlen, buf_index, mem_index in lc.iterate_with_interrupts(
            PTYPE, PTYPE, interrupts, select_file):
        if mem_index is None or state["exhausted"]:
            state["offset"] += readlen
            continue
        i1 = i0 + mem_index.stop - mem_index.start
        plan.append((state["file_index"], buf_index, state["offset"], slice(i0, i1)))
        state["offset"] += readlen
        i0 = i1
    return lc, plan


def plan_bytes(lc, plan):
    """Bytes the plan adds, counting mask arrays only if not already in _family_chunks."""
    preexisting = {id(m) for _, m, _ in lc._family_chunks[PTYPE]
                   if isinstance(m, np.ndarray)}
    tuple_overhead = 0
    fresh_array_bytes = 0
    n_fresh = n_shared = n_slice = 0
    for entry in plan:
        tuple_overhead += sys.getsizeof(entry) + 8      # tuple + list slot
        tuple_overhead += sys.getsizeof(entry[3])       # the mem slice object
        sel = entry[1]
        if isinstance(sel, np.ndarray):
            if id(sel) in preexisting:
                n_shared += 1
            else:
                n_fresh += 1
                fresh_array_bytes += sel.nbytes + sys.getsizeof(np.array([]))
        else:
            n_slice += 1
            tuple_overhead += sys.getsizeof(sel)
    return dict(tuple_overhead=tuple_overhead, fresh_array_bytes=fresh_array_bytes,
                n_fresh=n_fresh, n_shared=n_shared, n_slice=n_slice)


def loadcontrol_bytes(lc, take):
    """What LoadControl already holds, today, before any plan exists."""
    b = 0
    if take is not None:
        b += lc._ids.nbytes
        b += sum(v.nbytes for v in lc._family_ids.values())
    for nread, mask, mem in lc._family_chunks[PTYPE]:
        b += 8 + sys.getsizeof(mem)
        if isinstance(mask, np.ndarray):
            b += mask.nbytes + sys.getsizeof(np.array([]))
        else:
            b += sys.getsizeof(mask)
    return b


N = 200_000_000
NFILES = 64
cases = [
    ("full load (take=None)",            None),
    ("contiguous 1% block",              np.arange(0, N // 100)),
    ("contiguous 50% block",             np.arange(0, N // 2)),
    ("strided, every 100th particle",    np.arange(0, N, 100)),
    ("random 1% of particles",           None),   # filled below
    ("random 10% of particles",          None),
]
rng = np.random.default_rng(0)
cases[4] = ("random 1% of particles",
            np.unique(rng.choice(N, size=N // 100, replace=False)))
cases[5] = ("random 10% of particles",
            np.unique(rng.choice(N, size=N // 10, replace=False)))

print(f"{N/1e6:.0f}M particles on disk, {NFILES} files, _max_buf = {_max_buf}")
print(f"chunks bounded by ceil(N/(max_buf-1)) = {-(-N // (_max_buf - 1))}\n")
print(f"{'take pattern':32s} {'selected':>11s} {'plan':>7s} {'slice':>6s} {'fancy':>6s} "
      f"{'new':>5s} {'plan MB':>8s} {'LC already MB':>14s}")
for label, take in cases:
    lc, plan = build_plan(N, NFILES, take)
    st = plan_bytes(lc, plan)
    nsel = lc.mem_num_particles
    print(f"{label:32s} {nsel:11,d} {len(plan):7,d} {st['n_slice']:6,d} "
          f"{st['n_shared']:6,d} {st['n_fresh']:5,d} "
          f"{(st['tuple_overhead']+st['fresh_array_bytes'])/1e6:8.3f} "
          f"{loadcontrol_bytes(lc, take)/1e6:14.1f}")
    del lc, plan
