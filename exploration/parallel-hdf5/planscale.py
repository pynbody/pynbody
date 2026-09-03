"""What in the plan grows, and with what?

Entry count is bounded by the number of chunks (fixed by the on-disk length) plus one
split per file boundary. The bytes are dominated by the *split* masks that
concatenate_indexing allocates fresh at each file boundary -- and those are the only
thing the current generator does not already hold resident.
"""
import sys
import numpy as np
from pynbody import chunk
from pynbody.snapshot.gadgethdf import _max_buf
sys.path.insert(0, "/tmp/claude-0/-home-user-pynbody/7111840c-72b2-5c5f-836a-cf1664b7611d/scratchpad")
from plansize import build_plan, plan_bytes, PTYPE

N = 200_000_000
print(f"N_disk = {N/1e6:.0f}M, take = every 2nd particle (f = 0.5)")
print(f"n_chunks = ceil(N/(max_buf-1)) = {-(-N//(_max_buf-1))}\n")
print(f"{'n_files':>8} {'plan entries':>13} {'fresh arrays':>13} {'fresh MB':>10} "
      f"{'tuples MB':>10} {'already resident MB':>20}")
take = np.arange(0, N, 2)
for nfiles in (8, 64, 512, 4096):
    lc, plan = build_plan(N, nfiles, take)
    st = plan_bytes(lc, plan)
    resident = lc._ids.nbytes + sum(v.nbytes for v in lc._family_ids.values()) + sum(
        m.nbytes for _, m, _ in lc._family_chunks[PTYPE] if isinstance(m, np.ndarray))
    print(f"{nfiles:8d} {len(plan):13,d} {st['n_fresh']:13,d} "
          f"{st['fresh_array_bytes']/1e6:10.1f} {st['tuple_overhead']/1e6:10.3f} "
          f"{resident/1e6:20.1f}")
    del lc, plan
print("\nformula: entries <= n_chunks + n_files;  fresh bytes ~ n_files * f * max_buf * 8")
for nfiles, f, ndisk in ((4096, 0.5, 10_000_000_000), (4096, 0.01, 10_000_000_000),
                         (1024, 1.0, 10_000_000_000), (128, 1.0, 1_000_000_000)):
    est_entries = -(-ndisk // (_max_buf - 1)) + nfiles
    est_bytes = nfiles * f * _max_buf * 8
    print(f"  n_disk={ndisk/1e9:5.1f}e9  n_files={nfiles:5d}  f={f:4.2f}  ->  "
          f"~{est_entries:,} entries, ~{est_bytes/1e9:5.2f} GB of split masks")
