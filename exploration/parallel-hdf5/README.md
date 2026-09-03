# Parallel loading of spanned HDF5 snapshots — feasibility study

Exploratory only. Nothing here is proposed for merge; `parallel_proto.py` is a working
prototype used to size the problem, not a patch.

## Verdict in one paragraph

The refactor pynbody needs is easy and worth doing on its own merits. The hard part is
not pynbody, it is h5py: h5py releases the GIL during reads (measured), but holds its
own interpreter-wide `phil` lock around *every* libhdf5 call, so threads calling h5py
serialise completely — even on different files. Genuine parallelism therefore requires
the worker threads to reach the bytes without going through h5py. For the common case
(contiguous, unfiltered datasets) that is a plain `pread(2)` at the dataset's own byte
offset, which works, scales, and is about 180 lines of code. On Lustre the expected gain
is large and structural; on a laptop it is real but more modest, and for *compressed*
snapshots the bigger laptop win is parallel decompression rather than parallel I/O.

## 1. How much complexity?

### The refactor itself: low

`HDFArrayLoader.load_arrays` (56 lines) walks a stateful cursor across files, driven by
`chunk.LoadControl.iterate_with_interrupts`. The important property is that this
instruction stream is a *pure function of data already computed at construction time*:

* `_file_ptype_slice` and `_file_interrupt_points` — built in `__init_file_map`, which
  already iterates every file;
* `LoadControl`'s chunk maps — built from `take` in `__init_load_map`.

No HDF5 access is needed to generate the instruction stream. So it can be materialised
up front into a plan of `(file_index, source_sel, disk_offset, mem_slice)` tuples, and
the memory destinations are disjoint by construction. Grouping the plan by file gives
one independent task per file, each writing into its own set of rows.

`plan_for_group()` in `parallel_proto.py` is 25 lines and reproduces the existing
generator's output exactly, by reusing `iterate_with_interrupts` itself and simply
recording what it yields instead of acting on it immediately. That is deliberate: the
interrupt/offset bookkeeping is the fiddly part of the current loader (it is where #955
came from), and re-deriving it independently would be asking for trouble.

**Partial loading needs no special handling.** Both `take=` and SWIFT's
`take_region=`/`take_swift_cells=` funnel through `_get_take_parameter` into a single
sorted index array consumed by `LoadControl`; SWIFT additionally prunes files in
`SwiftMultiFileManager._select_files` before the plan is built. The plan inherits all of
it. Verified against the serial loader for: full load, strided `take`, scattered `take`,
`take` straddling a file boundary, `take` starting exactly on a `_max_buf` chunk
boundary, single-particle `take`, and empty `take`.

### Real complications to budget for

| Issue | Cost |
|---|---|
| h5py's `phil` lock — workers cannot use h5py for the bulk transfer | The whole design constraint; see §2 |
| Datasets that are chunked/filtered/compressed | No byte-offset path; must fall back to serial h5py (prototype does this automatically) |
| SWIFT VDS single-file snapshots | One logical dataset spanning many real files, `get_offset()` is `None`. Needs `Dataset.virtual_sources()` resolution to decompose, or falls back to serial |
| Folded n-vectors (1-D dataset holding 3N values) | `_HDFArrayFiller`'s `scaling_factor` logic must be replicated in the fast path. This was the one test that failed in the first prototype pass |
| `_DummyHDFData` (mass from header, `eps` from softening) | No I/O; trivially fine |
| Byte order / dtype mismatch between file and target | Read into a temporary of the file dtype and assign; already handled |
| Exception propagation and file-descriptor cleanup in workers | Ordinary but must be got right |
| Writing `sim_fam_array[mem_slice]` from a worker | Creating `SimArray` views in threads is GIL-safe, but a production version should hand workers a plain `np.asarray` view so no pynbody Python machinery runs concurrently |
| Thread count policy | `config.ini` already has `number_of_threads`; needs its own knob so it can be tuned per filesystem |

### Prototype status

181 significant lines. All 113 tests in `swift_test.py`, `gadgethdf_test.py`,
`subfindhdf_test.py`, `pkdgravhdf_test.py`, `subfindhdf_gadget4_test.py` and
`split_swift_snapshot_test.py` pass with the prototype substituted for
`HDFArrayLoader.load_arrays` (`PYTHONPATH=. pytest -p conftest_patch ...`).

## 2. Does the GIL get in the way?

**No — but h5py's own lock does, completely.**

### h5py releases the GIL (measured, `gilmon.py`)

A monitor thread polling a monotonic clock in a tight Python loop records its largest
gap while another thread reads:

| operation | read time | max GIL stall |
|---|---|---|
| `dset[:]`, uncompressed 64 MB | 394 ms | 1.3 ms |
| `read_direct`, uncompressed 64 MB | 346 ms | 1.7 ms |
| `dset[:]`, gzip 64 MB | 489 ms | 0.8 ms |
| `read_direct`, gzip 64 MB | 449 ms | 4.0 ms |

Stalls stay at the `sys.setswitchinterval()` scale, so the GIL is free for essentially
the whole read — including the decompression filter pipeline.

And pynbody's own Python overhead is negligible: instrumenting `read_direct` shows
**97–99 % of the wall time of a pynbody array load is inside h5py**, with pynbody's
bookkeeping costing $\sim$10 ms per array load regardless of array size.

### But `phil` serialises everything (measured, `scale.py`, `philtest.py`)

h5py wraps every libhdf5 call in an interpreter-wide reentrant lock (`h5py._objects.phil`).
Reading 8 separate 64 MB files, warm cache:

| | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|
| uncompressed, wall | 0.187 s | 0.213 s | 0.199 s | 0.200 s |
| gzip, wall | 4.518 s | 3.920 s | 4.109 s | 4.106 s |
| gzip, CPU | 4.513 s | 3.986 s | 4.155 s | 4.107 s |

CPU $\approx$ wall throughout: only one core is ever busy. On a cold cache, 4 threads
through h5py was *slower* than serial (0.6–0.8×) from pure lock contention. Directly:
a trivial `dset.shape` in one thread blocks for 44 ms while another thread is inside a
`read_direct`.

This is documented behaviour, not a build accident — h5py's own docs say "multiple calls
to the h5py API will not run in parallel - not even if they operate on different
datasets or different files", and that free-threaded builds do *not* disable `phil`
because it protects libhdf5, not the interpreter. The libhdf5 bundled in the PyPI wheel
reports `Threadsafety: OFF`, so `phil` is load-bearing. Even a threadsafe libhdf5 has a
global API lock unless built in HDF5 2.x multi-thread mode, and `phil` would remain
regardless.

### Ways round it, all measured

| Strategy | Universal? | Parallel I/O | Parallel decompress | Measured |
|---|---|---|---|---|
| (a) `pread` at `dset.id.get_offset()` in threads | contiguous + unfiltered + non-virtual only | yes | n/a | 4 threads: CPU 0.14 s vs wall 0.04 s |
| (b) prefetch threads warming the page cache, h5py stays serial | yes | yes | no | comparable to (a) on this box |
| (c) worker processes + shared memory | yes | yes | yes | 4 procs 2.5–3.2× vs serial |
| (d) `read_direct_chunk()` + inflate in threads | chunked + compressed only | no | yes | 3.4× on 4 cores; 0.87 s → 0.35 s |

(a) is what the prototype implements. (d) is worth knowing about: `zlib.decompress`
releases the GIL, so pulling the still-compressed chunks with h5py (serialised, but
that is only the `pread`, 0.112 s of the 0.87 s) and inflating them in a thread pool
recovers most of the parallelism for compressed data without processes.

Only (c) can parallelise the *metadata* phase, because `H5Fopen` is under `phil` too.

## 2b. Alternative HDF5 readers that *are* thread-safe

Anything that links libhdf5 inherits the serialisation, h5py or not. The readers that
scale are the ones that parse the HDF5 container themselves.

8 × 64 MB gzip datasets, 4 threads, warm cache; all four verified byte-identical to h5py
(`altlibs.py`):

| reader | reads via | wall | cores busy | scales |
|---|---|---|---|---|
| h5py 3.16 | libhdf5 + `phil` | 3.37 s | 1.02 | no |
| PyTables 3.11 | libhdf5 1.14.6 | 3.80 s | 1.02 | no |
| pyfive 1.1.2 | pure Python + numpy | 0.82 s | 3.69 | **yes** |
| hidefix 0.12 | Rust chunk index | 0.48 s | 3.77 | **yes** |

"Cores busy" is CPU time over wall time — the number that gives the lock away. pyfive
scales 3.8× from 1 to 4 threads; hidefix is already parallel *inside* one Python thread,
hence its one-thread figure being 5× h5py's.

### pyfive — the interesting one for pynbody

[pyfive](https://github.com/NCAS-CMS/pyfive) is a pure-Python HDF5 *reader* (NCAS-CMS,
numpy-only dependency, [JOSS paper](https://www.theoj.org/joss-papers/joss.09688/10.21105.joss.09688.pdf)).
It parses the container in Python and reads through `mmap`/numpy, so there is no global
lock; the pure-Python metadata walk is GIL-bound but small. On the API surface pynbody's
loader touches it is very nearly a drop-in — `File`, group indexing, `attrs`, `shape`,
`dtype`, and a `read_direct` with h5py's exact signature (verified for whole-dataset and
partial `source_sel` reads).

Substituting it for `gadgethdf._open_hdf_file` (`pyfive_patch.py`) and running pynbody's
HDF5 suites gives 29/65 passing with a two-line shim. The failures are a short list of
concrete gaps rather than scattered incompatibility:

| gap | consequence |
|---|---|
| Read-only, no `write_array` | 7 failures. Structural — pynbody would keep h5py for writing |
| Virtual datasets return `None` (shape/dtype advertised correctly) | SWIFT single-file virtual snapshots fail as a downstream `TypeError`. Guardable, but must be guarded |
| Shuffle filter breaks when `fletcher32` is applied *first* | Blocks SWIFT `take_region=` today — see below |
| `Dataset` has no `__len__` | Accounted for 54 failures before shimming; one line on either side |
| File handles not closed deterministically | `ResourceWarning`s under pynbody's warning filters |

Serial performance is not an obstacle: on gzip it matches h5py to within noise, and on
uncompressed contiguous data it was 3× *faster* here (it memory-maps rather than copying
through HDF5's pipeline).

**The filter-order bug** (reproducer: `pyfive_shuffle_bug.py`). pyfive assumes h5py's
default filter order, shuffle → deflate → fletcher32. SWIFT writes its cell metadata as
fletcher32 → shuffle → deflate, so after inflating, the 4-byte checksum is still attached
and unshuffling a 4100-byte buffer at itemsize 8 computes 513 elements instead of 512:

```
ValueError: attempt to assign bytes of size 512 to extended slice of size 513
```

This hits `Cells/Counts`, which is exactly what `take_region=` reads. Worth reporting
upstream.

### hidefix — fastest, but a bigger bet

[hidefix](https://github.com/gauteh/hidefix) is a Rust reader with Python bindings that
builds a serialisable chunk index (1.2 ms/file here, using libhdf5 once) and then reads
without libhdf5, internally parallel. Quickest thing measured, but self-describes as
experimental, has a thin Python API (`Index(path).dataset(name)[(slice(...),)]`, no
`attrs`), and pulls in `xarray` and `netCDF4`.

### Worth knowing about, wrong shape here

`h5pyd` is genuinely concurrent but talks HTTP to an HSDS server. `kerchunk` /
`VirtualiZarr` pre-scan an HDF5 file into a Zarr-style byte-range index, after which
`zarr` + `numcodecs` read it with full thread parallelism — the same trick as strategy
(a) above, productised and extended to compressed chunks. Worth a look if the byte-offset
route is taken, since it is that idea already debugged.

### What this changes

pyfive is a more attractive execution backend than hand-rolled `pread`: it covers chunked
and compressed datasets too, so it parallelises decompression as well as I/O, and needs no
byte-offset bookkeeping. The plan/execute refactor is worth the same either way — it is
what lets a backend be swapped in per file. The trade is a dependency whose feature
coverage pynbody would have to keep testing, plus keeping h5py for writing and for what
pyfive cannot read.


## 3. Will it help on a well-configured Lustre?

**Yes, and for a structural reason rather than a marginal one.**

Gadget/SWIFT spanned output is a file-per-writer pattern, and such files are normally
created with `stripe_count=1` (which is the right choice for file-per-process I/O —
it avoids OST contention). The consequence is that reading the files one at a time
engages exactly one OST, and one client OSC's RPC pipeline (a small number of RPCs in
flight per OSC, of order 8 by default), at any instant. Sequential loading is therefore
capped near a single OST's per-client rate, typically well below what the client's LNET
interface could absorb. Issuing N per-file reads concurrently engages N OSTs and N OSC
pipelines, and aggregate throughput should scale close to linearly until the client
network — or the servers — saturate. This is the same reason `file-per-process` IOR runs
scale with rank count.

There is a second, separate Lustre-specific cost worth flagging: **metadata latency**.
Per file, `H5Fopen` + `H5Dopen` costs one `openat` plus $\sim$12–15 small synchronous
reads (measured with `strace`), and pynbody opens *every* file during construction
(`__init_file_map`, `__init_family_map`, `__init_loadable_keys`). On Lustre each of those
is a round trip; for a 1000-file snapshot at $\sim$1 ms each that is seconds of pure
latency, all serialised under `phil` and *not* fixable with threads. Two things would
help independently of any parallelism work: (i) processes, per (c) above; (ii) trimming
the redundant probing — a single `pynbody.load` of an 8-file snapshot currently does 6
`h5py.File` opens of file 0 and 13 `h5py.is_hdf5` calls, because each candidate HDF5
subclass re-opens the file in `_can_load`.

Caveats: little to gain if the snapshot's files are already widely striped, or if the
client is already network-saturated; and on a contended filesystem more concurrency can
make things worse. Default the thread count to something modest (4–8) and make it
configurable rather than deriving it from `cpu_count()`.

## 4. Any benefit on consumer hardware?

**Yes, but for different reasons and with a smaller ceiling.**

* **Queue depth.** An NVMe SSD needs many outstanding requests to reach its rated
  throughput; a single synchronous reader issues one at a time, so a drive rated at
  5–7 GB/s typically delivers 1–2 GB/s to one sequential reader. Four to eight
  concurrent per-file readers is exactly what fills the queue. Expect roughly 2–3×,
  not 10×.
* **Parallel decompression is probably the bigger laptop win.** Compressed snapshots
  (SWIFT with `gzip`/`shuffle`, common for archived data) are CPU-bound in inflate at
  a few hundred MB/s per core. Strategy (d) scales that with cores — measured 3.4× on
  4 cores here. A laptop user with a compressed multi-file snapshot gains more from
  that than from I/O concurrency.
* **Little gain when already cached.** Repeat analysis of a snapshot that fits in page
  cache is memcpy-bound and near the memory-bandwidth limit already.
* **Watch memory.** Strategy (b) doubles the page-cache footprint, and thread pools plus
  temporary buffers add anonymous memory. A laptop loading a snapshot comparable to its
  RAM would need the prefetch windowed.

The measurements on this sandbox (a virtio block device, cold page cache, 8-file 0.55 GB
load of `pos`/`vel`/`iord`/`mass`) came out at 2.8 s serial → 0.24 s with 4 threads.
The *direction* is right; the magnitude says more about this VM's poor single-stream
performance than about any real target, and should not be quoted.

## Suggested phasing

1. **Split plan generation from plan execution in `HDFArrayLoader`.** Pure refactor, no
   behaviour change, no threads. Worth doing regardless — it removes the stateful cursor
   that has already caused bugs, and makes everything below possible. Separately, trim
   the redundant `_can_load`/`is_hdf5` probing.
2. **Threaded raw-`pread` execution** for contiguous, unfiltered, non-virtual datasets,
   behind a config switch, with automatic fallback to the current serial path. Best
   Lustre gain per unit of effort — but weigh it against a pyfive backend (§2b), which
   covers compressed data too at the cost of a dependency.
3. **Optionally, threaded inflate** via `read_direct_chunk` for chunked+compressed
   datasets — the main win for compressed data and for laptops. pyfive gets there for
   free if its gaps (§2b) are closed; `read_direct_chunk` keeps it in-house.
4. **Processes + shared memory only if needed** — the one option that also fixes the
   serialised metadata phase, but much the most invasive (it wants `SimSnap._create_array`
   to be able to allocate into a shared buffer, and `fork` around live HDF5 handles needs
   care).

Step 1 is safe to do now. Steps 2–4 should be benchmarked on a real Lustre client with a
real many-file snapshot before committing to them; nothing measurable in this sandbox
predicts those numbers.

## Files

| file | what it does |
|---|---|
| `parallel_proto.py` | the prototype: plan generation + raw-`pread` reader + threaded executor |
| `verify.py` | checks the prototype against the serial loader, incl. partial-loading edge cases |
| `conftest_patch.py` | pytest plugin substituting the prototype, to run the real test suites through it |
| `mksnap.py` | writes a synthetic multi-file Gadget-HDF5 snapshot (optionally gzip+shuffle) |
| `mkplainfiles.py` | writes the plain/gzip HDF5 files the h5py probe scripts below read |
| `bench.py`, `bench2.py` | cold-cache serial vs threaded load timings |
| `gilmon.py` | measures GIL stalls during h5py reads |
| `scale.py`, `scale2.py`, `philtest.py` | thread/process scaling, and the `phil` lock's effect |
| `strategies.py` | side-by-side comparison of strategies (a)–(c) |
| `altlibs.py` | thread scaling of h5py vs pyfive vs hidefix vs PyTables |
| `pyfive_patch.py` | pytest plugin swapping pyfive in for h5py in the GadgetHDF reader |
| `pyfive_shuffle_bug.py` | minimal reproducer for pyfive's filter-order bug |

Reproduce with e.g.

```bash
python mkplainfiles.py                      # inputs for gilmon.py / scale*.py / philtest.py
python mksnap.py ./snap 8 2000000          # or: ... gzip
python verify.py
python bench2.py                            # needs privileges to drop the page cache
PYTHONPATH=. python -m pytest ../../tests/swift_test.py -p conftest_patch
```
