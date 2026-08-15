"""Tests for user-defined pairwise SPH operations.

These cover:

  * :meth:`pynbody.kdtree.KDTree.pair_blocks` -- iterating over neighbour
    pairs in blocks, as flat numpy arrays;
  * :meth:`pynbody.kdtree.KDTree.pair_reduce` -- accumulating a user-supplied
    pairwise function over those pairs;
  * :meth:`pynbody.sph.kernels.KernelBase.value` and
    :meth:`~pynbody.sph.kernels.KernelBase.gradient` -- vectorised kernel
    evaluation, and :attr:`pynbody.kdtree.KDTree.kernel` to reach the kernel
    the tree is actually using.

Together these let a user express operations such as SPH artificial viscosity
and artificial conduction without a python-level loop over ``KDTree.nn()``.

Most tests run against two implementations, supplied by the ``pairs`` fixture,
which maps a snapshot onto an object exposing the pair API:

  ``reference``  :class:`ReferencePairs`, a brute-force O(N^2) implementation
                 defined in this file. It is the specification of both the
                 pair set and the accumulation, and knows nothing of the
                 KDTree.
  ``kdtree``     the snapshot's own ``kdtree``, backed by the C++ tree walk.

:class:`ReferencePairs` mirrors the KDTree methods signature-for-signature, so
applying one test body to both is what checks the C++ implementation against
the specification.

The pair-set tests deliberately avoid the kernel API, using a reference kernel
defined here, and the kernel tests avoid the pair API, so that a failure points
at one thing or the other.
"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


# ---------------------------------------------------------------------------
# Reference implementation: this defines the semantics that the C++
# implementation must reproduce.
# ---------------------------------------------------------------------------

def _minimum_image(dx, boxsize):
    if boxsize is None:
        return dx
    return dx - boxsize * np.round(dx / boxsize)


def _boxsize_of(f):
    """Mirror of SimSnap._get_boxsize_for_kdtree, returning None if infinite."""
    boxsize = f.properties.get('boxsize', None)
    if not boxsize:
        return None
    if pynbody.units.is_unit_like(boxsize):
        return float(boxsize.in_units(f['pos'].units))
    return float(boxsize)


def _scatter_add(out, index, contrib):
    """``out[index] += contrib``, summing over repeated entries of ``index``."""
    npart = out.shape[0]
    if out.ndim == 1:
        out += np.bincount(index, weights=contrib, minlength=npart)
    else:
        for k in range(out.shape[1]):
            out[:, k] += np.bincount(index, weights=contrib[:, k],
                                     minlength=npart)


class ReferencePairs:
    """Brute-force stand-in for the pair methods of :class:`pynbody.kdtree.KDTree`.

    Every candidate pair is enumerated directly, so this depends on nothing but
    the positions and smoothing lengths -- in particular, not on the KDTree.
    The method signatures match KDTree's, so a single test body can exercise
    either implementation.
    """

    def __init__(self, f):
        self._npart = len(f)
        self._pos = np.asarray(f['pos'], dtype=np.float64)
        self._h = np.asarray(f['smooth'], dtype=np.float64)
        self._boxsize = _boxsize_of(f)

    def _all_pairs(self, mode):
        """Enumerate the whole pair set, O(N^2).

        ``symmetric`` gives each unordered pair ``{a, b}`` with
        ``r <= max(2h_a, 2h_b)`` once, canonicalised to ``i < j``; ``gather``
        gives each ordered pair ``(a, b)`` with ``r <= 2h_a``.

        The boundary is inclusive, matching smBallGather. Note ``r == 2h``
        arises by construction, h being half the distance to the nth
        neighbour, so pairs sitting exactly on it are not a rare edge case.
        """
        if mode not in ('symmetric', 'gather'):
            raise ValueError(f"unknown mode {mode!r}")

        i_list, j_list, dx_list = [], [], []

        for a in range(self._npart):
            dx = _minimum_image(self._pos - self._pos[a], self._boxsize)
            r = np.sqrt(np.einsum('kl,kl->k', dx, dx))

            if mode == 'symmetric':
                within = r <= np.maximum(2.0 * self._h[a], 2.0 * self._h)
                within[:a + 1] = False      # each unordered pair once, i < j
            else:
                within = r <= 2.0 * self._h[a]
                within[a] = False           # no self-pairs

            b = np.flatnonzero(within)
            i_list.append(np.full(len(b), a, dtype=np.int64))
            j_list.append(b.astype(np.int64))
            dx_list.append(dx[b])

        return (np.concatenate(i_list), np.concatenate(j_list),
                np.concatenate(dx_list))

    def pair_blocks(self, mode='symmetric', blocksize=1 << 18):
        i_all, j_all, dx_all = self._all_pairs(mode)

        for s in range(0, len(i_all), blocksize):
            sl = slice(s, s + blocksize)
            i, j, dx = i_all[sl], j_all[sl], dx_all[sl]
            r = np.sqrt(np.einsum('kl,kl->k', dx, dx))
            for arr in (i, j, dx, r):
                arr.flags.writeable = False
            yield i, j, dx, r

    def pair_reduce(self, func, mode='symmetric', blocksize=1 << 18,
                    dtype=np.float64):
        out = np.zeros(self._npart, dtype=np.float64)
        trailing = None

        for i, j, dx, r in self.pair_blocks(mode, blocksize):
            result = func(i, j, dx, r)
            contrib_i, contrib_j = (result if isinstance(result, tuple)
                                    else (result, None))
            contrib_i = np.asarray(contrib_i)

            if trailing is None:
                trailing = contrib_i.shape[1:]
                # accumulate in float64, convert on return; see KDTree.pair_reduce
                out = np.zeros((self._npart,) + trailing, dtype=np.float64)

            _scatter_add(out, i, contrib_i)
            if contrib_j is not None:
                _scatter_add(out, j, np.asarray(contrib_j))

        return out.astype(dtype, copy=False)


# Reference cubic spline, matching pynbody's normalisation:
#     W(r, h) = f(r/h) / (pi h^3),  support at r = 2h
# The cross-checks against pynbody's own C++ density and divergence use these
# rather than KernelBase.value/gradient, so that they test the pair set and the
# accumulation without also depending on the kernel API.

def _reference_kernel_value(r, h):
    q = np.asarray(r, dtype=np.float64) / h
    f = np.where(q < 1, 1 - 1.5 * q ** 2 + 0.75 * q ** 3,
                 0.25 * np.clip(2 - q, 0, None) ** 3)
    return f / (np.pi * np.asarray(h, dtype=np.float64) ** 3)


def _reference_kernel_gradient(r, h):
    """dW/dr, negative on (0, 2h)."""
    q = np.asarray(r, dtype=np.float64) / h
    df = np.where(q < 1, -3 * q + 2.25 * q ** 2,
                  -0.75 * np.clip(2 - q, 0, None) ** 2)
    return df / (np.pi * np.asarray(h, dtype=np.float64) ** 4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=['reference', 'kdtree'])
def pairs(request):
    """Map a snapshot onto an object exposing ``pair_blocks``/``pair_reduce``."""
    if request.param == 'reference':
        return ReferencePairs
    return lambda f: f.kdtree


@pytest.fixture
def snap():
    """A small non-periodic snapshot, with smoothing lengths already fixed."""
    npart = 400
    f = pynbody.new(gas=npart)
    np.random.seed(1337)
    f['pos'] = np.random.normal(scale=10.0, size=(npart, 3))
    f['mass'] = np.random.uniform(0.5, 1.5, size=npart)
    f['vel'] = np.random.normal(size=(npart, 3))
    f['u'] = np.random.uniform(1.0, 2.0, size=npart)
    f['smooth']  # force calculation now, so h is constant through the test
    return f


@pytest.fixture
def periodic_snap():
    """A periodic snapshot with plenty of boundary-straddling pairs."""
    npart = 600
    f = pynbody.new(gas=npart)
    np.random.seed(1338)
    f['pos'] = np.random.uniform(-0.5, 0.5, size=(npart, 3))
    f['mass'] = np.random.uniform(0.5, 1.5, size=npart)
    f.properties['boxsize'] = 1.0
    f['smooth']

    # minimum-image is only well defined while kernels are smaller than the box
    assert (2 * np.asarray(f['smooth'])).max() < 0.5
    return f


def _collect(blocks):
    """Concatenate an iterator of pair blocks into single arrays."""
    parts = list(blocks)
    if not parts:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty((0, 3)), np.empty(0))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def _pair_keys(i, j):
    """One integer key per pair, for order-insensitive comparison."""
    return i.astype(np.int64) * (1 << 32) + j.astype(np.int64)


#: Pairs closer than this (relative) to the r == 2h boundary are treated as
#: ambiguous when comparing two implementations. Smoothing lengths are defined
#: as half the distance to the nth neighbour, so r == 2h is hit exactly for one
#: neighbour of every particle. Two independent floating-point evaluations of
#: that same distance -- the C++ tree walk and numpy's einsum -- differ in the
#: last bit, so membership on the boundary is not reproducible between them.
#: It is also physically irrelevant, since W and dW/dr both vanish at r == 2h;
#: :func:`test_boundary_disagreements_carry_no_kernel_weight` asserts exactly
#: that, so nothing is being swept under the carpet here.
BOUNDARY_RTOL = 1e-12


def _boundary_threshold(i, j, h, mode):
    """The value of r at which each pair enters or leaves the pair set."""
    if mode == 'gather':
        return 2.0 * h[i]
    return np.maximum(2.0 * h[i], 2.0 * h[j])


def _away_from_boundary(i, j, r, h, mode, rtol=BOUNDARY_RTOL):
    """Mask selecting pairs whose membership is unambiguous."""
    threshold = _boundary_threshold(i, j, h, mode)
    return np.abs(r - threshold) > rtol * threshold


# ---------------------------------------------------------------------------
# pair_blocks: what the callback receives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_shapes_and_dtypes(pairs, snap, mode):
    seen_any = False
    for i, j, dx, r in pairs(snap).pair_blocks(mode=mode, blocksize=500):
        seen_any = True
        nblock = len(i)
        assert nblock <= 500
        assert i.shape == (nblock,) and j.shape == (nblock,)
        assert dx.shape == (nblock, 3)
        assert r.shape == (nblock,)
        assert np.issubdtype(i.dtype, np.integer)
        assert np.issubdtype(j.dtype, np.integer)
        assert dx.dtype == np.float64 and r.dtype == np.float64
    assert seen_any, "no pairs were generated at all"


@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_r_is_consistent_with_dx(pairs, snap, mode):
    """``r`` must be exactly ``|dx|``, so the callback may use either."""
    _, _, dx, r = _collect(pairs(snap).pair_blocks(mode=mode))
    npt.assert_allclose(r, np.sqrt(np.einsum('kl,kl->k', dx, dx)), rtol=1e-14)


@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_excludes_self_pairs(pairs, snap, mode):
    i, j, _, r = _collect(pairs(snap).pair_blocks(mode=mode))
    assert (i != j).all()
    assert (r > 0).all(), "with self-pairs excluded, r is never zero"


def test_pair_blocks_arrays_are_read_only(pairs, snap):
    """The callback must not be able to corrupt pynbody's internal buffers."""
    for block in pairs(snap).pair_blocks():
        for name, arr in zip(('i', 'j', 'dx', 'r'), block):
            assert not arr.flags.writeable, f"{name} should be read-only"
        break


@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_blocksize_is_respected(pairs, snap, mode):
    sizes = [len(b[0]) for b in
             pairs(snap).pair_blocks(mode=mode, blocksize=137)]
    assert len(sizes) > 1, "test is only meaningful with several blocks"
    assert max(sizes) <= 137


@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_invariant_to_blocksize(pairs, snap, mode):
    """Blocking is an implementation detail; the pair set cannot depend on it."""
    i1, j1, _, r1 = _collect(pairs(snap).pair_blocks(mode=mode,
                                                     blocksize=1 << 20))
    i2, j2, _, r2 = _collect(pairs(snap).pair_blocks(mode=mode, blocksize=97))

    npt.assert_array_equal(np.sort(_pair_keys(i1, j1)),
                           np.sort(_pair_keys(i2, j2)))
    npt.assert_allclose(np.sort(r1), np.sort(r2), rtol=1e-14)


def test_pair_blocks_handles_far_more_neighbours_than_the_default_buffer():
    """A smoothing length from a snapshot may enclose any number of particles.

    The gather buffer starts out sized from the configured neighbour count,
    which says nothing about how many particles an arbitrary h encloses. This
    is a regression test: a FLAMINGO snapshot, whose h comes from SWIFT rather
    than from pynbody's own neighbour search, raised "Buffer overflow while
    gathering neighbour pairs" here. The estimate that the density implies is
    not an upper bound either -- a particle whose h was fixed in a sparse
    region and which now sits beside a dense one encloses far more than
    rho * h^3 suggests.
    """
    npart = 1200
    f = pynbody.new(gas=npart)
    np.random.seed(99)
    f['pos'] = np.random.normal(size=(npart, 3))
    f['mass'] = np.ones(npart)

    # every kernel spans the whole distribution, so each particle has
    # npart - 1 neighbours
    f['smooth'] = np.full(npart, 100.0)
    f.build_tree()
    f.kdtree.set_array_ref('smooth',
                           np.asarray(f['smooth'], dtype=f['pos'].dtype))

    default_buffer = int(pynbody.config['sph']['smooth-particles']) + 500
    assert npart - 1 > default_buffer, "test must exceed the initial buffer"

    i, _, _, _ = _collect(f.kdtree.pair_blocks(mode='gather'))
    npt.assert_array_equal(np.bincount(i, minlength=npart),
                           np.full(npart, npart - 1))

    i, _, _, _ = _collect(f.kdtree.pair_blocks(mode='symmetric'))
    assert len(i) == npart * (npart - 1) // 2


def test_smoothing_operations_handle_more_neighbours_than_the_default_buffer():
    """The same buffer limits sph_divergence, sph_curl, sph_mean and rho.

    These share the gather used by pair_blocks, so they fail the same way once
    the smoothing lengths come from a snapshot rather than from pynbody's own
    search. Regression test: on a FLAMINGO snapshot both sph_divergence and
    sph_curl raised "Buffer overflow in smoothing operation".

    Ordinary pynbody usage does not reach this, because the smoothing lengths
    are recomputed to hold exactly the configured neighbour count; it takes an
    externally supplied h, which pair_blocks now makes routine.
    """
    npart = 1200
    f = pynbody.new(gas=npart)
    np.random.seed(101)
    f['pos'] = np.random.normal(size=(npart, 3))
    f['mass'] = np.ones(npart)
    f['vel'] = np.random.normal(size=(npart, 3))
    f['rho'] = np.ones(npart)
    f['smooth'] = np.full(npart, 100.0)         # encloses the whole snapshot

    assert npart - 1 > int(pynbody.config['sph']['smooth-particles']) + 500

    f.build_tree()
    for name in ('smooth', 'mass', 'rho'):
        f.kdtree.set_array_ref(name, np.asarray(f[name],
                                                dtype=f['pos'].dtype))

    nsmooth = int(pynbody.config['sph']['smooth-particles'])
    div = f.kdtree.sph_divergence(f['vel'], nsmooth)
    curl = f.kdtree.sph_curl(f['vel'], nsmooth)

    assert np.isfinite(div).all() and np.isfinite(curl).all()
    assert div.shape == (npart,) and curl.shape == (npart, 3)


@pytest.mark.parametrize("npart", [200, 200000])
@pytest.mark.parametrize("trailing", [(), (3,)])
def test_scatter_add_agrees_across_both_strategies(npart, trailing):
    """The accumulation must not depend on which strategy is chosen.

    ``_scatter_add`` switches on the ratio of output length to block length:
    ``np.bincount`` allocates a length-N temporary on every call, so it pays
    only while N is comparable to the block, whereas for a large snapshot the
    in-place ``np.add.at`` is faster by two orders of magnitude. Both branches
    are exercised here, against an independent reference.
    """
    from pynbody.kdtree import _scatter_add

    rng = np.random.default_rng(4)
    nblock = 5000
    index = rng.integers(0, npart, nblock)
    contrib = rng.normal(size=(nblock,) + trailing)

    got = np.zeros((npart,) + trailing)
    _scatter_add(got, index, contrib)

    if trailing:
        expected = np.stack(
            [np.bincount(index, weights=contrib[:, k], minlength=npart)
             for k in range(trailing[0])], axis=1)
    else:
        expected = np.bincount(index, weights=contrib, minlength=npart)

    npt.assert_allclose(got, expected, rtol=1e-12)


def test_pair_blocks_is_deterministic(pairs, snap):
    """Same tree and blocksize => identical blocks, so results are reproducible."""
    first = list(pairs(snap).pair_blocks(blocksize=311))
    second = list(pairs(snap).pair_blocks(blocksize=311))

    assert len(first) == len(second)
    for block1, block2 in zip(first, second):
        for arr1, arr2 in zip(block1, block2):
            npt.assert_array_equal(arr1, arr2)


# ---------------------------------------------------------------------------
# pair_blocks: which pairs are in the set
# ---------------------------------------------------------------------------

def _assert_pair_set_matches_reference(f, i, j, r, mode):
    """Compare a pair set against the brute-force reference, off the boundary."""
    h = np.asarray(f['smooth'], dtype=np.float64)
    exp_i, exp_j, _, exp_r = _collect(ReferencePairs(f).pair_blocks(mode=mode))

    got = _away_from_boundary(i, j, r, h, mode)
    expected = _away_from_boundary(exp_i, exp_j, exp_r, h, mode)

    npt.assert_array_equal(np.sort(_pair_keys(i[got], j[got])),
                           np.sort(_pair_keys(exp_i[expected],
                                              exp_j[expected])))
    npt.assert_allclose(np.sort(r[got]), np.sort(exp_r[expected]), rtol=1e-12)

    # Each particle contributes at most one boundary pair, namely its own nth
    # neighbour, which is what defines its smoothing length. Anything much
    # beyond that would mean the disagreement is systematic rather than a
    # last-bit tie.
    assert (~got).sum() <= 2 * len(f), (
        "%d of %d pairs sit on the 2h boundary, with only %d particles; that "
        "is more than the one-per-particle expected from the definition of h"
        % ((~got).sum(), len(i), len(f)))


def test_symmetric_mode_pair_set(pairs, snap):
    """symmetric mode == {a<b : r <= max(2h_a, 2h_b)}, each pair exactly once."""
    i, j, _, r = _collect(pairs(snap).pair_blocks(mode='symmetric'))

    assert (i < j).all(), "symmetric pairs must be canonicalised to i < j"
    keys = _pair_keys(i, j)
    assert len(np.unique(keys)) == len(keys), "pairs must not be duplicated"

    _assert_pair_set_matches_reference(snap, i, j, r, 'symmetric')


def test_gather_mode_pair_set(pairs, snap):
    """gather mode == ordered pairs {(a,b) : r <= 2h_a}."""
    i, j, _, r = _collect(pairs(snap).pair_blocks(mode='gather'))

    h = np.asarray(snap['smooth'], dtype=np.float64)
    assert (r <= 2.0 * h[i] * (1 + 1e-12)).all(), \
        "gather mode must not exceed i's kernel"

    _assert_pair_set_matches_reference(snap, i, j, r, 'gather')


def test_the_two_modes_cover_the_same_unordered_pairs(pairs, snap):
    gi, gj, _, _ = _collect(pairs(snap).pair_blocks(mode='gather'))
    si, sj, _, _ = _collect(pairs(snap).pair_blocks(mode='symmetric'))

    gather_unordered = set(zip(np.minimum(gi, gj).tolist(),
                               np.maximum(gi, gj).tolist()))
    assert gather_unordered == set(zip(si.tolist(), sj.tolist()))


def test_the_two_modes_differ_where_smoothing_lengths_differ(pairs, snap):
    """Check this snapshot can actually tell the two searches apart.

    Where ``h_a << h_b``, the pair is inside b's kernel but not a's. Such pairs
    are reached from one side only in gather mode, and are exactly what a
    symmetric search must find but a gather search would miss. Without any of
    them present, the mode tests above would pass vacuously.
    """
    gi, gj, _, _ = _collect(pairs(snap).pair_blocks(mode='gather'))
    ordered = set(zip(gi.tolist(), gj.tolist()))
    assert sum(1 for (a, b) in ordered if (b, a) not in ordered) > 0, \
        "no singly-counted gather pairs in this snapshot"

    i, j, _, r = _collect(pairs(snap).pair_blocks(mode='symmetric'))
    h = np.asarray(snap['smooth'], dtype=np.float64)
    one_sided = ((r > 2.0 * h[i]) & (r < 2.0 * h[j])) | \
                ((r > 2.0 * h[j]) & (r < 2.0 * h[i]))
    assert one_sided.sum() > 0, \
        "no pairs reachable only from the larger smoothing length"


# ---------------------------------------------------------------------------
# pair_blocks: periodicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_pair_blocks_displacement_is_minimum_image(pairs, periodic_snap, mode):
    """``dx`` must be wrapped; computing ``pos[j]-pos[i]`` by hand is wrong."""
    i, j, dx, r = _collect(pairs(periodic_snap).pair_blocks(mode=mode))
    boxsize = _boxsize_of(periodic_snap)
    pos = np.asarray(periodic_snap['pos'], dtype=np.float64)

    npt.assert_allclose(dx, _minimum_image(pos[j] - pos[i], boxsize),
                        atol=1e-12)
    assert (np.abs(dx) <= boxsize / 2 + 1e-12).all()


def test_pair_blocks_finds_pairs_across_the_periodic_boundary(pairs,
                                                              periodic_snap):
    """Some pairs must have an unwrapped separation of nearly a box length."""
    i, j, _, r = _collect(pairs(periodic_snap).pair_blocks())
    pos = np.asarray(periodic_snap['pos'], dtype=np.float64)
    d = pos[j] - pos[i]
    unwrapped = np.sqrt(np.einsum('kl,kl->k', d, d))

    assert (unwrapped > 2 * r).sum() > 0, (
        "no boundary-straddling pairs, so periodicity is untested here")


# ---------------------------------------------------------------------------
# pair_reduce: accumulation semantics
# ---------------------------------------------------------------------------

def test_pair_reduce_counts_pairs_per_particle(pairs, snap):
    """A callback returning 1.0 counts pairs, i.e. repeated indices sum.

    This is the ``np.bincount`` versus ``out[i] += c`` trap: the latter keeps
    only one contribution per particle per block.
    """
    counts = pairs(snap).pair_reduce(
        lambda i, j, dx, r: (np.ones(len(i)), np.ones(len(j))),
        mode='symmetric', blocksize=97)

    i, j, _, _ = _collect(pairs(snap).pair_blocks(mode='symmetric'))
    expected = (np.bincount(i, minlength=len(snap))
                + np.bincount(j, minlength=len(snap)))

    npt.assert_array_equal(counts, expected)
    assert counts.sum() == 2 * len(i)


def test_pair_reduce_single_return_accumulates_at_i_only(pairs, snap):
    """Returning one array rather than a tuple means 'accumulate at i'."""
    at_i = pairs(snap).pair_reduce(lambda i, j, dx, r: np.ones(len(i)),
                                   mode='gather')
    i, _, _, _ = _collect(pairs(snap).pair_blocks(mode='gather'))
    npt.assert_array_equal(at_i, np.bincount(i, minlength=len(snap)))


def test_pair_reduce_output_shape_scalar_and_vector(pairs, snap):
    m = np.asarray(snap['mass'], dtype=np.float64)

    scalar = pairs(snap).pair_reduce(lambda i, j, dx, r: m[j] / r)
    assert scalar.shape == (len(snap),)

    vector = pairs(snap).pair_reduce(
        lambda i, j, dx, r: (m[j] / r ** 3)[:, None] * dx)
    assert vector.shape == (len(snap), 3)


def test_pair_reduce_output_covers_every_particle(pairs, snap):
    """Output is always length N, whatever the pair distribution."""
    out = pairs(snap).pair_reduce(lambda i, j, dx, r: np.zeros(len(i)))
    assert out.shape == (len(snap),)
    npt.assert_array_equal(out, 0.0)


def test_pair_reduce_invariant_to_blocksize(pairs, snap):
    """A particle's pairs straddle block boundaries; that must not matter."""
    m = np.asarray(snap['mass'], dtype=np.float64)
    u = np.asarray(snap['u'], dtype=np.float64)

    def func(i, j, dx, r):
        c = m[j] * (u[j] - u[i]) / r
        return c, -c

    npt.assert_allclose(pairs(snap).pair_reduce(func, blocksize=1 << 20),
                        pairs(snap).pair_reduce(func, blocksize=53),
                        rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_pair_reduce_dtype_is_respected(pairs, snap, dtype):
    """Any reasonable output dtype works, integers included.

    Integer output is the case that constrains the implementation: bincount
    returns float64, so converting each block as it arrives would truncate the
    partial sums independently and make the answer depend on blocksize. The
    accumulation therefore has to stay in double precision until the end.
    """
    out = pairs(snap).pair_reduce(lambda i, j, dx, r: np.ones(len(i)),
                                  dtype=dtype)
    assert out.dtype == dtype

    i, _, _, _ = _collect(pairs(snap).pair_blocks())
    npt.assert_array_equal(out, np.bincount(i, minlength=len(snap)))


def test_pair_reduce_integer_output_truncates_once_not_per_block(pairs, snap):
    """Integer output must truncate the total, not each block separately.

    Converting each block to the output dtype as it arrived would discard a
    fraction every time, so the shortfall would grow with the number of
    blocks. Accumulating in double precision and converting once bounds the
    total truncation by one unit however the pairs happen to be grouped.

    Exact equality between blocksizes is not achievable here and is not
    claimed: the float64 partial sums themselves differ in the last bit with
    summation order, which can tip a value either side of an integer.
    """
    def fractional(i, j, dx, r):
        return 0.6 * np.ones(len(i))

    exact = pairs(snap).pair_reduce(fractional, blocksize=1 << 20)
    assert exact.max() > 5.0, "test needs several units to truncate away"

    # blocksize 3 gives roughly ten blocks per particle, so per-block
    # truncation would lose several units rather than at most one
    truncated = pairs(snap).pair_reduce(fractional, blocksize=3,
                                        dtype=np.int64)
    assert (np.abs(exact - truncated) < 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# pair_reduce: the conservation property that motivates the design
# ---------------------------------------------------------------------------

def test_antisymmetric_reduction_conserves_exactly(pairs, snap):
    """An antisymmetric callback conserves energy to roundoff, by construction.

    Because each unordered pair is visited once and both ends are accumulated
    from that single visit, ``sum_i m_i du_i/dt == 0`` is an algebraic identity
    rather than something a user has to verify numerically.
    """
    m = np.asarray(snap['mass'], dtype=np.float64)
    u = np.asarray(snap['u'], dtype=np.float64)
    h = np.asarray(snap['smooth'], dtype=np.float64)
    vel = np.asarray(snap['vel'], dtype=np.float64)

    def conduction_like(i, j, dx, r):
        v_d = np.abs(np.einsum('kl,kl->k', vel[j] - vel[i], dx) / r)
        g = 1.0 / h[i] ** 4 + 1.0 / h[j] ** 4
        c = v_d * (u[j] - u[i]) * g
        return m[j] * c, -m[i] * c

    du_dt = pairs(snap).pair_reduce(conduction_like, mode='symmetric')

    terms = m * du_dt
    residual = np.abs(terms.sum()) / np.abs(terms).sum()
    assert residual < 1e-13, f"energy residual {residual:.3e} is too large"


def test_symmetric_reduction_of_positive_definite_term_stays_positive(pairs,
                                                                      snap):
    """The viscous form is positive definite; that must survive the reduction."""
    m = np.asarray(snap['mass'], dtype=np.float64)
    vel = np.asarray(snap['vel'], dtype=np.float64)
    h = np.asarray(snap['smooth'], dtype=np.float64)

    def viscosity_like(i, j, dx, r):
        mu = np.minimum(np.einsum('kl,kl->k', vel[j] - vel[i], dx) / r, 0.0)
        c = mu ** 2 * (1.0 / h[i] ** 4 + 1.0 / h[j] ** 4)
        return m[j] * c, m[i] * c

    assert (pairs(snap).pair_reduce(viscosity_like) >= 0).all()


# ---------------------------------------------------------------------------
# pair_reduce cross-checked against pynbody's existing C++ SPH operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snap_name", ['snap', 'periodic_snap'])
def test_pair_reduce_reproduces_density(pairs, request, snap_name):
    """rho_i = sum_j m_j W(r, h_i), including the self-contribution."""
    f = request.getfixturevalue(snap_name)
    m = np.asarray(f['mass'], dtype=np.float64)
    h = np.asarray(f['smooth'], dtype=np.float64)

    rho = pairs(f).pair_reduce(
        lambda i, j, dx, r: m[j] * _reference_kernel_value(r, h[i]),
        mode='gather')
    rho += m * _reference_kernel_value(np.zeros(len(f)), h)

    npt.assert_allclose(rho, np.asarray(f['rho'], dtype=np.float64), rtol=1e-9)


def test_pair_reduce_reproduces_sph_divergence(pairs, snap):
    """Reproduce KDTree.sph_divergence, which uses the m_j/rho_j gather form.

        div v_i = sum_j (m_j/rho_j) (v_j - v_i) . grad_i W(r, h_i)

    with grad_i W = (dW/dr) * (-dx/r), since dx points from i towards j.
    """
    m = np.asarray(snap['mass'], dtype=np.float64)
    rho = np.asarray(snap['rho'], dtype=np.float64)
    h = np.asarray(snap['smooth'], dtype=np.float64)
    vel = np.asarray(snap['vel'], dtype=np.float64)

    def divergence(i, j, dx, r):
        dv_dot_dx = np.einsum('kl,kl->k', vel[j] - vel[i], dx)
        return -(m[j] / rho[j]) * dv_dot_dx * _reference_kernel_gradient(r, h[i]) / r

    div = pairs(snap).pair_reduce(divergence, mode='gather')
    expected = np.asarray(snap.kdtree.sph_divergence(
        snap['vel'], pynbody.config['sph']['smooth-particles']))

    npt.assert_allclose(div, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# Head-to-head comparison of the two implementations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ['symmetric', 'gather'])
def test_boundary_disagreements_carry_no_kernel_weight(snap, mode):
    """Where the two implementations disagree, the kernel is exactly zero.

    This is what makes the r == 2h ambiguity harmless: such pairs enter every
    SPH sum multiplied by W or dW/dr, both of which vanish at the edge of the
    kernel, so no physical quantity can depend on which side they fall.
    """
    h = np.asarray(snap['smooth'], dtype=np.float64)
    kernel = snap.kdtree.kernel

    reference = _collect(ReferencePairs(snap).pair_blocks(mode=mode))
    kdtree = _collect(snap.kdtree.pair_blocks(mode=mode))

    keys = [_pair_keys(i, j) for i, j, _, _ in (reference, kdtree)]
    disputed = np.setxor1d(*keys)

    if len(disputed) == 0:
        pytest.skip("the two implementations agree exactly on this snapshot")

    for key, (i, j, _, r) in zip(keys, (reference, kdtree)):
        sel = np.isin(key, disputed)
        if not sel.any():
            continue

        npt.assert_allclose(r[sel], _boundary_threshold(i[sel], j[sel], h, mode),
                            rtol=1e-12)

        # both the kernel and its gradient vanish there, for either smoothing
        for h_side in (h[i[sel]], h[j[sel]]):
            outside = r[sel] >= 2 * h_side
            assert (kernel.value(r[sel], h_side)[outside] == 0).all()
            assert (kernel.gradient(r[sel], h_side)[outside] == 0).all()


@pytest.mark.parametrize("snap_name", ['snap', 'periodic_snap'])
def test_matches_reference_reduction(request, snap_name):
    """Realistic kernel-weighted sums agree, boundary ambiguity notwithstanding.

    As in any real SPH sum the kernel gradient multiplies every term, so pairs
    sitting on r == 2h contribute exactly zero and the comparison is exact.
    """
    f = request.getfixturevalue(snap_name)
    m = np.asarray(f['mass'], dtype=np.float64)
    h = np.asarray(f['smooth'], dtype=np.float64)
    kernel = f.kdtree.kernel

    def scalar(i, j, dx, r):
        g = kernel.gradient(r, h[i]) + kernel.gradient(r, h[j])
        c = (h[j] - h[i]) * g
        return m[j] * c, -m[i] * c

    def vector(i, j, dx, r):
        g = kernel.gradient(r, h[i]) + kernel.gradient(r, h[j])
        c = (m[j] * g / r)[:, None] * dx
        return c, -(m[i] / m[j])[:, None] * c

    for func in (scalar, vector):
        npt.assert_allclose(ReferencePairs(f).pair_reduce(func),
                            f.kdtree.pair_reduce(func), rtol=1e-11)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_mode_raises(snap):
    with pytest.raises(ValueError):
        snap.kdtree.pair_blocks(mode='not-a-mode')


def test_invalid_blocksize_raises(snap):
    with pytest.raises(ValueError):
        snap.kdtree.pair_blocks(blocksize=0)


def test_callback_returning_wrong_length_raises(snap):
    with pytest.raises(ValueError):
        snap.kdtree.pair_reduce(lambda i, j, dx, r: np.ones(len(i) + 1))


def test_callback_returning_inconsistent_shape_raises(snap):
    """The trailing shape is fixed by the first block and may not change."""
    state = {'n': 0}

    def wobbly(i, j, dx, r):
        state['n'] += 1
        return np.ones(len(i)) if state['n'] == 1 else np.ones((len(i), 3))

    with pytest.raises(ValueError):
        snap.kdtree.pair_reduce(wobbly, blocksize=53)


# ---------------------------------------------------------------------------
# Vectorised kernel access
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel_name", ['CubicSplineKernel',
                                         'WendlandC2Kernel'])
def test_kernel_value_is_vectorised(kernel_name):
    """``kernel.value(r, h)`` takes arrays and matches the scalar ``get_value``."""
    kernel = pynbody.sph.kernels.create_kernel(kernel_name)
    r = np.linspace(0.0, 3.0, 41)
    h = 1.3

    npt.assert_allclose(kernel.value(r, h),
                        [kernel.get_value(ri / h, h) for ri in r], rtol=1e-12)


def test_kernel_value_accepts_per_pair_smoothing():
    """Both arguments vary per pair, so both must broadcast."""
    kernel = pynbody.sph.kernels.create_kernel('CubicSplineKernel')
    r = np.linspace(0.1, 2.0, 20)
    h = np.linspace(0.8, 1.5, 20)

    npt.assert_allclose(kernel.value(r, h),
                        [kernel.get_value(ri / hi, hi)
                         for ri, hi in zip(r, h)], rtol=1e-12)


@pytest.mark.parametrize("kernel_name", ['CubicSplineKernel',
                                         'WendlandC2Kernel'])
def test_kernel_gradient_matches_finite_difference(kernel_name):
    kernel = pynbody.sph.kernels.create_kernel(kernel_name)
    h = 1.1
    r = np.linspace(0.15, 1.85, 30)  # avoid r=0 and the support boundary
    eps = 1e-6

    numerical = (kernel.value(r + eps, h) - kernel.value(r - eps, h)) / (2 * eps)
    npt.assert_allclose(kernel.gradient(r, h), numerical, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("kernel_name", ['CubicSplineKernel',
                                         'WendlandC2Kernel'])
def test_kernel_is_normalised(kernel_name):
    """4 pi int W(r,h) r^2 dr == 1, which pins down the h convention."""
    kernel = pynbody.sph.kernels.create_kernel(kernel_name)
    h = 1.7
    r = np.linspace(0, 2 * h, 200001)
    integral = 4 * np.pi * _trapezoid(kernel.value(r, h) * r ** 2, r)
    npt.assert_allclose(integral, 1.0, rtol=1e-5)


def test_kernel_vanishes_outside_support():
    kernel = pynbody.sph.kernels.create_kernel('CubicSplineKernel')
    h = 0.9
    r = np.array([2 * h, 2.5 * h, 10 * h])
    npt.assert_array_equal(kernel.value(r, h), np.zeros(3))
    npt.assert_array_equal(kernel.gradient(r, h), np.zeros(3))


def test_kernel_agrees_with_the_reference_cubic_spline():
    """Tie the kernel API to the normalisation assumed by the cross-checks."""
    kernel = pynbody.sph.kernels.create_kernel('CubicSplineKernel')
    r = np.linspace(0.0, 2.5, 60)
    h = 1.15

    npt.assert_allclose(kernel.value(r, h), _reference_kernel_value(r, h),
                        rtol=1e-12)
    npt.assert_allclose(kernel.gradient(r, h),
                        _reference_kernel_gradient(r, h), rtol=1e-12)


@pytest.mark.parametrize("kernel_name", ['CubicSplineKernel',
                                         'WendlandC2Kernel'])
def test_cxx_kernel_gradient_normalisation_matches_its_value(kernel_name):
    """The C++ kernel's gradient must carry the same normalisation as its value.

    smDivQty and smCurlQty multiply ``Kernel::gradient`` by the same
    ``1/(pi h^n)`` factor that smDensity applies to ``Kernel::operator()``, so
    if the two disagree by a constant, every divergence and curl computed with
    that kernel is silently wrong by that factor.

    This is a regression test: ``WendlandC2Kernel::gradient`` in kernels.hpp
    once omitted the 21/16 prefactor that its ``operator()`` includes, which
    made :meth:`~pynbody.kdtree.KDTree.sph_divergence` and
    :meth:`~pynbody.kdtree.KDTree.sph_curl` wrong by 16/21 for that kernel
    while the cubic spline was unaffected -- so no existing test caught it.

    Comparing against the python kernel grounds out in code that does not
    share the bug: :func:`test_kernel_gradient_matches_finite_difference` ties
    ``gradient`` to ``value``, and :func:`test_kernel_value_is_vectorised` and
    :func:`test_kernel_is_normalised` tie ``value`` to the long-standing
    scalar ``get_value`` and to its own normalisation integral.
    """
    npart = 3000
    f = pynbody.new(gas=npart)
    np.random.seed(5)
    f['pos'] = np.random.normal(scale=8.0, size=(npart, 3))
    f['mass'] = np.random.uniform(0.5, 1.5, size=npart)
    f['vel'] = np.random.normal(size=(npart, 3))
    f['rho']

    f.kdtree.set_kernel(kernel_name)
    assert type(f.kdtree.kernel).__name__ == kernel_name

    m = np.asarray(f['mass'], dtype=np.float64)
    rho = np.asarray(f['rho'], dtype=np.float64)
    h = np.asarray(f['smooth'], dtype=np.float64)
    vel = np.asarray(f['vel'], dtype=np.float64)
    kernel = f.kdtree.kernel

    # grad_i W = (dW/dr) (x_i - x_j)/r = -(dW/dr) dx/r
    def divergence(i, j, dx, r):
        dv_dot_dx = np.einsum('kl,kl->k', vel[j] - vel[i], dx)
        return -(m[j] / rho[j]) * dv_dot_dx * kernel.gradient(r, h[i]) / r

    def curl(i, j, dx, r):
        cross = np.cross(vel[j] - vel[i], dx)
        return (m[j] / rho[j] * kernel.gradient(r, h[i]) / r)[:, None] * cross

    nsmooth = pynbody.config['sph']['smooth-particles']

    npt.assert_allclose(f.kdtree.pair_reduce(divergence, mode='gather'),
                        np.asarray(f.kdtree.sph_divergence(f['vel'], nsmooth)),
                        rtol=1e-8)

    npt.assert_allclose(f.kdtree.pair_reduce(curl, mode='gather'),
                        np.asarray(f.kdtree.sph_curl(f['vel'], nsmooth)),
                        rtol=1e-8)


def test_tree_exposes_the_kernel_it_uses(snap):
    """The user must be able to reach the kernel the C++ code is using."""
    snap.build_tree()

    snap.kdtree.set_kernel('WendlandC2Kernel')
    assert isinstance(snap.kdtree.kernel,
                      pynbody.sph.kernels.WendlandC2Kernel)

    snap.kdtree.set_kernel('CubicSplineKernel')
    assert isinstance(snap.kdtree.kernel,
                      pynbody.sph.kernels.CubicSplineKernel)
