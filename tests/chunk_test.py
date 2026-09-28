"""Tests for pynbody.chunk.LoadControl, the partial-loading logic shared by the readers."""

import numpy as np
import numpy.testing as npt
import pytest

from pynbody import family
from pynbody.chunk import LoadControl

MAX_CHUNK = 1000

# A deliberately awkward multi-file layout: an empty file, a file much shorter than a chunk, and a file
# exactly one chunk long.
FILE_LENGTHS = [2200, 0, 300, 1800, MAX_CHUNK, 700]
NUM_PARTICLES = sum(FILE_LENGTHS)


def _take_patterns():
    rng = np.random.default_rng(1234)
    return {
        'full': None,
        'contiguous_mid_file': np.arange(2500, 4000),
        'random_1pc': np.sort(rng.choice(NUM_PARTICLES, NUM_PARTICLES // 100, replace=False)),
        'random_60pc': np.sort(rng.choice(NUM_PARTICLES, (NUM_PARTICLES * 6) // 10, replace=False)),
        'strided': np.arange(0, NUM_PARTICLES, 7),
        'single': np.array([2500]),
        'first_and_last': np.array([0, NUM_PARTICLES - 1]),
        'empty': np.array([], dtype=np.int64),
    }


TAKE_PATTERNS = _take_patterns()


def _disk_positions(buffer_index, readlen):
    """Expand a yielded buffer index into positions relative to the start of the read"""
    if isinstance(buffer_index, slice):
        return np.arange(*buffer_index.indices(readlen))
    else:
        return np.asarray(buffer_index)


def _map_via_interrupts(control, fam, file_lengths):
    """Build a {disk position: memory index} map the way gadgethdf used to, i.e. with an interrupt callback.

    This is the reference implementation against which iterate_within is checked."""

    file_starts = np.concatenate(([0], np.cumsum(file_lengths)))
    interrupt_points = list(np.cumsum(file_lengths))

    state = {'file': 0, 'offset': 0}

    def next_file(disk_position):
        state['file'] += 1
        state['offset'] = 0

    mapping = {}
    for readlen, buffer_index, memory_index in control.iterate_with_interrupts(
            [fam], [fam], interrupt_points, next_file):
        if memory_index is None:
            state['offset'] += readlen
            continue
        positions = file_starts[state['file']] + state['offset'] + _disk_positions(buffer_index, readlen)
        for i, position in enumerate(positions):
            mapping[int(position)] = memory_index.start + i
        state['offset'] += readlen

    return mapping


def _map_via_iterate_within(control, fam, file_lengths):
    """Build a {disk position: memory index} map by treating each file independently"""

    mapping = {}
    lo = 0
    for file_length in file_lengths:
        hi = lo + file_length
        offset = 0
        for readlen, buffer_index, memory_index in control.iterate_within(fam, lo, hi):
            if memory_index is None:
                offset += readlen
                continue
            positions = lo + offset + _disk_positions(buffer_index, readlen)
            for i, position in enumerate(positions):
                mapping[int(position)] = memory_index.start + i
            offset += readlen
        lo = hi

    return mapping


@pytest.mark.parametrize('take_name', list(TAKE_PATTERNS.keys()))
def test_iterate_within_matches_interrupts(take_name):
    """iterate_within must reproduce exactly what the interrupt-callback route produced"""
    take = TAKE_PATTERNS[take_name]
    control = LoadControl({family.dm: slice(0, NUM_PARTICLES)}, MAX_CHUNK, take)

    reference = _map_via_interrupts(control, family.dm, FILE_LENGTHS)
    per_file = _map_via_iterate_within(control, family.dm, FILE_LENGTHS)

    assert per_file == reference

    # ... and that map must in fact be the identity between the selected particles and memory
    expected_ids = np.arange(NUM_PARTICLES) if take is None else np.asarray(take)
    npt.assert_equal(np.array(sorted(per_file.keys())), expected_ids)
    npt.assert_equal(np.array([per_file[int(i)] for i in expected_ids]), np.arange(len(expected_ids)))


@pytest.mark.parametrize('take_name', list(TAKE_PATTERNS.keys()))
def test_chunks_never_exceed_max_chunk(take_name):
    take = TAKE_PATTERNS[take_name]
    control = LoadControl({family.dm: slice(0, NUM_PARTICLES)}, MAX_CHUNK, take)

    for readlen, _, _ in control.iterate([family.dm], [family.dm]):
        assert readlen <= MAX_CHUNK

    lo = 0
    for file_length in FILE_LENGTHS:
        hi = lo + file_length
        for readlen, _, _ in control.iterate_within(family.dm, lo, hi):
            assert readlen <= MAX_CHUNK
        lo = hi


@pytest.mark.parametrize('take_name', list(TAKE_PATTERNS.keys()))
def test_iterate_within_reconstructs_whole_family(take_name):
    """Consecutive windows covering the family must be equivalent to a single pass with iterate"""
    take = TAKE_PATTERNS[take_name]
    control = LoadControl({family.dm: slice(0, NUM_PARTICLES)}, MAX_CHUNK, take)

    disk_position = 0
    from_iterate = {}
    for readlen, buffer_index, memory_index in control.iterate([family.dm], [family.dm]):
        if memory_index is not None:
            positions = disk_position + _disk_positions(buffer_index, readlen)
            for i, position in enumerate(positions):
                from_iterate[int(position)] = memory_index.start + i
        disk_position += readlen

    assert _map_via_iterate_within(control, family.dm, FILE_LENGTHS) == from_iterate


def test_iterate_within_zero_length_range():
    for take in (None, np.array([5, 10, 500])):
        control = LoadControl({family.dm: slice(0, 1000)}, MAX_CHUNK, take)
        assert list(control.iterate_within(family.dm, 300, 300)) == []


def test_iterate_within_rejects_unknown_family():
    control = LoadControl({family.dm: slice(0, 1000)}, MAX_CHUNK, None)
    with pytest.raises(ValueError):
        list(control.iterate_within(family.gas, 0, 100))


@pytest.mark.parametrize('disk_lo, disk_hi', [(500, 400),      # reversed
                                              (-5, 10),        # negative start
                                              (0, 1001),       # past the end of the family
                                              (1001, 1002),    # wholly past the end
                                              (-10, -5)])
@pytest.mark.parametrize('take', [None, np.array([10, 990])])
def test_iterate_within_rejects_windows_outside_the_family(disk_lo, disk_hi, take):
    """Out-of-range windows would otherwise silently produce bad destination slices or over-long reads"""
    control = LoadControl({family.dm: slice(0, 1000)}, MAX_CHUNK, take)
    with pytest.raises(ValueError):
        list(control.iterate_within(family.dm, disk_lo, disk_hi))


def test_iterate_within_accepts_the_whole_family_as_one_window():
    control = LoadControl({family.dm: slice(0, 1000)}, MAX_CHUNK, None)
    assert sum(readlen for readlen, _, _ in control.iterate_within(family.dm, 0, 1000)) == 1000


@pytest.mark.parametrize('take_name', list(TAKE_PATTERNS.keys()))
def test_iterate_within_memory_index_is_family_relative(take_name):
    """Each family's memory index restarts from zero, so callers add mem_family_slice themselves.

    Readers rely on this: gadgethdf keeps one load-control family per HDF particle group and lays
    several of them end-to-end into a single pynbody family array."""
    take = TAKE_PATTERNS[take_name]
    # split the same particles across two families, each spanning several files
    boundary = sum(FILE_LENGTHS[:3])
    family_slice = {family.gas: slice(0, boundary), family.dm: slice(boundary, NUM_PARTICLES)}
    control = LoadControl(family_slice, MAX_CHUNK, take)

    global_mapping = {}
    for fam, file_lengths in ((family.gas, FILE_LENGTHS[:3]), (family.dm, FILE_LENGTHS[3:])):
        disk_start = family_slice[fam].start
        mem_start = control.mem_family_slice[fam].start
        per_family = _map_via_iterate_within(control, fam, file_lengths)

        if per_family:
            assert min(per_family.values()) == 0, "memory index should restart at zero for each family"
        for disk_position, memory_index in per_family.items():
            global_mapping[disk_start + disk_position] = mem_start + memory_index

    expected_ids = np.arange(NUM_PARTICLES) if take is None else np.asarray(take)
    npt.assert_equal(np.array(sorted(global_mapping.keys())), expected_ids)
    npt.assert_equal(np.array([global_mapping[int(i)] for i in expected_ids]),
                     np.arange(len(expected_ids)))


def test_multifamily_slices():
    family_slice = {family.gas: slice(0, 100), family.dm: slice(100, 350), family.star: slice(350, 400)}

    control = LoadControl(family_slice, MAX_CHUNK, None)
    assert control.disk_num_particles == 400
    assert control.mem_num_particles == 400
    assert control.mem_family_slice == family_slice

    # 10 gas, 5 dm, 2 star
    take = np.concatenate((np.arange(10), np.arange(200, 205), np.array([350, 399])))
    control = LoadControl(family_slice, MAX_CHUNK, take)
    assert control.disk_num_particles == 400
    assert control.mem_num_particles == 17
    assert control.mem_family_slice == {family.gas: slice(0, 10), family.dm: slice(10, 15),
                                        family.star: slice(15, 17)}


@pytest.mark.parametrize('take, expected', [(slice(10), np.arange(10)),
                                            (slice(-100, None), np.arange(300, 400)),
                                            (slice(50, 400, 25), np.arange(50, 400, 25)),
                                            (slice(1000), np.arange(400))])
def test_slice_take_is_normalised_against_disk_length(take, expected):
    control = LoadControl({family.gas: slice(0, 100), family.dm: slice(100, 400)}, MAX_CHUNK, take)
    assert control.mem_num_particles == len(expected)

    mem_slice = control.mem_family_slice[family.dm]
    npt.assert_equal(np.sum(expected >= 100), mem_slice.stop - mem_slice.start)


def test_negative_step_slice_take_rejected():
    with pytest.raises(ValueError, match="negative step"):
        LoadControl({family.dm: slice(0, 400)}, MAX_CHUNK, slice(None, None, -1))


@pytest.mark.parametrize('take', [np.array([5, 3, 10]), np.array([1, 1, 2])])
def test_non_ascending_take_rejected(take):
    with pytest.raises(ValueError, match="strictly ascending"):
        LoadControl({family.dm: slice(0, 400)}, MAX_CHUNK, take)
