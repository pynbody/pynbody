"""Undoing HDF5's deflate, shuffle and fletcher32 filters."""

from __future__ import annotations

import zlib

import numpy as np

from .common import _DEFLATE_FILTER, _FLETCHER32_FILTER, _SHUFFLE_FILTER


class _DecodedChunk:
    """A chunk with its filters undone: either its bytes (*array*, a uint8 array), or, if its first filter was a
    shuffle, still shuffled, as *planes*: a (element size, number of elements) uint8 array holding byte b of
    element i at [b, i]."""
    __slots__ = ('array', 'planes')

    def __init__(self, array: np.ndarray | None = None, planes: np.ndarray | None = None):
        self.array = array
        self.planes = planes


def _shuffled_planes(data: np.ndarray, element_size: int, num_elements: int, checksummed: bool) -> np.ndarray:
    """Return the byte planes of *num_elements* shuffled elements of *element_size* bytes, from a chunk's bytes
    after all but its first filter or two are undone: a shuffle, or (if *checksummed*) fletcher32 then a shuffle.

    In the second case the checksum is verified. The shuffle then covered the data with the checksum appended, as a
    whole number of elements (which may include the checksum) followed by any bytes left over, unshuffled."""
    if not checksummed:
        return data.reshape(element_size, num_elements)
    shuffled_elements = len(data) // element_size
    planes = data[:shuffled_elements * element_size].reshape(element_size, shuffled_elements)
    # the elements after the data, unshuffled, followed by the bytes left over: the four bytes of the checksum
    checksum_bytes = np.concatenate((planes[:, num_elements:].T.reshape(-1), data[shuffled_elements * element_size:]))
    data_planes = planes[:, :num_elements]
    _check_fletcher32(_fletcher32_of_planes(data_planes), checksum_bytes)
    return data_planes


def decode_chunk(raw, filter_mask: int, pipeline: list, itemsize: int, nbytes: int | None = None,
                 first_filter: int = 0) -> np.ndarray:
    """Undo the filter pipeline applied to a raw HDF5 chunk, returning its bytes as a uint8 array.

    Parameters
    ----------
    raw : bytes
        The chunk as stored in the file.
    filter_mask : int
        Bit *i* is set if filter *i* of the pipeline was skipped for this chunk.
    pipeline : list of dict
        The filters, in the order they were applied on writing, each with at least a ``filter_id``; shuffle
        filters may also carry their element size as ``client_data[0]``.
    itemsize : int
        The dataset's element size, used by the shuffle filter if its client data does not give one.
    nbytes : int, optional
        The size the chunk should decode to. Decompressing is then done into a buffer of exactly the right size
        (rather than one grown as needed, which briefly takes twice the memory); nothing depends on it being right.
    first_filter : int
        Undo only filters *first_filter* onwards, leaving the ones applied before them on writing.
    """
    data = np.frombuffer(raw, dtype=np.uint8)
    for i in reversed(range(first_filter, len(pipeline))):
        if filter_mask & (1 << i):
            continue
        filter_id = pipeline[i]['filter_id']
        if filter_id == _DEFLATE_FILTER:
            if nbytes is None:
                data = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
            else:
                # the decompressed size is nbytes, plus the checksums of any fletcher32 filters applied before this
                expected = nbytes + 4 * sum(1 for j in range(i) if pipeline[j]['filter_id'] == _FLETCHER32_FILTER
                                            and not filter_mask & (1 << j))
                data = np.frombuffer(zlib.decompress(data, bufsize=max(expected, 1)), dtype=np.uint8)
        elif filter_id == _SHUFFLE_FILTER:
            client_data = pipeline[i].get('client_data') or ()
            data = _unshuffle(data, int(client_data[0]) if len(client_data) > 0 else itemsize)
        elif filter_id == _FLETCHER32_FILTER:
            _verify_fletcher32(data)
            data = data[:-4]
        else:
            raise NotImplementedError(f"HDF5 filter {filter_id} is not supported")
    return data


def _unshuffle(data: np.ndarray, element_size: int) -> np.ndarray:
    """Undo HDF5's shuffle filter.

    As in HDF5, only the largest whole number of elements is shuffled; any bytes left over (for instance, a
    checksum appended by a filter applied before the shuffle) are passed through unchanged."""
    num_elements = len(data) // element_size
    if element_size <= 1 or num_elements <= 1:
        return data
    shuffled_nbytes = num_elements * element_size
    out = np.empty_like(data)
    _unshuffle_planes_into(data[:shuffled_nbytes].reshape(element_size, num_elements), out[:shuffled_nbytes])
    out[shuffled_nbytes:] = data[shuffled_nbytes:]
    return out


def _unshuffle_planes_into(planes: np.ndarray, out: np.ndarray):
    """Interleave byte planes into elements: *planes* has shape (element_size, num_elements), holding byte b of
    element i at [b, i]; *out* is a contiguous uint8 array of element_size * num_elements bytes.

    Copying one plane at a time is two to three times faster in numpy than copying the transpose as a whole."""
    elements = np.reshape(out, (planes.shape[1], planes.shape[0]), copy=False)  # (raises rather than copying)
    for b in range(planes.shape[0]):
        elements[:, b] = planes[b]


def fletcher32(data) -> int:
    """Return the checksum HDF5's fletcher32 filter computes for *data* (bytes or a uint8 array).

    HDF5 reads the data as big-endian 16-bit words (padding an odd final byte with zero) and keeps the two running
    sums below 2**16 by end-around carry, ``x = (x & 0xffff) + (x >> 16)``. That is arithmetic modulo 65535, except
    that a nonzero sum divisible by 65535 is represented as 0xffff rather than 0. Both sums are nonzero unless every
    word is, so each can be found from its exact value modulo 65535. The second sum accumulates the first after every
    word, making it sum_i w_i (n - i) for words w_0 ... w_{n-1}.
    """
    data = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    words = data[:len(data) - len(data) % 2].view('>u2')
    num_words = len(words)
    total, index_weighted = _sum_and_index_weighted_sum(words)
    if len(data) % 2:
        # an odd final byte is padded to a final word, handled here rather than by copying the data
        final_word = int(data[-1]) << 8
        total += final_word
        index_weighted += num_words * final_word
        num_words += 1
    return _fletcher32_from_sums(total, num_words * total - index_weighted)


def _fletcher32_of_planes(planes: np.ndarray) -> int:
    """Return the fletcher32 checksum of the data whose shuffled byte planes are *planes*, without unshuffling.

    Byte b of element i is at position p = i e + b of the unshuffled data, for element size e (which must be even),
    so it falls in word p // 2 = i e / 2 + b // 2 of the W = n e / 2 words, as the high byte if b is even. With
    S_b = sum_i x_bi and T_b = sum_i i x_bi, the first sum is then sum_b c_b S_b and the second (sum_k w_k (W - k))
    is sum_b c_b ((W - b // 2) S_b - (e / 2) T_b), where c_b is 256 for even b and 1 for odd b."""
    element_size, num_elements = planes.shape
    if element_size % 2:
        raise ValueError("Checksums of shuffled data can only be computed for even element sizes")
    num_words = element_size * num_elements // 2
    total, weighted_total = 0, 0
    for b in range(element_size):
        plane_sum, plane_index_weighted = _sum_and_index_weighted_sum(planes[b])
        weight = 256 if b % 2 == 0 else 1
        total += weight * plane_sum
        weighted_total += weight * ((num_words - b // 2) * plane_sum - (element_size // 2) * plane_index_weighted)
    return _fletcher32_from_sums(total, weighted_total)


_column_sum_block_rows = 65536


def _sum_and_index_weighted_sum(values: np.ndarray) -> tuple[int, int]:
    """Return sum_i x_i exactly and sum_i i x_i modulo 65535, for a 1-D array of integers below 2**16.

    To compute sum_i i x_i with vectorised reductions, view the values as a matrix of `width` columns: then it is
    width * sum_r r R_r + sum_c c C_c, where R and C are the row and column sums of the matrix. Every term is reduced
    modulo 65535 before being multiplied, so nothing can overflow int64. The sums themselves accumulate in uint32,
    which numpy does far faster than int64: a row sum is at most width * 65535 < 2**32, and column sums are taken over
    blocks of at most _column_sum_block_rows (65536) rows, so none can overflow."""
    width = 1024
    num_rows = len(values) // width
    total, index_weighted = 0, 0
    if num_rows > 0:
        matrix = values[:num_rows * width].reshape(num_rows, width)
        row_sums = matrix.sum(axis=1, dtype=np.uint32).astype(np.int64)
        column_sums = np.zeros(width, dtype=np.int64)
        for block_start in range(0, num_rows, _column_sum_block_rows):
            column_sums += matrix[block_start:block_start + _column_sum_block_rows].sum(axis=0, dtype=np.uint32)
        total = int(row_sums.sum())
        index_weighted = width * int(np.dot(np.arange(num_rows, dtype=np.int64) % 65535, row_sums % 65535)) \
                         + int(np.dot(np.arange(width, dtype=np.int64), column_sums % 65535))
    remainder = values[num_rows * width:].astype(np.int64)
    total += int(remainder.sum())
    index_weighted += int(np.dot(np.arange(num_rows * width, len(values), dtype=np.int64) % 65535, remainder))
    return total, index_weighted % 65535


def _fletcher32_from_sums(total: int, weighted_total: int) -> int:
    """Return the checksum from the exact sum of the words, and the second sum (sum_k w_k (n - k)) modulo 65535"""
    if total == 0:
        return 0
    sum1 = (total - 1) % 65535 + 1
    sum2 = (weighted_total % 65535 - 1) % 65535 + 1
    return (sum2 << 16) | sum1


def _verify_fletcher32(data: np.ndarray):
    """Check the fletcher32 checksum at the end of a chunk, raising OSError if it does not match."""
    if len(data) < 4:
        raise OSError("Chunk is too short to carry a fletcher32 checksum")
    _check_fletcher32(fletcher32(data[:-4]), data[-4:])


def _check_fletcher32(computed: int, stored_bytes: np.ndarray):
    """Compare a computed checksum with the four bytes stored, raising OSError if they do not match.

    The checksum is stored little-endian. Like HDF5, this also accepts the checksum with the bytes of each 16-bit
    half swapped, as written by some old versions of the library."""
    if len(stored_bytes) != 4:
        raise OSError("Chunk is too short to carry a fletcher32 checksum")
    stored = int(np.ascontiguousarray(stored_bytes).view('<u4')[0])
    reversed_bytes = ((computed & 0x00ff00ff) << 8) | ((computed >> 8) & 0x00ff00ff)
    if stored != computed and stored != reversed_bytes:
        raise OSError("fletcher32 checksum of HDF5 chunk is invalid; the file may be corrupt")
