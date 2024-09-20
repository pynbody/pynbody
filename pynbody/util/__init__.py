"""
Routines used internally by pynbody.

"""
from __future__ import annotations

import fractions
import functools
import gzip
import logging
import math
import pathlib
import sys
import threading
import warnings

import numpy as np

logger = logging.getLogger('pynbody.util')
from ._util import *


def open_(filename: str | pathlib.Path, *args):
    """Open a file, enabling use of gzip decompression

    If the filename ends with .gz, the file is assumed to be gzipped. If the file does not exist, but a file
    with .gz appended does exist, that file is opened instead, on the assumption it is gzipped.

    Other arguments are passed through to the standard ``open`` function."""


    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)

    if not filename.exists():
        filename_with_gz = filename.parent / (filename.name+".gz")
        if filename_with_gz.exists():
            filename = filename_with_gz

    is_gzipped = filename.name.endswith('.gz')

    if is_gzipped:
        return gzip.open(filename, *args)
    else:
        return open(filename, *args)

def cutgz(x: str | pathlib.Path):
    """Strip the .gz ending off a string or path"""
    if isinstance(x, pathlib.Path):
        return x.parent / x.name.removesuffix(".gz")
    elif x[-3:] == '.gz':
        return x[:-3]
    else:
        return x


def arrays_are_same(a1, a2):
    """Returns True if a1 and a2 are numpy views pointing to the exact same underlying data; False otherwise."""
    try:
        return a1.__array_interface__['data'] == a2.__array_interface__['data'] \
            and a1.strides == a2.strides
    except AttributeError:
        return False


def set_array_if_not_same(a_store, a_in, index=None):
    """Checks whether a_store and a_in ultimately point to the same buffer; if not, copy a_in into a_store.

    Optionally an index or slice can be specified to specify a sub-region of a_store to copy into.

    If a_store has units, they are copied from a_in if they are present.

    Parameters
    ----------
    a_store : array-like
        The array to copy into

    a_in : array-like
        The array to copy from

    index : slice | array-like, optional
        The slice or index within a_store to copy into. If None, the target is the whole array of ``a_store``.

    """
    if index is None:
        index = slice(None)
    if not arrays_are_same(a_store[index], a_in):
        a_store[index] = a_in
        if not hasattr(a_in.units, "_no_unit"):
            a_store.units = a_in.units


def equipartition(ar, nbins, vmin=None, vmax=None):
    """Return nbins+1 monotonically increasing bin edges such that the number of items from ar in each bin is ~equal

    Parameters
    ----------
    ar : array-like
        The array to bin

    nbins : int
        The number of bins to create

    vmin : float, optional
        The minimum value to consider

    vmax : float, optional
        The maximum value to consider

    Returns
    -------
    array-like
        The bin edges

    """

    a_s = np.sort(ar)

    if vmax is not None:
        a_s = a_s[a_s <= vmax]
    if vmin is not None:
        a_s = a_s[a_s > vmin]

    return a_s[np.array(np.linspace(0, len(a_s) - 1, nbins + 1), dtype='int')]


def bisect(left, right, f, epsilon=None, eta=0, niter_max=200):
    """Finds the value x such that f(x)=0 for a monotonically increasing function f, using a binary search.

    The search stops when either the bounding domain is smaller than *epsilon* (by default 10^-7 times the original
    region) OR a value f(x) is found such that |f(x)|<eta (by default eta=0, so this criterion is never satisfied).

    Parameters
    ----------
    left : float
        The left-hand boundary of the search region

    right : float
        The right-hand boundary of the search region

    f : function
        The function to find the root of

    epsilon : float, optional
        The tolerance for the search. If not specified, this is set to 10^-7 times the original region.

    eta : float, optional
        The tolerance for the function value. If a value f(x) is found such that |f(x)|<eta, the search stops.

    niter_max : int, optional
        The maximum number of iterations to perform. If the search does not converge after this, a ValueError is raised.

    Returns
    -------
    float
        The value x such that f(x)=0

    """

    if epsilon is None:
        epsilon = (right - left) * 1.e-7

    logger.info("Entering bisection search algorithm")
    for i in range(niter_max):

        if (right - left) < epsilon:
            return (right + left) / 2

        mid = (left + right) / 2
        z = f(mid)

        logger.info(f"{left:f} {mid:f} {right:f} {z:f}")

        if (abs(z) < eta):
            return mid
        elif(z < 0):
            left = mid
        else:
            right = mid

    raise ValueError("Bisection algorithm did not converge")


def _gauss_jordan(matrix):
    """A simple Gauss-Jordan matrix inverter, especially useful for inverting matrices of fractions

    This performs Gauss-Jordan column elimination on a w x 2w matrix, where the first w columns are the matrix to be
    inverted and the second w columns are originally the identity matrix. On return, the first w columns are the
    identity matrix and the second w columns are the inverse of the original matrix.

    Based on public domain code by Jarno Elonen.

    Parameters
    ----------
    matrix : array-like
        The w x 2w matrix prepared for Gauss-Jordan elimination. The matrix is manipulated in place.

    Returns
    -------
    array-like
        The matrix is returned for convenience, but note that it has been manipulated in place.

    """

    h, w = matrix.shape

    assert w > h

    for y in range(0, h):

        maxrow = matrix[y:, y].argmax() + y

        (matrix[y], matrix[maxrow]) = (matrix[maxrow], matrix[y].copy())

        if matrix[y][y] == 0:
            # this will be a problem, see if we can do a row
            # operation to fix it
            for y2 in range(y+1,h):
                if matrix[y2][y]!=0:
                    matrix[y]+=matrix[y2]
                    break

            # no, out of options, must be a singular matrix
            if matrix[y][y]==0:
                raise np.linalg.linalg.LinAlgError("Singular matrix")

        for y2 in range(y + 1, h):    # Eliminate column y
            c = matrix[y2][y] / matrix[y][y]
            matrix[y2] -= matrix[y] * c

    for y in range(h - 1, 0 - 1, -1):  # Backsubstitute
        c = matrix[y][y]
        for y2 in range(0, y):
            for x in range(w - 1, y - 1, -1):
                matrix[y2][x] -= matrix[y][x] * matrix[y2][y] / c
        matrix[y][y] /= c
        for x in range(h, w):       # Normalize row y
            matrix[y][x] /= c

    return matrix


def rational_matrix_inv(matrix):
    """A replacement for numpy linalg matrix inverse which handles fractions exactly.

    Unlike numpy's linalg package, this does not convert matrices to floats before inverting and is therefore
    completely accurate for fractional matrices.

    However, it is only suitable for small matrices as otherwise it's slow!

    Based on public domain code by Jarno Elonen.

    Parameters
    ----------
    matrix : array-like
        The matrix to invert

    Returns
    -------
    array-like
        The inverted matrix
    """

    assert len(matrix) == len(matrix[0])
    x = np.ndarray(
        shape=(len(matrix), len(matrix[0]) + len(matrix)), dtype=fractions.Fraction)
    x[:, :] = fractions.Fraction(0)
    for i in range(len(x)):
        x[i, len(x) + i] = fractions.Fraction(1)

    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j] = fractions.Fraction(matrix[i][j])

    return _gauss_jordan(x)[:, len(x):]


def random_rotation_matrix():
    """Return a random rotation matrix (Haar measure for 3x3 case), using fast algorithm from Graphics Gems III

    (http://tog.acm.org/resources/GraphicsGems/gemsiii/rand_rotation.c)
    """

    x = np.random.uniform(size=3)
    theta = x[0]*2*math.pi
    phi = x[1]*2*math.pi
    z = x[2]*2

    r = math.sqrt(z)
    vx = math.sin(phi)*r
    vy = math.cos(phi)*r
    vz = math.sqrt(2.0-z)

    st = math.sin(theta)
    ct = math.cos(theta)

    sx = vx*ct-vy*st
    sy = vx*st+vy*ct

    return np.array([[vx*sx-ct, vx*sy-st, vx*vz],
                     [vy*sx+st, vy*sy-ct, vy*vz],
                     [vz*sx,vz*sy,1.0-z]])



class ExecutionControl:
    """Class to control execution flow in a with statement.

    For example, one may use this to control whether a block of code should be executed or not, based on some condition
    which is externally controlled at runtime.

    Example:

    .. code-block:: python

        c = ExecutionControl()
        with c:
         if c:
             print("This will be executed")
        if c:
          print("This will not be executed")

    This is used for implementing the various execution control mechanisms in :class:`pynbody.snapshot.simsnap.SimSnap`.

    """

    def __init__(self):
        self.count = 0
        self.on_exit = None

    def __enter__(self):
        self.count += 1

    def __exit__(self, *excp):
        self.count -= 1
        assert self.count >= 0
        if self.count == 0 and self.on_exit is not None:
            self.on_exit()

    def __bool__(self):
        return self.count > 0

    def __repr__(self):
        return "<ExecutionControl: %s>" % ('True' if self.count > 0 else 'False')

class SettingControl:
    """Class to control a setting using a with statement.

    This is used by :mod:`pynbody.analysis.luminosity` and :mod:`pynbody.analysis.ionfrac` to control the table
    used by calculations.

    Given a dictionary, the key to modify, and the value to set the key to, this class will set the key to the value
    on creation or when entering the with block, and reset it to the original value when exiting the block.
    """
    def __init__(self, dictionary, key, value):
        """Create a new setting control object

        Parameters
        ----------

        dictionary : dict
            The dictionary to control; this is modified in place.

        key : str
            The key to modify

        value : object
            The value to set the key to when creating or entering the with block.
        """
        self._dict = dictionary
        self._key = key
        self._value = value
        self._is_set = False
        self.__enter__()

    def __enter__(self):
        if not self._is_set:
            self._old_value = self._get()
            self._set(self._value)
            self._is_set = True

    def __exit__(self, *excp):
        if self._is_set:
            self._set(self._old_value)
            del self._old_value
            self._is_set = False

    def _set(self, value):
        self._dict[self._key] = value

    def _get(self):
        return self._dict[self._key]

#################################################################
# Code for incomplete gamma function accepting complex arguments
#################################################################

def _gammainc_series(a, x, eps=3.e-7, itmax=700):
    """Series representation of the incomplete gamma function, based on numerical recipes 3rd ed"""
    if x == 0.0:
        return 0.0
    ap = a
    sum = 1. / a
    delta = sum
    n = 1
    while n <= itmax:
        ap = ap + 1.
        delta = delta * x / ap
        sum = sum + delta
        if (abs(delta) < abs(sum) * eps):
            return (sum * np.exp(-x + a * np.log(x)))
        n = n + 1
    raise RuntimeError("Maximum iterations exceeded in gser")


def _gammainc_continued_fraction(a, x, eps=3.e-7, itmax=200):
    """Continued fraction representation of the incomplete gamma function, based on numerical recipes 3rd ed"""

    gold = 0.
    a0 = 1.
    a1 = x
    b0 = 0.
    b1 = 1.
    fac = 1.
    n = 1
    while n <= itmax:
        an = n
        ana = an - a
        a0 = (a1 + a0 * ana) * fac
        b0 = (b1 + b0 * ana) * fac
        anf = an * fac
        a1 = x * a0 + anf * a1
        b1 = x * b0 + anf * b1
        if (a1 != 0.):
            fac = 1. / a1
            g = b1 * fac
            if (abs((g - gold) / g) < eps):
                return (g * np.exp(-x + a * np.log(x)))
            gold = g
            n = n + 1
    raise RuntimeError("Maximum iterations exceeded in gcf")


def gamma_inc(a, z, eps=3.e-7):
    """Incomplete gamma function accepting complex z, based on algorithm given in numerical recipes (3rd ed)"""
    import scipy.special

    if (abs(z) < a + 1.):
        return _gammainc_series(a, z, eps)
    else:
        return scipy.special.gamma(a) - _gammainc_continued_fraction(a, z, eps)



def thread_map(func, *args):
    """Run func in separate threads, mapping over the arguments in the same way as map(...)

    There is no thread pool here: a new thread is created for each function call. This is used by the kdtree code.
    """

    def r_func(*afunc):
        try:
            this_t = threading.current_thread()
            this_t.ret_value = func(*afunc)
        except Exception as e:
            this_t.ret_excp = sys.exc_info()

    threads = []
    for arg_this in zip(*args):
        threads.append(threading.Thread(target=r_func, args=arg_this))
        threads[-1].start()
    rets = []
    excp = None
    for t in threads:
        while t.is_alive():
            # just calling t.join() with no timeout can make it harder to
            # debug deadlocks!
            t.join(1.0)
        if hasattr(t, 'ret_excp'):
            _, excp, trace = t.ret_excp
        else:
            rets.append(t.ret_value)

    if excp is None:
        return rets
    else:
        raise excp.with_traceback(trace)  # Note this is a re-raised exception from within a thread


def deprecated(func, message=None):
    """Mark a method or function as deprecated"""
    if isinstance(func, str):
        return functools.partial(deprecated, message=func)

    if message is None:
        message = f"Call to deprecated function {func.__name__}."

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(message, category=DeprecationWarning)
        return func(*args, **kwargs)
    return new_func
