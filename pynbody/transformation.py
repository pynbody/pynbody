import numpy as np
import weakref
from . import snapshot


class Transformation(object):

    def __init__(self, f, defer=False):
        if isinstance(f, snapshot.SimSnap):
            self.sim = f
            self.next_transformation = None
        elif isinstance(f, Transformation):
            self.sim = None
            self.next_transformation = f
        else:
            raise ValueError("Transformation must either act on another Transformation or on a SimSnap")

        self.applied = False
        if not defer:
            self.apply(force=False)

    @property
    def sim(self):
        return self._sim()

    @sim.setter
    def sim(self, sim):
        if sim is None:
            self._sim = lambda: None
        else:
            self._sim = weakref.ref(sim)

    def apply_to(self, f):
        if self.next_transformation is not None:
            # this is a chained transformation, get the SimSnap to operate on
            # from the level below
            f = self.next_transformation.apply_to(f)

        self._apply(f)

        return f

    def apply_inverse_to(self, f):
        self._revert(f)

        if self.next_transformation is not None:
            self.next_transformation.apply_inverse_to(f)



    def apply(self, force=True):

        if self.next_transformation is not None:
            # this is a chained transformation, get the SimSnap to operate on
            # from the level below
            f = self.next_transformation.apply(force=force)
        else:
            f = self.sim

        if self.applied and force:
            raise RuntimeError("Transformation has already been applied")

        if not self.applied:
            self._apply(f)
            self.applied = True
            self.sim = f

        return f

    def revert(self):
        if not self.applied:
            raise RuntimeError("Transformation has not been applied")
        self._revert(self.sim)
        self.applied = False
        if self.next_transformation is not None:
            self.next_transformation.revert()

    def _apply(self, f):
        pass

    def _revert(self, f):
        pass

    def __enter__(self):
        self.apply(force=False)

    def __exit__(self, *args):
        self.revert()


class GenericTranslation(Transformation):

    def __init__(self, f, arname, shift):
        self.shift = shift
        self.arname = arname
        super(GenericTranslation, self).__init__(f)

    def _apply(self, f):
        f[self.arname] += self.shift

    def _revert(self, f):
        f[self.arname] -= self.shift


class GenericRotation(Transformation):

    def __init__(self, f, matrix, ortho_tol=1.e-8):
        # Check that the matrix is orthogonal
        resid = np.dot(matrix, np.asarray(matrix).T) - np.eye(3)
        resid = (resid ** 2).sum()
        if resid > ortho_tol or resid != resid:
            raise ValueError("Transformation matrix is not orthogonal")
        self.matrix = matrix
        super(GenericRotation, self).__init__(f)

    def _apply(self, f):
        f._transform(self.matrix)

    def _revert(self, f):
        f._transform(self.matrix.T)


def translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    spatial *shift*.

    This allows you to enclose a code block within which the simulation is offset
    by the specified amount. On exiting the code block, you are guaranteed the
    simulation is put back to where it started, so

    with translate(f, shift) :
        print f['pos'][0]

    is equivalent to

    try:
        f['pos']+=shift
        print f['pos'][0]
    finally:
        f['pos']-=shift

    On the other hand,

    translate(f, shift)
    print f['pos'][0]

    Performs the translation but does not revert it at any point.
    """
    return GenericTranslation(f, 'pos', shift)


def inverse_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the spatial
    vector *-shift*.

    For a fuller description, see *translate*"""
    return translate(f, -np.asarray(shift))


def v_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    velocity *shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return GenericTranslation(f, 'vel', shift)


def inverse_v_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    velocity *-shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return GenericTranslation(f, 'vel', -np.asarray(shift))


def xv_translate(f, x_shift, v_shift):
    """Form a context manager for translating the simulation *f* by the given
    position *x_shift* and velocity *v_shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return translate(v_translate(f, v_shift),
                     x_shift)


def inverse_xv_translate(f, x_shift, v_shift):
    """Form a context manager for translating the simulation *f* by the given
    position *-x_shift* and velocity *-v_shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return translate(v_translate(f, -np.asarray(v_shift)),
                     -np.asarray(x_shift))


def transform(f, matrix):
    """Form a context manager for rotating the simulation *f* by the given 3x3
    *matrix*"""

    return GenericRotation(f, matrix)


def null(f):
    """Form a context manager for the null transformation (useful to avoid messy extra logic for situations where it's
    unclear whether any transformation will be applied)"""

    return Transformation(f)
