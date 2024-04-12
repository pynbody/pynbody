"""Module for describing, applying and reverting Galilean transformations on simulations.

Generally it is not necessary to access this module directly, but it is used by
:class:`~pynbody.snapshot.simsnap.SimSnap` class to implement e.g.
:meth:`~pynbody.snapshot.simsnap.SimSnap.rotate_x` etc, and also by analysis
transformations such as :func:`~pynbody.analysis.halo.center` and
:func:`~pynbody.analysis.angmom.faceon` etc.

The main feature of this module is the :class:`Transformation` class, which performs
transformations of various kinds on simulations then reverts them.  You can use these objects
as context managers, which means you can write code like this:

>>> with f.translate(fshift) :
>>>     print(f['pos'][0])

which is equivalent to

>>> try:
>>>     f['pos']+=shift
>>>     print(f['pos'][0])
>>> finally:
>>>     f['pos']-=shift

You can also chain transformations together, like this:

>>> with f.translate(fshift).rotate_x(45):
>>>    print(f['pos'][0])

This implies a translation followed by a rotation. When reverting the transformation, they are
of course undone in the opposite order.
"""

from __future__ import annotations

import typing
import weakref

if typing.TYPE_CHECKING:
    from . import snapshot

import numpy as np

from . import util


class Transformable:
    """A mixin class for objects that can generate a Transformation object"""

    def translate(self, offset):
        """Translate by the given offset.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation."""
        return GenericTranslation(self, 'pos', offset, description="translate")

    def offset_velocity(self, offset):
        """Shift the velocity by the given offset.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation."""
        return GenericTranslation(self, 'vel', offset, description = "offset_velocity")

    def rotate_x(self, angle):
        """Rotates about the current x-axis by 'angle' degrees.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation."""
        angle_rad = angle * np.pi / 180
        return self.rotate(np.array([[1, 0, 0],
                                     [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                     [0, np.sin(angle_rad),  np.cos(angle_rad)]]),
                           description = f"rotate_x({angle})")

    def rotate_y(self, angle):
        """Rotates about the current y-axis by 'angle' degrees.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation."""
        angle_rad = angle * np.pi / 180
        return self.rotate(np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                     [0,                1,        0],
                                     [-np.sin(angle_rad),   0,   np.cos(angle_rad)]]),
                           description = f"rotate_y({angle})")

    def rotate_z(self, angle):
        """Rotates about the current z-axis by 'angle' degrees.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation."""
        angle_rad = angle * np.pi / 180
        return self.rotate(np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                     [np.sin(angle_rad),  np.cos(angle_rad), 0],
                                     [0,             0,        1]]),
                           description = f"rotate_z({angle})")

    def rotate(self, matrix, description = None):
        """Rotates using a specified matrix.

        Returns a :class:`pynbody.transformation.GenericTranslation` object which can be used
        as a context manager to ensure that the translation is undone.

        For more information, see the :mod:`pynbody.transformation` documentation.

        Parameters
        ----------

        matrix : array_like
            The 3x3 orthogonal matrix to rotate by

        description : str
            A description of the rotation to be returned from str() and repr()
        """
        return GenericRotation(self, matrix, description = description)

    @util.deprecated("This method is deprecated and will be removed in a future version. Use the rotate method instead.")
    def transform(self, matrix):
        """Deprecated alias for rotate"""
        return self.rotate(matrix)


class Transformation(Transformable):
    """The base class for all transformations.

    Note that this class inherits from :class:`Transformable`, so all transformations are themselves
    transformable. This means that you can chain transformations together, e.g. for a
    :class:`~pynbody.snapshot.simsnap.SimSnap` object *f*:

    >>> with f.translate(fshift).rotate_x(45):
    >>>    ...
    """

    def __init__(self, f, defer = False, description = None):
        """Initialise a transformation, and apply it if not explicitly deferred

        Parameters
        ----------
        f : SimSnap or Transformation
            The simulation or transformation to act on. If a transformation is given, this
            transformation will be chained to it, i.e. the result will represent the composed
            transformation.
        defer : bool
            If True, the transformation is not applied immediately. Otherwise, as soon as the object
            is constructed the transformation is applied to the simulation
        description : str
            A description of the transformation to be returned from str() and repr()
        """
        from . import snapshot
        if isinstance(f, snapshot.SimSnap):
            self.sim = f
            self.next_transformation = None
        elif isinstance(f, Transformation):
            self.sim = f.sim
            self.next_transformation = f
        else:
            raise TypeError("Transformation must either act on another Transformation or on a SimSnap")

        self._description = description

        self.applied = False
        if not defer:
            self.apply(force=False)

    @property
    def sim(self) -> snapshot.SimSnap | None:
        """The simulation to which this transformation applies"""
        return self._sim()

    @sim.setter
    def sim(self, sim):
        if sim is None:
            self._sim = lambda: None
        else:
            self._sim = weakref.ref(sim)

    def __repr__(self):
        return "<Transformation " + str(self) + ">"

    def __str__(self):
        if self.next_transformation is not None:
            s = str(self.next_transformation)+", "
        else:
            s = ""

        return s + self._describe()

    def _describe(self):
        if self._description:
            return self._description
        else:
            return self.__class__.__name__

    def apply_to(self, f: snapshot.SimSnap) -> snapshot.SimSnap:
        """Apply this transformation to a specified simulation.

        Chained transformations are applied recursively.

        Parameters
        ----------
        f : SimSnap
            The simulation to apply the transformation to. Any simulation reference stored within
            the transformation itself is ignored

        Returns
        -------
        SimSnap
            The input simulation (not a copy)
        """
        if self.next_transformation is not None:
            # this is a chained transformation, get the SimSnap to operate on
            # from the level below
            f = self.next_transformation.apply_to(f)

        self._apply(f)

        return f

    def apply_inverse_to(self, f):
        """Apply the inverse of this transformation to a specified simulation.

        Chained transformations are applied recursively.

        Parameters
        ----------
        f : SimSnap
            The simulation to apply the transformation to

        Returns
        -------
        SimSnap
            The input simulation (not a copy)
        """
        self._revert(f)

        if self.next_transformation is not None:
            self.next_transformation.apply_inverse_to(f)



    def apply(self, force=True):
        """Apply the transformation to the simulation it is associated with.

        This is either the simulation passed to the constructor or the one passed to the last
        transformation in the chain. If the transformation has already been applied, a RuntimeError
        is raised unless *force* is True, in which case the transformation is applied again.

        Parameters
        ----------

        force : bool
            If True, the transformation is applied even if it has already been applied. Otherwise,
            a RuntimeError is raised if the transformation has already been applied.

        Returns
        -------

        SimSnap
            The simulation after the transformation has been applied
        """

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
        """Revert the transformation. If it has not been applied, a RuntimeError is raised."""
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
        return self

    def __exit__(self, *args):
        self.revert()


class GenericTranslation(Transformation):
    """A translation on a specified array of a simulation"""

    def __init__(self, f, arname, shift, description = None):
        """Initialise a translation on a named array

        Parameters
        ----------
        f : Transformable
            The transformable to act on
        arname : str
            The name of the array to translate
        shift : array_like
            The shift to apply
        description : str
            A description of the translation to be returned from str() and repr()
        """
        self.shift = shift
        self.arname = arname
        super().__init__(f, description=description)

    def _apply(self, f):
        f[self.arname] += self.shift

    def _revert(self, f):
        f[self.arname] -= self.shift


class GenericRotation(Transformation):
    """A rotation on all 3d vectors in a simulation, by a given orthogonal 3x3 matrix"""

    def __init__(self, f, matrix, ortho_tol=1.e-8, description = None):
        """Initialise a rotation on a simulation.

        The matrix must be orthogonal to within *ortho_tol*.

        Parameters
        ----------
        f : SimSnap
            The simulation to act on

        matrix : array_like
            The 3x3 orthogonal matrix to rotate by

        ortho_tol : float
            The tolerance for orthogonality of the matrix. If the matrix is not orthogonal to within
            this tolerance, a ValueError is raised.

        description: str
            A description of the rotation to be returned from str() and repr()


        """
        # Check that the matrix is orthogonal
        resid = np.dot(matrix, np.asarray(matrix).T) - np.eye(3)
        resid = (resid ** 2).sum()
        if resid > ortho_tol or resid != resid:
            raise ValueError("Transformation matrix is not orthogonal")
        self.matrix = matrix
        super().__init__(f, description=description)

    def _apply(self, f):
        self._transform(self.matrix)

    def _revert(self, f):
        self._transform(self.matrix.T)

    def _transform(self, matrix):
        """Transforms the snapshot according to the 3x3 matrix given."""

        sim = self.sim

        # NB though it might seem more efficient to access _arrays and
        # _family_arrays directly, this would not work for SubSnaps.
        snapshot_keys = sim.keys()

        for array_name in snapshot_keys:
            ar = sim[array_name]
            if len(ar.shape) == 2 and ar.shape[1] == 3:
                ar[:] = np.dot(matrix, ar.transpose()).transpose()

        for fam in sim.families():
            family_keys = sim[fam].keys()
            family_keys_not_in_snapshot = set(family_keys) - set(snapshot_keys)
            for array_name in family_keys_not_in_snapshot:
                ar = sim[fam][array_name]
                if len(ar.shape) == 2 and ar.shape[1] == 3:
                    ar[:] = np.dot(matrix, ar.transpose()).transpose()


@util.deprecated("This function is deprecated and will be removed in a future version. Use the translate method of a SimSnap object instead.")
def translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    spatial *shift*.

    """
    return GenericTranslation(f, 'pos', shift)

@util.deprecated("This function is deprecated and will be removed in a future version. Use the translate method of a SimSnap object instead.")
def inverse_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the spatial
    vector *-shift*.

    For a fuller description, see *translate*"""
    return translate(f, -np.asarray(shift))

@util.deprecated("This function is deprecated and will be removed in a future version. Use the vel_translate method of a SimSnap object instead.")
def v_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    velocity *shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return GenericTranslation(f, 'vel', shift)

@util.deprecated("This function is deprecated and will be removed in a future version. Use the vel_translate method of a SimSnap object instead.")
def inverse_v_translate(f, shift):
    """Form a context manager for translating the simulation *f* by the given
    velocity *-shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return GenericTranslation(f, 'vel', -np.asarray(shift))

@util.deprecated("This function is deprecated and will be removed in a future version. Use sim.translate(...).vel_translate(...) instead.")
def xv_translate(f, x_shift, v_shift):
    """Form a context manager for translating the simulation *f* by the given
    position *x_shift* and velocity *v_shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""

    return translate(v_translate(f, v_shift), x_shift)


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
