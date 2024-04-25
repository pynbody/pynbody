"""
Filters are used to define subsets of simulations, especially (but not exclusively) spatial sub-regions.

The basic idea is that a :class:`Filter` object stores the abstract definition of the subset, and then it can be
called with a simulation to return a boolean array indicating which particles are in the subset. The implementation
of filters is designed to be as efficient as possible, and in many cases the selection is done in C with OpenMP
parallelisation. Additionally, if a simulation has a :class:`pynbody.kdtree.KDTree` built (via
:meth:`pynbody.snapshot.simsnap.SimSnap.build_tree`), then the selection can be done using the KDTree, which is considerably
faster for very large simulations.

Filters can be combined using the logical operators `&`, `|` and `~` to create more complex selections.

For a friendly introduction, see :doc:`/tutorials/filters`.
"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .. import family, units
from . import geometry_selection

if TYPE_CHECKING:
    from .. import snapshot

class Filter:
    """Base class for all filters. Filters are callables that take simulations as input and return a boolean mask"""

    def where(self, sim: snapshot.SimSnap):
        """Return the indices of particles that are in the filter.

        This is a convenience method that is equivalent to np.where(f(sim)) but may be more efficient for some filters.
        """
        return np.where(self(sim))

    def __call__(self, sim: snapshot.SimSnap):
        """Return a boolean mask indicating which particles are in the filter."""
        return np.ones(len(sim), dtype=bool)

    def __and__(self, f2):
        return And(self, f2)

    def __invert__(self):
        return Not(self)

    def __or__(self, f2):
        return Or(self, f2)

    def __repr__(self):
        return "Filter()"

    def __hash__(self):
        return hash(pickle.dumps(self))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for k, v in self.__dict__.items():
            if k not in other.__dict__:
                return False
            else:
                equal = other.__dict__[k]==v
                if isinstance(equal, np.ndarray):
                    equal = equal.all()
                if not equal:
                    return False

        return True

    def _get_wrap_in_position_units(self, sim):
        """Helper method to get the simulation box wrap in units of the position array.

        If the boxsize is undefined, returns -1.0"""
        sim_ancestor = sim.ancestor
        if 'boxsize' in sim.properties:
            wrap = sim_ancestor.properties['boxsize']
            if units.is_unit_like(wrap):
                wrap = float(wrap.in_units(sim_ancestor['pos'].units,
                                           **sim_ancestor.conversion_context()))
        else:
            wrap = -1.0

        return wrap

    def cubic_cell_intersection(self, centroids):
        """Compute the intersection with cubic cells with the specified centroids.

        This is currently used by the swift loader to figure out which cells to load"""
        raise NotImplementedError("Cell intersections are not implemented for this filter")

    @classmethod
    def _get_boxsize_and_delta_from_centroids(self, centroids):
        """Helper for calculating cubic cell intersections"""
        deltax = (centroids[1] - centroids[0]).max()
        # check we are interpreting this right: the ((deltax+max-min)/deltax)^3
        # should be the number of cells
        ncells = len(centroids)
        maxpos = centroids.max()
        minpos = centroids.min()
        boxsize = (deltax + maxpos - minpos)
        ncells_from_geometry = np.round(boxsize / deltax) ** 3
        assert ncells == ncells_from_geometry, "Geometry of cells doesn't match expectations"
        return boxsize, deltax

class FamilyFilter(Filter):
    """A filter that selects particles based on their family."""

    def __init__(self, family_):
        self.family = family.get_family(family_, False)

    def __repr__(self):
        return "FamilyFilter("+self.family.name+")"

    def __call__(self, sim):
        slice_ = sim._get_family_slice(self.family)
        flags = np.zeros(len(sim), dtype=bool)
        flags[slice_] = True
        return flags

class And(Filter):
    """A filter that selects particles that are in both of two other filters.

    You can construct this filter conveniently using the ``&`` operator, i.e.

    >>> f = f1 & f2

    returns a filter that selects particles that are in both ``f1`` and ``f2``.
    """

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def __call__(self, sim):
        return self.f1(sim) * self.f2(sim)

    def __repr__(self):
        return "(" + repr(self.f1) + " & " + repr(self.f2) + ")"

    def cubic_cell_intersection(self, centroids):
        return self.f1.cubic_cell_intersection(centroids) & \
               self.f2.cubic_cell_intersection(centroids)

class Or(Filter):
    """A filter that selects particles that are in either of two other filters.

    You can construct this filter conveniently using the ``|`` operator, i.e.

    >>> f = f1 | f2

    returns a filter that selects particles that are in either ``f1`` or ``f2``."""

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def __call__(self, sim):
        return self.f1(sim) + self.f2(sim)

    def __repr__(self):
        return "(" + repr(self.f1) + " | " + repr(self.f2) + ")"

    def cubic_cell_intersection(self, centroids):
        return self.f1.cubic_cell_intersection(centroids) | \
            self.f2.cubic_cell_intersection(centroids)


class Not(Filter):
    """A filter that selects particles that are not in another filter.

    You can construct this filter conveniently using the ``~`` operator, i.e.

    >>> f = ~f1

    returns a filter that selects particles that are not in ``f1``.
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, sim):
        x = self.f(sim)
        return np.logical_not(x)

    def __repr__(self):
        return "~" + repr(self.f)

    def cubic_cell_intersection(self, centroids):
        return ~self.f1.cubic_cell_intersection(centroids)


class Sphere(Filter):
    """
    A filter that selects particles within `radius` of the point `cen`.
    """

    def __init__(self, radius: float | str | units.UnitBase, cen: ArrayLike = (0, 0, 0)):
        """Create a sphere filter.

        Parameters
        ----------

        radius :
            The radius of the sphere. If a string, it is interpreted as a unit string.

        cen :
            The centre of the sphere. If a :class:`pynbody.snapshot.simsnap.SimArray`, units can be provided and
            will be correctly accounted for.
        """

        self.cen = np.asarray(cen)
        if self.cen.shape != (3,):
            raise ValueError("Centre must be length 3 array")

        if isinstance(radius, str):
            radius = units.Unit(radius)

        self.radius = radius

    def __call__(self, sim):
        with sim.immediate_mode:
            pos = sim['pos']

        cen, radius = self._get_cen_and_radius_as_float(pos)

        wrap = self._get_wrap_in_position_units(sim)

        return geometry_selection.selection(np.asarray(pos),'sphere',(cen[0], cen[1], cen[2], radius), wrap)

    def _get_cen_and_radius_as_float(self, pos):
        radius = self.radius
        cen = self.cen
        if units.has_units(cen):
            cen = cen.in_units(pos.units)
        if units.is_unit_like(radius):
            radius = float(radius.in_units(pos.units,
                                           **pos.conversion_context()))
        return cen, radius

    def where(self, sim):
        if hasattr(sim, "kdtree"):
            cen, radius = self._get_cen_and_radius_as_float(sim["pos"])
            return (np.sort(sim.kdtree.particles_in_sphere(cen, radius)),)
        else:
            return super().where(sim)

    def cubic_cell_intersection(self, centroids):
        boxsize, deltax = self._get_boxsize_and_delta_from_centroids(centroids)

        # the maximum offset from the cell centre to any corner:
        expand_distance = (deltax/2) * np.sqrt(3)

        return geometry_selection.selection(np.asarray(centroids), 'sphere',
                                           (self.cen[0], self.cen[1], self.cen[2], self.radius+expand_distance),
                                            boxsize)

    def __repr__(self):
        if units.is_unit(self.radius):
            return f"Sphere('{str(self.radius)}', {repr(self.cen)})"
        else:
            return f"Sphere({self.radius:.2e}, {repr(self.cen)})"


class Cuboid(Filter):
    """A filter that selects particles within a cuboid defined by two opposite corners."""

    def __init__(self, x1: float | str | units.UnitBase, y1: float | str | units.UnitBase = None,
                 z1: float | str | units.UnitBase = None, x2: float | str | units.UnitBase = None,
                 y2: float | str | units.UnitBase = None, z2: float | str | units.UnitBase = None):
        """Create a cuboid filter.

        If any of the cube coordinates ``x1``, ``y1``, ``z1``, ``x2``, ``y2``, ``z2`` are not specified they are
        determined as ``y1=x1``; ``z1=x1``; ``x2=-x1``; ``y2=-y1``; ``z2=-z1``.
        """
        x1, y1, z1, x2, y2, z2 = (
            units.Unit(x) if isinstance(x, str) else x for x in (x1, y1, z1, x2, y2, z2))
        if y1 is None:
            y1 = x1
        if z1 is None:
            z1 = x1
        if x2 is None:
            x2 = -x1
        if y2 is None:
            y2 = -y1
        if z2 is None:
            z2 = -z1
        self.x1, self.y1, self.z1, self.x2, self.y2, self.z2 = x1, y1, z1, x2, y2, z2

    def __call__(self, sim):
        wrap = self._get_wrap_in_position_units(sim)
        return self._get_mask(sim['pos'], self._get_boundaries(sim, wrap), wrap)

    def _get_mask(self, pos, boundaries, wrap):
        x1, y1, z1, x2, y2, z2 = boundaries
        return geometry_selection.selection(pos, 'cube', (x1, y1, z1, x2, y2, z2), wrap).view(dtype=bool)

    def _get_boundaries(self, sim, wrap=None):
        if wrap is None:
            wrap = self._get_wrap_in_position_units(sim)
        x1,y1,z1,x2,y2,z2 = (x.in_units(sim["pos"].units, **sim["pos"].conversion_context())
                            if units.is_unit_like(x) else x
                            for x in (self.x1, self.y1, self.z1, self.x2, self.y2, self.z2))
        if x2 < x1:
            x2 += wrap
        if y2 < y1:
            y2 += wrap
        if z2 < z1:
            z2 += wrap

        if x2 < x1 or y2 < y1 or z2 < z1:
            raise ValueError("Cuboid boundaries are not well defined")

        return x1, y1, z1, x2, y2, z2

    def _get_bounding_sphere(self, cuboid_boundaries):
        x1, y1, z1, x2, y2, z2 = cuboid_boundaries
        return ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2), np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) / 2

    def where(self, sim):

        if hasattr(sim, "kdtree"):
            # KDTree doesn't currently natively find cuboid regions, so we get the bounding sphere
            # and the search for the cuboid within that
            cuboid_boundaries = self._get_boundaries(sim)
            cen, radius = self._get_bounding_sphere(cuboid_boundaries)
            in_sphere_particles = sim.kdtree.particles_in_sphere(cen, radius)
            pos = sim["pos"][in_sphere_particles]
            wrap = self._get_wrap_in_position_units(sim)
            mask = self._get_mask(pos, cuboid_boundaries, wrap)
            return np.sort(in_sphere_particles[mask]),
        else:
            return super().where(sim)

    def cubic_cell_intersection(self, centroids):
        boxsize, deltax = self._get_boxsize_and_delta_from_centroids(centroids)
        shift = deltax/2
        boundaries = (self.x1 - shift, self.y1 - shift, self.z1 - shift,
                      self.x2 + shift, self.y2 + shift, self.z2 + shift)
        return self._get_mask(centroids, boundaries, boxsize)



    def __repr__(self):
        x1, y1, z1, x2, y2, z2 = ("'%s'" % str(x)
                                  if units.is_unit_like(x) else x
                                  for x in (self.x1, self.y1, self.z1, self.x2, self.y2, self.z2))
        return f"Cuboid({x1}, {y1}, {z1}, {x2}, {y2}, {z2})"


class Disc(Filter):
    """A filter that selects particles within a disc of specified extent and thickness."""

    def __init__(self, radius: float | str | units.UnitBase, height: float | str | units.UnitBase,
                 cen: ArrayLike = (0, 0, 0)):
        """Create a disc filter.

        In keeping with other parts of pynbody, the disc is defined in the x-y plane, with the z-axis being the
        symmetry axis. This is useful in conjunction with automated disc alignment e.g.
        :meth:`pynbody.analysis.angmom.sideon`.

        Parameters
        ----------

        radius :
            The radius of the disc (in the xy-plane). If a string, it is interpreted as a unit string.

        height :
            The thickness of the disc (in the z-direction). If a string, it is interpreted as a unit string.

        cen :
            The centre of the disc. If a :class:`pynbody.snapshot.simsnap.SimArray`, units can be provided and
            will be correctly accounted for.
        """
        self.cen = np.asarray(cen)
        if self.cen.shape != (3,):
            raise ValueError("Centre must be length 3 array")

        if isinstance(radius, str):
            radius = units.Unit(radius)

        if isinstance(height, str):
            height = units.Unit(height)

        self.radius = radius
        self.height = height

    def __call__(self, sim):
        radius = self.radius
        height = self.height

        if units.is_unit_like(radius):
            radius = float(
                radius.in_units(sim["pos"].units, **sim["pos"].conversion_context()))
        if units.is_unit_like(height):
            height = float(
                height.in_units(sim["pos"].units, **sim["pos"].conversion_context()))
        distance = (((sim["pos"] - self.cen)[:, :2]) ** 2).sum(axis=1)
        return (distance < radius ** 2) * (np.abs(sim["z"] - self.cen[2]) < height)

    def __repr__(self):
        radius = self.radius
        height = self.height

        radius, height = (
            ("'%s'" % str(x) if units.is_unit_like(x) else '%.2e' % x) for x in (radius, height))

        return f"Disc({radius}, {height}, {repr(self.cen)})"


class BandPass(Filter):
    """Selects particles in a bandpass of a named property"""

    def __init__(self, prop: str, min: float | str | units.UnitBase, max: float | str | units.UnitBase):
        """Create a bandpass filter.

        Parameters
        ----------

        prop :
            The name of the simulation array to filter on.

        min :
            The minimum value of the property. If a string, it is interpreted as a unit string.

        max :
            The maximum value of the property. If a string, it is interpreted as a unit string.
        """
        if isinstance(min, str):
            min = units.Unit(min)

        if isinstance(max, str):
            max = units.Unit(max)

        self._prop = prop
        self._min = min
        self._max = max

    def __call__(self, sim):
        min_ = self._min
        max_ = self._max
        prop = self._prop

        if units.is_unit_like(min_):
            min_ = float(
                min_.in_units(sim[prop].units, **sim.conversion_context()))
        if units.is_unit_like(max_):
            max_ = float(
                max_.in_units(sim[prop].units, **sim.conversion_context()))

        return ((sim[prop] > min_) * (sim[prop] < max_))

    def __repr__(self):
        min_, max_ = (("'%s'" % str(x) if units.is_unit_like(
            x) else '%.2e' % x) for x in (self._min, self._max))
        return f"BandPass('{self._prop}', {min_}, {max_})"


class HighPass(Filter):
    """Selects particles exceeding a specified threshold of a named property
    """

    def __init__(self, prop: str, min: float | str | units.UnitBase):
        """Create a high-pass filter.

        Parameters
        ----------

        prop :
            The name of the simulation array to filter on.

        min :
            The minimum value of the property. If a string, it is interpreted as a unit string.
        """
        if isinstance(min, str):
            min = units.Unit(min)

        self._prop = prop
        self._min = min

    def __call__(self, sim):
        min_ = self._min

        prop = self._prop

        if units.is_unit_like(min_):
            min_ = float(
                min_.in_units(sim[prop].units, **sim.conversion_context()))

        return (sim[prop] > min_)

    def __repr__(self):
        min = ("'%s'" % str(self._min) if units.is_unit_like(
            self._min) else '%.2e' % self._min)
        return f"HighPass('{self._prop}', {min})"


class LowPass(Filter):

    """Selects particles below a specified threshold of a named property
    """

    def __init__(self, prop: str, max: float | str | units.UnitBase):
        """Create a low-pass filter.

        Parameters
        ----------

        prop :
            The name of the simulation array to filter on.

        max :
            The maximum value of the property. If a string, it is interpreted as a unit string.
        """
        if isinstance(max, str):
            max = units.Unit(max)

        self._prop = prop
        self._max = max

    def __call__(self, sim):
        max_ = self._max

        prop = self._prop

        if units.is_unit_like(max_):
            max_ = float(
                max_.in_units(sim[prop].units, **sim.conversion_context()))

        return (sim[prop] < max_)

    def __repr__(self):
        max = ("'%s'" % str(self._max) if isinstance(
            self._max, units.UnitBase) else '%.2e' % self._max)
        return f"LowPass('{self._prop}', {max})"


class Annulus(And):
    """A filter that selects particles in between two spheres specified by radii `r1` and `r2` centered on `cen`."""

    def __init__(self, r1: float | str | units.UnitBase, r2: float | str | units.UnitBase, cen: ArrayLike = (0, 0, 0)):
        """Create an annulus filter.

        Parameters
        ----------

        r1 :
            The inner radius of the annulus. If a string, it is interpreted as a unit string.

        r2 :
            The outer radius of the annulus. If a string, it is interpreted as a unit string.

        cen :
            The centre of the annulus. If a :class:`pynbody.snapshot.simsnap.SimArray`, units can be provided and
            will be correctly accounted for.
        """
        super().__init__(~Sphere(r1, cen), Sphere(r2, cen))


class SolarNeighborhood(And):
    """A filter that selects particles in a disc between 2d radii `r1` and `r2` and thickness `height`.

    As for :class:`Disc`, the galaxy disc is defined in the x-y plane, with the z-axis being the symmetry axis.

    Default parameters are provided that are approximately the solar neighborhood (coarsely selected).
    """

    def __init__(self, r1: float | str | units.UnitBase = units.Unit("5 kpc"),
                 r2: float | str | units.UnitBase = units.Unit("10 kpc"),
                 height: float | str | units.UnitBase = units.Unit("2 kpc"),
                 cen: ArrayLike = (0, 0, 0)):
        """Create a solar neighborhood filter.

        Parameters
        ----------

        r1 :
            The inner radius of the disc. If a string, it is interpreted as a unit string.

        r2 :
            The outer radius of the disc. If a string, it is interpreted as a unit string.

        height :
            The thickness of the disc. If a string, it is interpreted as a unit string.

        cen :
            The centre of the disc. If a :class:`pynbody.snapshot.simsnap.SimArray`, units can be provided and
            will be correctly accounted for.
        """
        super().__init__(~Disc(r1, height, cen), Disc(r2, height, cen))
