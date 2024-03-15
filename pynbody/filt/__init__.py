"""

filt
====

Defines and implements 'filters' which allow abstract subsets
of data to be specified.

See the `filter tutorial
<http://pynbody.github.io/pynbody/tutorials/filters.html>`_ for some
sample usage.

"""


import pickle

import numpy as np

from .. import family, units
from . import geometry_selection


class Filter:

    def __init__(self):
        self._descriptor = "filter"
        pass

    def where(self, sim):
        return np.where(self(sim))

    def __call__(self, sim):
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
    def __init__(self, family_):
        assert isinstance(family_, family.Family)
        self._descriptor = family_.name
        self.family = family_

    def __repr__(self):
        return "FamilyFilter("+self.family.name+")"

    def __call__(self, sim):
        slice_ = sim._get_family_slice(self.family)
        flags = np.zeros(len(sim), dtype=bool)
        flags[slice_] = True
        return flags

class And(Filter):

    def __init__(self, f1, f2):
        self._descriptor = f1._descriptor + "&" + f2._descriptor
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

    def __init__(self, f1, f2):
        self._descriptor = f1._descriptor + "|" + f2._descriptor
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

    def __init__(self, f):
        self._descriptor = "~" + f._descriptor
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
    Return particles that are within `radius` of the point `cen`.

    Inputs:
    -------

    *radius* : extent of the sphere. Can be a number or a string specifying the units.

    *cen* : center of the sphere. default = (0,0,0)
    """

    def __init__(self, radius, cen=(0, 0, 0)):
        self._descriptor = "sphere"
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

    """Create a cube with specified edge coordinates. If any of the cube
    coordinates `x1`, `y1`, `z1`, `x2`, `y2`, `z2` are not specified
    they are determined as `y1=x1;` `z1=x1;` `x2=-x1;` `y2=-y1;`
    `z2=-z1`.

    """

    def __init__(self, x1, y1=None, z1=None, x2=None, y2=None, z2=None):

        self._descriptor = "cube"
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

    """
    Return particles that are within a disc of extent `radius` and
    thickness `height` centered on `cen`.
    """

    def __init__(self, radius, height, cen=(0, 0, 0)):
        self._descriptor = "disc"
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

    """
    Return particles whose property `prop` is within `min` and `max`,
    which can be specified as unit strings.
    """

    def __init__(self, prop, min, max):
        self._descriptor = "bandpass_" + prop

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

    """
    Return particles whose property `prop` exceeds `min`, which can be
    specified as a unit string.
    """

    def __init__(self, prop, min):
        self._descriptor = "highpass_" + prop

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

    """Return particles whose property `prop` is less than `max`, which can be
    specified as a unit string.
    """

    def __init__(self, prop, max):
        self._descriptor = "lowpass_" + prop

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


def Annulus(r1, r2, cen=(0, 0, 0)):
    """
    Convenience function that returns a filter which selects particles
    in between two spheres specified by radii `r1` and `r2` centered
    on `cen`.
    """

    x = Sphere(max(r1, r2), cen) & ~Sphere(min(r1, r2), cen)
    x._descriptor = "annulus"
    return x


def SolarNeighborhood(r1=units.Unit("5 kpc"), r2=units.Unit("10 kpc"), height=units.Unit("2 kpc"), cen=(0, 0, 0)):
    """
    Convenience function that returns a filter which selects particles
    in a disc between radii `r1` and `r2` and thickness `height`.
    """

    x = Disc(max(r1, r2), height, cen) & ~Disc(min(r1, r2), height, cen)
    x._descriptor = "Solar Neighborhood"
    return x
