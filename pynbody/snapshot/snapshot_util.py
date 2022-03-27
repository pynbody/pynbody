"""

util
====

Utility functions for the snapshot module.

"""

import logging
from functools import reduce

from .. import array, units

logger = logging.getLogger('pynbody.snapshot')

class ContainerWithPhysicalUnitsOption:
    """
    Defines an abstract class that has properties and arrays
    that can be converted to physical units.
    """
    _autoconvert = None
    _units_conversion_cache = {}

    @classmethod
    def _cached_unit_conversion(cls, from_unit, dims, ucut=3):
        key = (
            from_unit.dimensionality_as_string(),
            tuple(dims),
            ucut,
        )
        if key in cls._units_conversion_cache:
            return cls._units_conversion_cache[key]

        try:
            d = from_unit.dimensional_project(dims)
        except units.UnitsException:
            cls._units_conversion_cache[key] = None
            return

        new_unit = reduce(
            lambda x, y: x * y,
            [a ** b for a, b in zip(dims, d[:ucut])]
        )
        cls._units_conversion_cache[key] = new_unit

        if new_unit is not None and new_unit != from_unit:
            logger.info("Converting units from %s to %s" %
                        (from_unit, new_unit))
            return new_unit

    def _get_dims(self, dims=None):
        if dims is None:
            return self.ancestor._autoconvert
        else:
            return dims

    def _autoconvert_array_unit(self, ar, dims=None, ucut=3):
        """Given an array ar, convert its units such that the new units span
        dims[:ucut]. dims[ucut:] are evaluated in the conversion (so will be things like
        a, h etc).

        If dims is None, use the internal autoconvert state to perform the conversion."""
        dims = self._get_dims(dims)
        if dims is None:
            return

        if ar.units is None or isinstance(ar.units,units.NoUnit):
            return

        new_unit = self._cached_unit_conversion(ar.units, dims, ucut=ucut)
        if new_unit is not None:
            logger.info("Converting %s units from %s to %s" %
                        (ar.name, ar.units, new_unit))
            ar.convert_units(new_unit)


    def _autoconvert_properties(self, dims=None):
        dims = self._get_dims(dims)
        if dims is None:
            return

        for k, v in list(self.properties.items()):
            if isinstance(v, units.UnitBase):
                new_unit = self._cached_unit_conversion(v, dims, ucut=3)
                if new_unit is not None:
                    new_unit *= v.ratio(new_unit, **self.conversion_context())
                    self.properties[k] = new_unit
            elif isinstance(v, array.SimArray):
                self._autoconvert_array_unit(v, dims)

    def _autoconvert_arrays(self, dims=None):
        dims = self._get_dims(dims)
        if dims is None:
            return

        all = list(self._arrays.values())
        for x in self._family_arrays:
            all += list(self._family_arrays[x].values())

        for ar in all:
            self._autoconvert_array_unit(ar.ancestor, dims)

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=False):
        """
        Converts all array's units to be consistent with the
        distance, velocity, mass basis units specified.

        Base units can be specified using keywords.

        **Optional Keywords**:

           *distance*: string (default = 'kpc')

           *velocity*: string (default = 'km s^-1')

           *mass*: string (default = 'Msol')

           *persistent*: boolean (default = True); apply units change to future lazy-loaded arrays if True

           *convert_parent*: boolean (default = None); ignored for SimSnap objects

        """
        dims = [units.Unit(x) for x in (distance, velocity, mass, 'a', 'h')]

        self._autoconvert_arrays(dims)
        self._autoconvert_properties(dims)

        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None
