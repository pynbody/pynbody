"""

util
====

Utility functions for the snapshot module.

"""

from functools import reduce

from .. import array, units


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

    def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=True, convert_parent=True):
        """
        Converts all arrays' units to be consistent with the distance, velocity, mass basis units specified.

        Parameters
        ----------

        distance: string (default = 'kpc')
            The distance unit to convert to.

        velocity: string (default = 'km s^-1')
            The velocity unit to convert to.

        mass: string (default = 'Msol')
            The mass unit to convert to.

        persistent: boolean (default = True)
            Apply units change to future lazy-loaded arrays if True.

        convert_parent: boolean (default = True)
            Propagate units change from a halo catalogue to a parent snapshot. See note below.


        .. note::

            The option `convert_parent` is only applicable to :class:`Halo` objects. It is ignored by all other objects,
            including :class:`pynbody.snapshot.simsnap.SimSnap`, :class:`pynbody.snapshot.subsnap.SubSnap`,
            and :class:`pynbody.halo.HaloCatalogue` objects.

            When ``physical_units`` is called on a :class:`pynbody.halo.Halo` and `convert_parent` is True, no immediate
            action is taken on the :class:`pynbody.halo.Halo` itself; rather the request is passed upwards to the
            :class:`pynbody.halo.HaloCatalogue`.

            The catalogue object then calls ``physical_units`` on the parent snapshot and on all cached
            halos, setting ``convert_parent=False`` so that the units change is then applied to the
            :class:`pynbody.halo.Halo` object itself.

            This ensures that unit changes propagate through to properties of all halos. Most users will not need to
            worry about this subtlety; things should 'just work' if you ignore the `convert_parent` option.

        """

        dims = [units.Unit(x) for x in (distance, velocity, mass, 'a', 'h')]

        self._autoconvert_arrays(dims)
        self._autoconvert_properties(dims)

        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None
