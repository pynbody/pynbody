"""Tool for mapping from individual simulation code's naming conventions to pynbody naming conventions.

This module is a detail that is not intended to be used directly by end users. It is used by the snapshot classes to
map between the names of arrays in different simulation codes.
"""

import configparser
import sys

from .. import config_parser


def setup_name_maps(config_name, gadget_blocks=False, with_alternates=False):
    name_map = {}
    rev_name_map = {}
    alternates = {}
    try:
        for a, b in config_parser.items(config_name):
            if gadget_blocks:
                a = a.upper().ljust(4).encode("utf-8")

            b_alternates = alternates.get(b, [])
            b_alternates.append(a)
            alternates[b] = b_alternates

            rev_name_map[a] = b
            name_map[b] = a

    except configparser.NoOptionError:
        pass

    if with_alternates:
        return name_map, rev_name_map, alternates
    else:
        return name_map, rev_name_map


def name_map_function(name_map, rev_name_map):
    def _translate_array_name(name, reverse=False):
        try:
            if reverse:
                return rev_name_map[name]
            else:
                return name_map[name]
        except KeyError:
            return name

    return _translate_array_name


class AdaptiveNameMapper:
    """A class to map between the names of arrays in different simulation codes.

    This class is designed to be used in a context where the names of arrays in a simulation code (a 'format name') may
    not be fully known in advance. For example, we might be unsure whether pynbody's ``pos`` array corresponds to the
    format name ``Coordinates`` or  ``Position`` in a given simulation snapshot. This class allows us possible different
    mappings to be given in the configuration, and then once one of the mappings is used, it is locked in for the
    duration of the object's lifetime.

    One may wish to set return_all_format_names to True to just get all possible format names for a given pynbody name.

    """
    def __init__(self, config_name, gadget_blocks=False, return_all_format_names=False):
        """Create a new AdaptiveNameMapper object.

        Parameters
        ----------

        config_name : str
            The name of the section in the configuration file to use for the mapping.

        gadget_blocks : bool
            If True, the mapping is case-insensitive and pads the names to 4 characters, as is the convention for GADGET
            block names.

        return_all_format_names : bool
            If True, return all the possible format-specific names for a pynbody name, rather than only the most
            recently accessed (which is the default behaviour).

        Notes
        -----

        Generally, return_all_format_names = False is the obvious choice, where we don't know in advance the name of the
        array in the simulation snapshot, but we know there will be only one such name. However, swift uses a different
        name for the masses of its black holes (``SubgridMasses``) than for the masses of its other particles
        (``Masses``), so in this case we need to set ``allow_ambiguous = True``. There may be other cases where this
        is necessary.
        """
        self._pynbody_to_format_map, self._format_to_pynbody_map, self._pynbody_to_all_format_map = setup_name_maps(config_name, gadget_blocks, True)
        self._return_all_format_names = return_all_format_names

    def _select_alternate_target_if_required(self, name):
        if name not in list(self._pynbody_to_format_map.values()):
            for k, v in self._pynbody_to_all_format_map.items():
                if name in v:
                    self._pynbody_to_format_map[k] = name


    def __call__(self, name, reverse=False):
        """Map a pynbody name to a format name.

        If reverse=True, the mapping is done in the reverse direction (i.e. from format name to pynbody name).

        If return_all_format_names=True, when reverse=False, this function returns a list of all possible format names
        """

        result_if_no_match = name
        if reverse:
            self._select_alternate_target_if_required(name)
            current_map = self._format_to_pynbody_map
        else:
            if self._return_all_format_names:
                current_map = self._pynbody_to_all_format_map
                result_if_no_match = [name]
            else:
                current_map = self._pynbody_to_format_map

        return current_map.get(name, result_if_no_match)
