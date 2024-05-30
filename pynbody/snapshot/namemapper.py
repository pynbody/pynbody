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
            if sys.version_info[0] == 2:
                if gadget_blocks:
                    a = a.upper().ljust(4)
            else:
                if gadget_blocks:
                    a = a.upper().ljust(4).encode("utf-8")
            if b in name_map:
                alternates[name_map[b]] = b
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

    This class is designed to be used in a context where the names of arrays in a simulation code may not be fully
    known in advance. For example, we might be unsure whether pynbody's ``pos`` array corresponds to the ``Coordinates``
    array or the ``Position`` array in a given simulation snapshot. This class allows us possible different mappings
    to be given in the configuration, and then once one of the mappings is used, it is locked in for the duration of
    the object's lifetime.

    """
    def __init__(self, config_name, gadget_blocks=False):
        self._name_map, self._rev_name_map, self._alternates = setup_name_maps(config_name, gadget_blocks,True)

    def _select_alternate_target(self,name):
        self._name_map[self._alternates[name]]=name
        self._rev_name_map[name]=self._alternates[name]

    def _select_alternate_target_if_required(self,name):
        if name not in list(self._name_map.values()) and name in self._alternates:
            self._select_alternate_target(name)

    def __call__(self, name, reverse=False):
        if reverse:
            self._select_alternate_target_if_required(name)
        current_map = self._rev_name_map if reverse else self._name_map
        return current_map.get(name, name)
